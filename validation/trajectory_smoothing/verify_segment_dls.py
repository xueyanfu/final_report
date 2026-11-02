# 实验14独立验证：保存曲线重放、MDH复核、系数Jacobian与解析导数检查。
# 使用不同于优化/绘图的采样网格，核对固定节点、限位和采样jerk预算。
# 来源：用户MDH参数实现rizon4_mdh.py；原样条与区段DLS接口复用项目代码。
"""输出JSON验证证据，不生成Markdown。

MDH FK与Pinocchio FK分别计算，避免只用求解器内部残差验证自身。
区段DLS复用来源及相关研究见experiments/trajectory_smoothing/segment_dls.py。
SciPy样条API：https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
import numpy as np
import pinocchio as pin
from experiments.common.rizon4_analysis_common import Robot
from experiments.inverse_kinematics.rizon4_ik import frame_pose
from experiments.kinematics.rizon4_mdh import fk
from experiments.trajectory_tracking.experiments_07_heart_trajectory import reference_position
from experiments.trajectory_smoothing.smooth_trajectory import JointCurve
from experiments.trajectory_smoothing.segment_dls import (
    SegmentCurve, SegmentOptions, base_margin, segment_system,
)

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def load_curve(path):
    with np.load(path, allow_pickle=False) as data:
        base = JointCurve(data['times'].copy(), data['angles'].copy(), 'quintic')
        curve = SegmentCurve(base, data['coefficients'].copy())
        target = pin.SE3(data['rotation'].copy(), data['translation'].copy())
        return curve, target, float(data['scale'])


def new_grid_check(robot, curve, target, scale, grid):
    ep, er = [], []
    matrix_difference = 0.
    sampled_margin = np.inf
    for t in grid:
        q = curve.values(t)
        mdh = fk(q)
        pin_pose = frame_pose(robot, q)
        matrix_difference = max(matrix_difference, float(np.abs(mdh-pin_pose.homogeneous).max()))
        desired = reference_position(t, target.translation, scale, curve.duration)
        ep.append(np.linalg.norm(mdh[:3, 3]-desired))
        er.append(np.linalg.norm(pin.log3(mdh[:3, :3].T @ target.rotation)))
        sampled_margin = min(sampled_margin, float(np.minimum(q-robot.lower, robot.upper-q).min()))
    require(matrix_difference < 1e-12, 'Independent MDH and Pinocchio disagree')
    require(sampled_margin >= 0, 'Sampled joint-limit violation')
    return dict(position_max_um=float(np.max(ep)*1e6),
                position_rmse_um=float(np.sqrt(np.mean(np.square(ep)))*1e6),
                rotation_max_rad=float(np.max(er)),
                mdh_pin_matrix_max=matrix_difference, sampled_joint_margin_rad=sampled_margin)


def check_jacobian_and_derivatives(robot, curve, target, scale, rng, weight):
    max_jacobian_error = max_velocity_error = max_acceleration_error = 0.
    indices = np.unique(np.r_[rng.integers(0, len(curve.times)-1, 2),
                               np.argmax(np.abs(curve.coefficients).max(axis=(1, 2)))])
    for index in indices:
        # 显式加入非零系数，防止仅在零修正状态验证链式矩阵。
        C = curve.coefficients[index].copy() + rng.normal(0, 2e-6, curve.coefficients[index].shape)
        fractions = np.array([.173, .419, .763])
        _, analytic, _, _ = segment_system(robot, target, curve.base, index, C,
                                           fractions, scale, weight, True)
        finite_difference = np.zeros_like(analytic)
        epsilon = 1e-6
        for column in range(C.size):
            delta = np.zeros_like(C)
            delta.flat[column] = epsilon
            plus = segment_system(robot, target, curve.base, index, C+delta,
                                   fractions, scale, weight, False)[0]
            minus = segment_system(robot, target, curve.base, index, C-delta,
                                    fractions, scale, weight, False)[0]
            finite_difference[:, column] = (plus-minus)/(2*epsilon)
        max_jacobian_error = max(max_jacobian_error, float(np.abs(analytic-finite_difference).max()))
        width = curve.times[index+1]-curve.times[index]
        epsilon_t = min(1e-5, width*1e-3)
        for fraction in fractions:
            t = curve.times[index]+fraction*width
            v_fd = (curve.values(t+epsilon_t)-curve.values(t-epsilon_t))/(2*epsilon_t)
            a_fd = (curve.values(t+epsilon_t, 1)-curve.values(t-epsilon_t, 1))/(2*epsilon_t)
            max_velocity_error = max(max_velocity_error, float(np.abs(v_fd-curve.values(t, 1)).max()))
            max_acceleration_error = max(max_acceleration_error, float(np.abs(a_fd-curve.values(t, 2)).max()))
    require(max_jacobian_error < 2e-7, 'Coefficient residual Jacobian finite difference failed')
    require(max_velocity_error < 2e-7, 'Analytic velocity finite difference failed')
    require(max_acceleration_error < 2e-6, 'Analytic acceleration finite difference failed')
    return dict(jacobian_max_difference=max_jacobian_error,
                velocity_fd_max_difference=max_velocity_error,
                acceleration_fd_max_difference=max_acceleration_error)


def jerk_checks(curve, opts):
    grid = np.linspace(0, curve.duration, 8003)
    base_jerk = curve.base.spline(grid, 3)
    jerk = curve.values(grid, 3)
    base_rms = float(np.sqrt(np.mean(base_jerk**2)))
    base_peak = float(np.abs(base_jerk).max())
    rms = float(np.sqrt(np.mean(jerk**2)))
    peak = float(np.abs(jerk).max())
    require(rms <= opts.max_jerk_rms_ratio*base_rms+1e-10, 'New-grid jerk RMS budget exceeded')
    require(peak <= opts.max_jerk_peak_ratio*base_peak+1e-10, 'New-grid jerk peak budget exceeded')
    # C2不保证三阶导数连续；额外报告每段两端的左右极限，不冒充连续极值。
    boundary_peak = 0.
    max_jump = 0.
    left_at_nodes, right_at_nodes = [], []
    for i, width in enumerate(np.diff(curve.times)):
        left = curve.base.spline(curve.times[i], 3)
        right = curve.base.spline(curve.times[i+1], 3)
        left += np.array([b.deriv(3)(0.) for b in curve.basis]) @ curve.coefficients[i]/width**3
        right += np.array([b.deriv(3)(1.) for b in curve.basis]) @ curve.coefficients[i]/width**3
        right_at_nodes.append(left)
        left_at_nodes.append(right)
        boundary_peak = max(boundary_peak, float(np.abs(left).max()), float(np.abs(right).max()))
    for i in range(len(left_at_nodes)-1):
        max_jump = max(max_jump, float(np.abs(left_at_nodes[i]-right_at_nodes[i+1]).max()))
    return dict(jerk_samples=len(grid), jerk_rms_rad_s3=rms, jerk_peak_rad_s3=peak,
                jerk_rms_ratio=rms/base_rms, jerk_peak_ratio=peak/base_peak,
                knot_one_sided_jerk_peak_rad_s3=boundary_peak,
                knot_jerk_jump_max_rad_s3=max_jump)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/segment_dls')
    args = parser.parse_args()
    folder = args.results
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    opts = SegmentOptions(**meta['options'])
    opts.validate()
    for path, digest in meta['source_sha256'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest,
                f'Source differs from experimental record: {path}')
    robot = Robot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
    if 'model_sha256' in meta:
        require(hashlib.sha256(robot.path.read_bytes()).hexdigest() == meta['model_sha256'], 'Model hash mismatch')
    with (folder/'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        summary = list(csv.DictReader(f))
    expected = {(case, method) for case in ['standard', 'finer', 'larger']
                for method in ['base', 'two_profiles', 'segment_dls']}
    require(len(summary) == 9 and {(r['case'], r['method']) for r in summary} == expected,
            'Expected nine unique case/method rows')
    rng = np.random.default_rng(20261102)
    checks = []
    for case, expected_nodes, duration, expected_scale in [
            ('standard', 41, 10., .004), ('finer', 81, 10., .004), ('larger', 41, 10., .005)]:
        base, target, scale = load_curve(folder/f'{case}_base.npz')
        require(len(base.times) == expected_nodes and base.duration == duration,
                f'Unexpected case parameters: {case}')
        require(scale == expected_scale and not np.any(base.coefficients), 'Invalid baseline')
        grid = np.unique(np.r_[np.linspace(0, duration, 2003), rng.uniform(0, duration, 300)])
        baseline = new_grid_check(robot, base, target, scale, grid)
        for method in ['base', 'two_profiles', 'segment_dls']:
            curve, T, current_scale = load_curve(folder/f'{case}_{method}.npz')
            require(np.array_equal(curve.times, base.times) and np.array_equal(curve.angles, base.angles),
                    'Methods do not preserve identical IK nodes')
            require(np.array_equal(T.homogeneous, target.homogeneous) and current_scale == scale,
                    'Methods use different Cartesian targets')
            error = new_grid_check(robot, curve, T, scale, grid)
            require(error['position_max_um'] <= baseline['position_max_um']+1e-6,
                    f'Independent maximum position error worsened: {case}/{method}')
            require(error['position_rmse_um'] <= baseline['position_rmse_um']+1e-6,
                    f'Independent position RMSE worsened: {case}/{method}')
            require(error['rotation_max_rad'] <= baseline['rotation_max_rad']+1e-12,
                    f'Independent maximum rotation error worsened: {case}/{method}')
            node_errors = [float(np.abs(curve.values(curve.times, k)-base.base.spline(base.times, k)).max())
                           for k in range(3)]
            require(max(node_errors) < 1e-9, 'Node position/velocity/acceleration changed')
            # 包含每段右端的解析极限，避免右连续选择掩盖左端值。
            endpoint_basis_error = max(abs(b.deriv(k)(s)) for b in curve.basis
                                       for k in range(3) for s in [0., 1.])
            require(endpoint_basis_error < 1e-8, 'Bubble basis does not lock second-order endpoints')
            margin = base_margin(robot, curve.base)-float(np.abs(curve.coefficients).sum(axis=1).max())
            require(margin > 0, 'Conservative joint-limit bound failed')
            row = next(r for r in summary if r['case'] == case and r['method'] == method)
            require(int(row['nodes']) == expected_nodes, 'Summary node count mismatch')
            require(abs(float(row['conservative_joint_margin_rad'])-margin) < 1e-10,
                    'Summary joint margin mismatch')
            derivatives = check_jacobian_and_derivatives(robot, curve, T, scale, rng, opts.rotation_weight)
            jerk = jerk_checks(curve, opts)
            checks.append(dict(case=case, method=method, independent_pose_samples=len(grid),
                               node_q_difference_rad=node_errors[0], node_v_difference_rad_s=node_errors[1],
                               node_a_difference_rad_s2=node_errors[2],
                               conservative_joint_margin_rad=margin, **error, **derivatives, **jerk))
    evidence = dict(passed=True, seed=20261102, source_hashes_checked=len(meta['source_sha256']),
                    cases=checks, verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mdh_source_sha256=hashlib.sha256((ROOT/'experiments/kinematics/rizon4_mdh.py').read_bytes()).hexdigest(),
                    scope='Finite independent pose and jerk samples; no continuous Cartesian/jerk bound or dynamic tracking claim')
    (folder/'verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
