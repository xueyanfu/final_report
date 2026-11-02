# 实验15：区段DLS的MuJoCo动力学跟踪对照。
# 固定控制器、增益、时间律和步长，对照标准41、放大41和稀疏21节点。
# 复用实验10 simulate、实验14 segment_dls；保存完整轨迹和误差统计。
"""来源：项目实验10动力学控制器、实验14区段修正器、实验07心形目标。

Pinocchio IK上游及SciPy来源见segment_dls.py文件头；模型来源见MODEL_SOURCE.txt。
本实验新增动力学对照与误差分层统计，未修改原控制律或模型参数。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import hashlib
import platform
import time
import numpy as np
import pinocchio as pin
import mujoco
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common.rizon4_analysis_common import Robot, NAMES
from experiments.inverse_kinematics.rizon4_ik import frame_pose
from experiments.trajectory_tracking.experiments_07_heart_trajectory import reference_position, save
from experiments.trajectory_tracking.experiments_09_heart_tracking_comparison import check_dynamics
from experiments.trajectory_smoothing.experiments_10_smooth_tracking import simulate
from experiments.trajectory_smoothing.smooth_trajectory import JointCurve, refine_nodes
from experiments.trajectory_smoothing.segment_dls import SegmentCurve, correct_segments, SegmentOptions, base_margin
from experiments.trajectory_smoothing.experiments_14_segment_dls import measure

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/segment_tracking')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    robot = Robot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
    dynamics = check_dynamics(robot)
    summary, primary, source_inputs = [], {}, {}
    cases = [('standard', 41, .004), ('larger', 41, .005), ('coarse', 21, .004)]
    for case, nodes, scale in cases:
        if case != 'coarse':
            path = ROOT/f'results/segment_dls/{case}_segment_dls.npz'
            with np.load(path, allow_pickle=False) as data:
                base = JointCurve(data['times'], data['angles'], 'quintic')
                T = pin.SE3(data['rotation'], data['translation'])
                corrected = SegmentCurve(base, data['coefficients'].copy())
            source_inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            T = frame_pose(robot, np.array([0., 0, 0, 1.57, 0, 0, 0]))
            ts = np.linspace(0, 10., nodes)
            base = JointCurve(ts, refine_nodes(robot, T, ts, scale), 'quintic')
            corrected, history, detail = correct_segments(robot, T, base, scale, SegmentOptions())
            save(args.out/'coarse_optimization.csv', history)
            (args.out/'coarse_optimization.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
        original = SegmentCurve(base, np.zeros_like(corrected.coefficients))
        for name, curve in [('base', original), ('segment_dls', corrected)]:
            start = time.perf_counter()
            reference, _, _ = measure(robot, T, curve, scale, np.linspace(0, 10., 16001))
            rows, metrics = simulate(robot, T, curve, scale)
            # 同步分解：实际-目标 = (实际-参考FK) + (参考FK-目标)。
            for row in rows:
                qref = np.array([row[f'qref_{j}'] for j in NAMES])
                # simulate记录的qref是本步开始的指令；同步参考必须按步进后时间重算。
                sync_q = curve.evaluate(row['time_s'])[0]
                ref = robot.position(sync_q)
                actual = np.array([row[f'actual_{c}'] for c in 'xyz'])
                target = np.array([row[f'target_{c}'] for c in 'xyz'])
                row['actual_to_reference_m'] = float(np.linalg.norm(actual-ref))
                row['reference_to_target_m'] = float(np.linalg.norm(ref-target))
            draw = [r for r in rows if r['phase'] == 'draw']
            servo_rmse = np.sqrt(np.mean([r['actual_to_reference_m']**2 for r in draw]))*1e6
            geometry_rmse = np.sqrt(np.mean([r['reference_to_target_m']**2 for r in draw]))*1e6
            result = dict(case=case, method=name, nodes=nodes, duration_s=10., scale=scale,
                          reference_position_max_um=reference['position_max_um'],
                          reference_position_rmse_um=reference['position_rmse_um'],
                          tracking_rmse_um=metrics['rmse_mm']*1000,
                          tracking_max_um=metrics['max_error_mm']*1000,
                          actual_to_reference_rmse_um=float(servo_rmse),
                          synchronous_reference_rmse_um=float(geometry_rmse),
                          draw_samples=len(draw), total_samples=len(rows),
                          conservative_reference_margin_rad=float(base_margin(robot, base)-np.abs(curve.coefficients).sum(axis=1).max()),
                          evaluation_wall_s=time.perf_counter()-start, **metrics)
            if result['conservative_reference_margin_rad'] < 0:
                raise RuntimeError('Reference joint limit violated')
            np.savez_compressed(args.out/f'{case}_{name}.npz', times=base.times, angles=base.angles,
                                coefficients=curve.coefficients, rotation=T.rotation, translation=T.translation, scale=scale)
            save(args.out/f'{case}_{name}.csv', rows)
            summary.append(result)
            if case == 'coarse':
                primary[name] = rows
            print(json.dumps(result), flush=True)
    save(args.out/'summary.csv', summary)
    sources = [str(Path(__file__).relative_to(ROOT)), 'experiments/trajectory_smoothing/segment_dls.py',
               'experiments/trajectory_smoothing/experiments_10_smooth_tracking.py',
               'experiments/trajectory_smoothing/smooth_trajectory.py',
               'experiments/trajectory_smoothing/experiments_14_segment_dls.py',
               'experiments/trajectory_tracking/experiments_09_heart_tracking_comparison.py',
               'experiments/trajectory_tracking/experiments_07_heart_trajectory.py',
               'experiments/inverse_kinematics/rizon4_ik.py', 'experiments/common/rizon4_analysis_common.py']
    meta = dict(cases=cases, controller='unchanged experiment10 velocity + inverse-dynamics feedforward with position servos',
                dt_s=float(robot.mj.opt.timestep), hold_s=1., source_inputs_sha256=source_inputs,
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                model_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
                python=platform.python_version(), pinocchio=pin.__version__, mujoco=mujoco.__version__,
                numpy=np.__version__, scipy=scipy.__version__, dynamics_check=dynamics,
                design='Cases fixed before simulation; no controller tuning or selection by observed gain',
                scope='Deterministic model simulation, not hardware; per-method geometry/servo RMSE cannot be added as scalars')
    (args.out/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(3)
    for name, label, offset, color in [('base', '原精化五次', -.19, '#999999'), ('segment_dls', '区段DLS', .19, '#22805c')]:
        g = [r for r in summary if r['method'] == name]
        for ax, field in zip(axes[:2], ['reference_position_rmse_um', 'tracking_rmse_um']):
            bars = ax.bar(x+offset, [r[field] for r in g], .38, label=label, color=color)
            ax.bar_label(bars, fmt='%.3f', fontsize=9)
            ax.margins(y=.25)
        rows = [r for r in primary[name] if r['phase'] == 'draw']
        axes[2].plot([r['time_s'] for r in rows], [r['position_error_m']*1e6 for r in rows], label=label, color=color)
    for ax, title in zip(axes[:2], ['规划位置RMSE（μm）', '动力学跟踪RMSE（μm）']):
        ax.set(xticks=x, xticklabels=['标准41点', '放大41点', '稀疏21点'], title=title)
        ax.legend(fontsize=8)
    axes[2].set(title='稀疏21点：实际末端跟踪误差', xlabel='时间（s）', ylabel='位置误差（μm）')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('控制器、时长、模型和步长不变；每组只更换参考轨迹')
    fig.tight_layout()
    fig.savefig(args.out/'segment_tracking_comparison.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
