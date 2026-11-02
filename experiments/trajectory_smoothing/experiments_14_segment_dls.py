# 实验14：固定节点的区段残差DLS与精化五次样条对照。
# 对照相同节点、精度、时间律及姿态目标，仅改变区间修正过程。
# 输出CSV、可重放系数、运行元数据和图表，不生成Markdown。
"""复用实验07心形、实验10节点精化和样条；新增区段DLS对照。

来源和课程创新边界见同目录segment_dls.py文件头。
基础样条API：https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
本实验不复现TORM或Hauser算法，不宣称其性能排名。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, replace
import numpy as np
import scipy
import pinocchio as pin
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common.rizon4_analysis_common import Robot
from experiments.inverse_kinematics.rizon4_ik import frame_pose
from experiments.trajectory_tracking.experiments_07_heart_trajectory import reference_position, save
from experiments.trajectory_smoothing.smooth_trajectory import JointCurve, refine_nodes
from experiments.trajectory_smoothing.segment_dls import SegmentCurve, SegmentOptions, correct_segments, base_margin

ROOT = Path(__file__).resolve().parents[2]


def measure(robot, T, curve, scale, grid):
    qs = curve.values(grid)
    ep, er = [], []
    for t, q in zip(grid, qs):
        actual = frame_pose(robot, q)
        goal = reference_position(t, T.translation, scale, curve.duration)
        ep.append(float(np.linalg.norm(actual.translation-goal)))
        er.append(float(np.linalg.norm(pin.log3(actual.rotation.T@T.rotation))))
    ep, er = np.array(ep), np.array(er)
    jerk = curve.values(grid, 3)
    return dict(position_max_um=float(ep.max()*1e6),
                position_rmse_um=float(np.sqrt(np.mean(ep**2))*1e6),
                rotation_max_rad=float(er.max()),
                jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))),
                jerk_peak_rad_s3=float(np.abs(jerk).max())), ep, er


def plot(out, summary, raw):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    colors = ['#9b9b9b', '#3981ac', '#207f5a']
    labels = ['精化五次基线', '区段DLS：2个基函数', '区段DLS：4个基函数']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    rows = [r for r in summary if r['case'] == 'standard']
    for ax, field, title in zip(axes[0], ['position_max_um', 'position_rmse_um'], ['最大位置误差（μm）', '位置误差RMSE（μm）']):
        bars = ax.bar(range(3), [r[field] for r in rows], color=colors)
        ax.bar_label(bars, fmt='%.4f')
        ax.set(xticks=range(3), xticklabels=labels, title=title)
        ax.margins(y=.2)
    ax = axes[1, 0]
    for key, label, color in zip(['base', 'two_profiles', 'segment_dls'], labels, colors):
        ts, ep = raw[key]
        windows = np.array_split(np.arange(len(ts)), 100)
        ax.plot([ts[w].mean() for w in windows], [ep[w].max()*1e6 for w in windows], color=color, label=label)
    ax.set(xlabel='时间（s）', ylabel='窗口内采样最大误差（μm）', title='相同41节点；修正节点之间的误差')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    values = [r['jerk_rms_rad_s3'] for r in rows]
    bars = ax.bar(range(3), values, color=colors)
    ax.bar_label(bars, fmt='%.4f')
    ax.set(xticks=range(3), xticklabels=labels, title='参考轨迹jerk RMS（rad/s³）')
    ax.margins(y=.2)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('区段残差DLS：固定节点与边界导数；额外优化区间系数', fontsize=15)
    fig.tight_layout()
    fig.savefig(out/'segment_dls_comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/segment_dls')
    parser.add_argument('--eval-samples', type=int, default=16001)
    args = parser.parse_args()
    if args.eval_samples < 4001:
        parser.error('Use at least 4001 evaluation samples')
    args.out.mkdir(parents=True, exist_ok=True)
    robot = Robot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
    home = np.array([0., 0, 0, 1.57, 0, 0, 0])
    T = frame_pose(robot, home)
    options = SegmentOptions()
    summary, history, raw = [], [], {}
    for case, nodes, duration, scale in [('standard', 41, 10., .004), ('finer', 81, 10., .004), ('larger', 41, 10., .005)]:
        ts = np.linspace(0, duration, nodes)
        start = time.perf_counter()
        angles = refine_nodes(robot, T, ts, scale)
        base = JointCurve(ts, angles, 'quintic')
        base_time = time.perf_counter()-start
        grid = np.linspace(0, duration, args.eval_samples)
        for name, k in [('base', 4), ('two_profiles', 2), ('segment_dls', 4)]:
            start = time.perf_counter()
            if name == 'base':
                curve = SegmentCurve(base, np.zeros((nodes-1, k, 7)))
                details = dict(global_scale=0., modified_segments=0, pose_evaluations=0, jacobian_evaluations=0,
                               conservative_joint_margin_rad=float(base_margin(robot, base)))
            else:
                curve, steps, details = correct_segments(robot, T, base, scale, replace(options, profiles=k))
                history += [dict(case=case, method=name, **s) for s in steps]
            seconds = base_time+time.perf_counter()-start
            metrics, ep, er = measure(robot, T, curve, scale, grid)
            result = dict(case=case, method=name, nodes=nodes, **metrics, planning_s=seconds,
                          extra_parameters=0 if name == 'base' else (nodes-1)*k*7, **details)
            summary.append(result)
            np.savez_compressed(args.out/f'{case}_{name}.npz', times=ts, angles=angles,
                                coefficients=curve.coefficients, rotation=T.rotation, translation=T.translation, scale=scale)
            save(args.out/f'{case}_{name}_errors.csv', [dict(time_s=float(t), position_error_m=float(p), rotation_error_rad=float(r))
                                                       for t, p, r in zip(grid, ep, er)])
            if case == 'standard':
                raw[name] = grid, ep
            print(json.dumps(result), flush=True)
    save(args.out/'summary.csv', summary)
    save(args.out/'iterations.csv', history)
    sources = ['experiments/trajectory_smoothing/'+p for p in ['segment_dls.py', 'experiments_14_segment_dls.py', 'smooth_trajectory.py']]
    sources += ['experiments/inverse_kinematics/rizon4_ik.py', 'experiments/trajectory_tracking/experiments_07_heart_trajectory.py',
                'experiments/common/rizon4_analysis_common.py']
    meta = dict(options=asdict(options), evaluation_samples=args.eval_samples, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, pinocchio=pin.__version__, mujoco=mujoco.__version__,
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                model_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
                scope='Offline kinematic path error; no controller or real robot evaluation',
                timing='Single-run diagnostic, includes shared IK and correction but excludes final evaluation and plots; no repeated timing claim',
                attribution='Existing DLS and spline methods; new project-specific segment residual and endpoint-locked correction; not a literature-first claim',
                experiment_design='profiles/damping selected in preliminary standard/finer experiments; larger scale evaluated after fixing defaults; neither is a broad independent benchmark')
    (args.out/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    plot(args.out, summary, raw)


if __name__ == '__main__':
    main()
