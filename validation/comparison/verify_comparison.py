# 验证：实验09的跟踪对照
# 功能：检查保存的跟踪数据、标准工况复跑及动力学一致性。
# 输出：综合验证记录；包含部分运行检查。

"""独立复算对照实验结果，并复跑标准心形；SciPy 用于密集轮廓近邻检查。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree
from experiments.common.rizon4_analysis_common import Robot,NAMES
from experiments.trajectory_tracking.experiments_09_heart_tracking_comparison import run,check_dynamics
from experiments.trajectory_tracking.experiments_07_heart_trajectory import heart


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    root=Path(__file__).resolve().parents[2];out=root/'results'
    robot=Robot(root/'flexiv_rizon4/flexiv_rizon4.xml')
    summaries=read(out/'heart_comparison/summary.csv');contours=[];replays=[]
    for case in ['standard','faster','slower','larger']:
        nodes=read(out/f'heart_comparison/{case}_plan.csv')
        times=np.array([float(r['time_s']) for r in nodes]);angles=np.array([[float(r[f'{n}_rad']) for n in NAMES] for r in nodes])
        pin.framesForwardKinematics(robot.model,robot.data,robot.q(angles[0]));T=robot.data.oMf[robot.frame].copy()
        cfg=SimpleNamespace(duration=float(times[-1]),scale=.005 if case=='larger' else .004)
        dense=np.array([heart(t,T.translation,cfg.scale) for t in np.linspace(0,2*np.pi,40001)])
        tree=cKDTree(dense);max_gap=float(np.linalg.norm(np.diff(dense,axis=0),axis=1).max())
        for method in ['baseline','velocity_ff','dynamics_ff']:
            records=read(out/f'heart_comparison/{case}_{method}.csv');draw=[r for r in records if r['phase']=='draw']
            positions=np.array([[float(r[f'actual_{c}']) for c in 'xyz'] for r in draw])
            targets_xyz=np.array([[float(r[f'target_{c}']) for c in 'xyz'] for r in draw])
            errors=np.linalg.norm(positions-targets_xyz,axis=1)
            summary=next(r for r in summaries if r['case']==case and r['method']==method)
            assert abs(float(summary['rmse_mm'])-np.sqrt(np.mean(errors**2))*1000)<1e-10
            assert abs(float(summary['max_error_mm'])-errors.max()*1000)<1e-10
            # 轮廓指标不考虑时间对应，仅作辅助；有限密集点近似会略高估曲线距离。
            distances=tree.query(positions)[0]
            contours.append(dict(case=case,method=method,contour_rmse_mm=float(np.sqrt(np.mean(distances**2))*1000),
                contour_max_mm=float(distances.max()*1000),dense_curve_max_gap_mm=max_gap*1000))
            for r in records[::97]:
                a=np.array([float(r[f'actual_{n}']) for n in NAMES]);p=robot.position(a)
                assert np.max(np.abs(p-np.array([float(r[f'actual_{c}']) for c in 'xyz'])))<1e-10
            if case=='standard':
                replay,metrics,_=run(robot,T,times,angles,cfg,method)
                for fresh,saved in zip(replay,records):
                    assert fresh['position_error_m']==float(saved['position_error_m'])
                assert len(replay)==len(records)
                replays.append(method)
    original=json.loads((out/'heart_trajectory/summary.json').read_text())['physics']['draw_rmse_m']
    current=float(next(r for r in summaries if r['case']=='standard' and r['method']=='baseline')['rmse_mm'])/1000
    assert abs(original-current)<1e-12
    report=dict(
        standard_heart_replayed=replays,prior_project_baseline_reproduced=True,
        dynamics=check_dynamics(robot),contour_metrics=contours)
    (out/'comparison_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
