# 验证：实验10平滑轨迹
# 功能：检查样条导数、限位、连续性与统计，并复跑部分标准工况。
# 输出：results/smooth_tracking/verification.json。

"""复算统计、核对解析导数和整段限位，并复跑标准工况。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import csv,json
import numpy as np
import pinocchio as pin
from scipy.interpolate import PPoly,BSpline
from experiments.common.rizon4_analysis_common import Robot,NAMES
from experiments.trajectory_smoothing.smooth_trajectory import JointCurve
from experiments.trajectory_smoothing.experiments_10_smooth_tracking import simulate


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def spline_extrema(curve,robot):
    """逐段求导数多项式的根，检查整个区间，而非仅检查采样点。"""
    minimum=[];maximum=[]
    for j in range(7):
        spl=curve.spline
        poly=PPoly.from_spline(BSpline(spl.t,spl.c[:,j],spl.k)) if hasattr(spl,'t') else PPoly(spl.c[:,:,j],spl.x)
        extrema=poly.derivative().roots(extrapolate=False)
        points=np.r_[0,curve.duration,extrema[np.isfinite(extrema)&(extrema>=0)&(extrema<=curve.duration)]]
        values=poly(points);minimum.append(values.min());maximum.append(values.max())
    margin=float(min(np.min(np.array(minimum)-robot.lower),np.min(robot.upper-np.array(maximum))))
    assert margin>0
    return margin


def main():
    root=Path(__file__).resolve().parents[2];out=root/'results/smooth_tracking'
    meta=json.loads((out/'summary.json').read_text());robot=Robot(root/'flexiv_rizon4/flexiv_rizon4.xml')
    checks=[];dt=meta['dt_s']
    for summary in meta['summary']:
        case=summary['case'];method=summary['method'];nodes=np.load(out/f'{case}_nodes.npz')
        angles=nodes['refined'] if method=='refined_quintic' else nodes['original']
        curve=JointCurve(nodes['times'],angles,'quintic' if method=='refined_quintic' else method)
        rows=read(out/f'{case}_{method}.csv');draw=[r for r in rows if r['phase']=='draw']
        xyz=np.array([[float(r[f'actual_{c}'])-float(r[f'target_{c}']) for c in 'xyz'] for r in draw])
        rmse=np.sqrt(np.mean(np.sum(xyz**2,axis=1)))*1000
        acc=np.array([[float(r[f'a_{n}']) for n in NAMES] for r in rows]);jerk=np.diff(acc,axis=0)/dt
        assert abs(rmse-summary['rmse_mm'])<1e-10
        assert abs(float(np.sqrt(np.mean(jerk**2)))-summary['discrete_jerk_rms_rad_s3'])<1e-10
        margin=spline_extrema(curve,robot) if curve.spline is not None else summary['sampled_reference_margin_rad']
        derivative_error=0.
        if curve.spline is not None:
            # 三次样条的三阶导数在节点可跳变；差分检查选在每段内部。
            test_times=(curve.times[:-1]+curve.times[1:])/2
            for t in test_times[::13]:
                eps=1e-6;q,v,a=curve.evaluate(t)
                derivative_error=max(derivative_error,float(abs((curve.evaluate(t+eps)[0]-curve.evaluate(t-eps)[0])/(2*eps)-v).max()),
                    float(abs((curve.evaluate(t+eps)[1]-curve.evaluate(t-eps)[1])/(2*eps)-a).max()))
            assert derivative_error<1e-6,(case,method,derivative_error)
        if method=='refined_quintic':
            for t in [0,curve.duration]:
                assert np.abs(curve.spline(t,1)).max()<1e-8
                assert np.abs(curve.spline(t,2)).max()<1e-8
        for r in rows[::89]:
            a=np.array([float(r[f'q_{n}']) for n in NAMES]);pos=robot.position(a)
            assert np.max(np.abs(pos-np.array([float(r[f'actual_{c}']) for c in 'xyz'])))<1e-10
        replay=False
        if case=='standard' and method in ['linear','refined_quintic']:
            T=pin.SE3(nodes['rotation'],nodes['translation'])
            new,metrics=simulate(robot,T,curve,float(nodes['scale']))
            assert len(new)==len(rows)
            assert max(abs(r['position_error_m']-float(old['position_error_m'])) for r,old in zip(new,rows))<1e-12
            replay=True
        if method=='linear':
            prior=json.loads((root/'results/heart_comparison/summary.json').read_text())['summary']
            prior=next(r for r in prior if r['case']==case and r['method']=='dynamics_ff')
            assert abs(prior['rmse_mm']-rmse)<1e-10
        checks.append(dict(case=case,method=method,continuous_interval_margin_rad=margin,
            derivative_max_abs_error=derivative_error,statistics_recomputed=True,standard_replayed=replay))
    result=dict(checks=checks,reference_interval_limit_check='Polynomial derivative roots and endpoints; floating point arithmetic',
        prior_baseline_reproduced=True,dt_s=dt)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
