# 实验10：轨迹平滑与跟踪对照
# 功能：固定控制器，比较线性、三次、五次和精化加五次轨迹。
# 输出：results/smooth_tracking，包含误差、离散jerk和步间力矩差。

"""同一时长和控制器下，对照线性插值、直接三次样条与约束五次样条。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib
import numpy as np
import scipy
import pinocchio as pin
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common.rizon4_analysis_common import Robot,NAMES
from experiments.trajectory_tracking.experiments_07_heart_trajectory import plan,save,reference_position
from experiments.trajectory_tracking.experiments_09_heart_tracking_comparison import check_dynamics
from experiments.trajectory_smoothing.smooth_trajectory import JointCurve,refine_nodes


def simulate(robot,T,curve,scale):
    model=mujoco.MjModel.from_xml_path(str(robot.path));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in NAMES]
    for aid,name in zip(ids,NAMES):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if aid<0 or model.actuator_trnid[aid,0]!=jid or not np.allclose(model.actuator_gear[aid],[1,0,0,0,0,0]):raise ValueError('Unsupported actuator')
    kp=model.actuator_gainprm[ids,0];kv=-model.actuator_biasprm[ids,2]
    if np.any(kp<=0) or not np.allclose(model.actuator_biasprm[ids,1],-kp):raise ValueError('Unsupported servo')
    data.qpos[robot.mq]=curve.angles[0];data.ctrl[ids]=curve.angles[0];mujoco.mj_forward(model,data)
    rows=[];dt=model.opt.timestep
    for k in range(int(np.ceil((curve.duration+1)/dt))):
        qref,vref,aref=curve.evaluate(data.time)
        q=robot.q(qref);v=np.zeros(7);a=np.zeros(7);v[robot.iv]=vref;a[robot.iv]=aref
        tau=pin.rnea(robot.model,robot.data,q,v,a).copy()-pin.computeGeneralizedGravity(robot.model,robot.data,q).copy()
        command=qref+(kv*vref+tau[robot.iv])/kp
        applied=command.copy()
        for j,aid in enumerate(ids):
            if model.actuator_ctrllimited[aid]:applied[j]=np.clip(command[j],*model.actuator_ctrlrange[aid])
        data.ctrl[ids]=applied;mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
        target=reference_position(data.time,T.translation,scale,curve.duration)
        actual=data.xpos[robot.body].copy();angles=data.qpos[robot.mq].copy()
        if not np.isfinite(angles).all() or abs(data.time-(k+1)*dt)>1e-7:raise RuntimeError('Simulation reset/diverged')
        rows.append(dict(time_s=float(data.time),phase='draw' if data.time<=curve.duration else 'hold',
            position_error_m=float(np.linalg.norm(actual-target)),rotation_error_rad=float(np.linalg.norm(pin.log3(data.xmat[robot.body].reshape(3,3).T@T.rotation))),
            contacts=int(data.ncon),clipped=bool(np.any(abs(applied-command)>1e-12)),
            margin_rad=float(np.min(np.minimum(angles-robot.lower,robot.upper-angles))),
            **{f'actual_{c}':float(actual[j]) for j,c in enumerate('xyz')},
            **{f'target_{c}':float(target[j]) for j,c in enumerate('xyz')},
            **{f'{field}_{n}':float(values[j]) for field,values in [('qref',qref),('vref',vref),('aref',aref),('q',angles),('v',data.qvel[robot.mv]),('a',data.qacc[robot.mv]),('force',data.qfrc_actuator[robot.mv])] for j,n in enumerate(NAMES)}))
    draw=[r for r in rows if r['phase']=='draw'];err=np.array([r['position_error_m'] for r in draw])
    acc=np.array([[r[f'a_{n}'] for n in NAMES] for r in rows]);force=np.array([[r[f'force_{n}'] for n in NAMES] for r in rows])
    # 实际加速度的离散差分用于同 dt 下比较，不称为连续时间 jerk 的精确值。
    jerk=np.diff(acc,axis=0)/dt
    metrics=dict(rmse_mm=float(np.sqrt(np.mean(err**2))*1000),max_error_mm=float(err.max()*1000),
        max_actual_acceleration_rad_s2=float(abs(acc).max()),discrete_jerk_rms_rad_s3=float(np.sqrt(np.mean(jerk**2))),
        discrete_jerk_peak_rad_s3=float(abs(jerk).max()),max_force_Nm=float(abs(force).max()),
        force_step_max_Nm=float(abs(np.diff(force,axis=0)).max()),max_contacts=max(r['contacts'] for r in rows),
        clipped_steps=sum(r['clipped'] for r in rows),min_margin_rad=min(r['margin_rad'] for r in rows))
    return rows,metrics


def validate_curve(robot,T,curve,scale):
    errors=[];rotation=[];margin=[]
    for t in np.linspace(0,curve.duration,4001):
        q,_,_=curve.evaluate(t);p=robot.position(q)
        errors.append(np.linalg.norm(p-reference_position(t,T.translation,scale,curve.duration)))
        rotation.append(np.linalg.norm(pin.log3(robot.data.oMf[robot.frame].rotation.T@T.rotation)))
        margin.append(np.min(np.minimum(q-robot.lower,robot.upper-q)))
    if max(errors)>1e-4 or max(rotation)>1e-3:raise RuntimeError('Spline changes Cartesian path too much')
    return dict(fk_path_max_error_mm=float(max(errors)*1000),fk_rotation_max_rad=float(max(rotation)),sampled_reference_margin_rad=float(min(margin)))


def main():
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=root/'results/smooth_tracking')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    robot=Robot(root/'flexiv_rizon4/flexiv_rizon4.xml');check_dynamics(robot);robot.validate(42)
    summary=[];primary={}
    for case,duration,scale in [('standard',10.,.004),('faster',6.,.004),('larger',10.,.005)]:
        cfg=SimpleNamespace(samples=401,duration=duration,scale=scale)
        T,times,old,nodes=plan(robot,cfg);refined=refine_nodes(robot,T,times,scale)
        np.savez(args.out/f'{case}_nodes.npz',times=times,original=old,refined=refined,rotation=T.rotation,translation=T.translation,scale=scale)
        # 增加五次但未精化节点的一组，区分边界约束与 IK 精度的贡献。
        methods=[('linear','linear',old),('cubic','cubic',old),('quintic','quintic',old),('refined_quintic','quintic',refined)]
        for name,kind,angles in methods:
            curve=JointCurve(times,angles,kind);checks=validate_curve(robot,T,curve,scale)
            rows,metrics=simulate(robot,T,curve,scale)
            save(args.out/f'{case}_{name}.csv',rows)
            result=dict(case=case,method=name,duration_s=duration,scale=scale,**metrics,**checks,**curve.continuity())
            summary.append(result)
            if case=='standard':primary[name]=(curve,rows)
            print(json.dumps(result),flush=True)
    # 不同方法的连续性字段不同，汇总 CSV 补齐列。
    keys=list(dict.fromkeys(k for r in summary for k in r));save(args.out/'summary.csv',[{k:r.get(k) for k in keys} for r in summary])
    meta=dict(summary=summary,scipy_version=scipy.__version__,pinocchio_version=pin.__version__,mujoco_version=mujoco.__version__,
        dt_s=robot.mj.opt.timestep,model_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
        changes='SciPy quintic B-spline, zero endpoint velocity/acceleration, IK tolerances 1e-10, analytical feedforward derivatives',
        unchanged='Same Cartesian temporal law, duration, gains, model, controller formula and simulation timestep',
        not_claimed='No TOPPRA call, no time-optimality, no jerk bound; no new spline algorithm; reference collision-free status not certified')
    (args.out/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False})
    labels={'linear':'上一版线性','cubic':'直接三次样条','quintic':'约束五次样条','refined_quintic':'精化 IK＋约束五次'}
    fig,axs=plt.subplots(1,3,figsize=(14,4.5));g=[r for r in summary if r['case']=='standard']
    for ax,key,title in zip(axs,['discrete_jerk_rms_rad_s3','force_step_max_Nm','rmse_mm'],['实际加速度变化率 RMS（rad/s³）','每仿真步最大力矩变化（Nm）','同步位置 RMSE（mm）']):
        bars=ax.bar(range(4),[r[key] for r in g],color=['#d98238','#809ab0','#6caaa0','#237d61']);ax.bar_label(bars,fmt='%.4f',fontsize=8)
        ax.set(xticks=range(4),xticklabels=['线性','三次','五次','精化＋五次'],title=title);ax.margins(y=.2)
    fig.suptitle('同一10秒心形：控制器不变，仅改轨迹构造');fig.tight_layout();fig.savefig(args.out/'smooth_metrics.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,9))
    # 展示 joint4 在固定 4.0～4.2 秒窗口的节点跳变，避免按结果挑最明显片段。
    for name in ['linear','cubic','refined_quintic']:
        curve,rows=primary[name];grid=np.linspace(4.,4.2,2001)
        axs[0].plot(grid,[curve.evaluate(t)[1][3] for t in grid],label=labels[name])
        axs[1].plot([r['time_s'] for r in rows],[r['a_joint4'] for r in rows],label=labels[name])
        axs[2].plot([r['time_s'] for r in rows],[r['position_error_m']*1000 for r in rows],label=labels[name])
    axs[0].set(ylabel='joint4 参考速度（rad/s）',title='局部放大：分段常速变为连续速度')
    axs[1].set(ylabel='joint4 实际加速度（rad/s²）',xlim=(4.,4.2))
    axs[2].set(ylabel='三维误差（mm）',xlabel='时间（s）')
    for ax in axs:ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/'smooth_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
