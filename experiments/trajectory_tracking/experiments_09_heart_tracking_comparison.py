# 实验09：心形跟踪控制输入对照
# 功能：比较位置指令、速度前馈、速度加动力学前馈及不同工况。
# 输出：results/heart_comparison；比较同一仿真时刻的末端误差。

"""固定模型和增益，比较原位置指令与参考速度前馈；运动学继续使用 Pinocchio。

官方位置执行器：tau = kp*(ctrl-q) - kv*qdot。
扩展：速度前馈；进一步用 Pinocchio RNEA 补偿惯性及科氏项。
不改 XML、不直接施加外力，不在运行过程中覆盖 qpos。
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import hashlib
import numpy as np
import pinocchio as pin
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common.rizon4_analysis_common import Robot,NAMES
from experiments.trajectory_tracking.experiments_07_heart_trajectory import plan,interpolate,reference_position,validate_dense,save


def check_dynamics(robot):
    """核对惯量和逆动力学，尤其防止重复计入 armature 或重力补偿。"""
    rng=np.random.default_rng(71);mass_error=torque_error=0.
    for _ in range(12):
        a=rng.uniform(robot.lower*.7,robot.upper*.7);q=robot.q(a)
        v=rng.normal(0,.2,7);acc=rng.normal(0,.4,7)
        robot.md.qpos[robot.mq]=a;robot.md.qvel[robot.mv]=v[robot.iv]
        mujoco.mj_forward(robot.mj,robot.md)
        M=np.zeros((7,7));mujoco.mj_fullM(robot.mj,robot.md,M)
        P=pin.crba(robot.model,robot.data,q).copy()
        mass_error=max(mass_error,float(np.max(np.abs(M[np.ix_(robot.mv,robot.mv)]-P[np.ix_(robot.iv,robot.iv)]))))
        tau=pin.rnea(robot.model,robot.data,q,v,acc).copy()
        gravity=pin.computeGeneralizedGravity(robot.model,robot.data,q).copy()
        robot.md.qacc[robot.mv]=acc[robot.iv];mujoco.mj_inverse(robot.mj,robot.md)
        torque_error=max(torque_error,float(np.max(np.abs(robot.md.qfrc_inverse[robot.mv]-(tau-gravity)[robot.iv]))))
    if mass_error>1e-8 or torque_error>1e-8:raise RuntimeError(f'Dynamics mismatch {mass_error}, {torque_error}')
    return dict(mass_matrix_max_abs=mass_error,inverse_dynamics_without_gravity_max_abs_Nm=torque_error)


def run(robot,T,times,angles,args,method):
    model=mujoco.MjModel.from_xml_path(str(robot.path));data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in NAMES]
    if min(ids)<0:raise ValueError('Missing named actuators')
    for aid,name in zip(ids,NAMES):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if model.actuator_trnid[aid,0]!=jid or not np.allclose(model.actuator_gear[aid],[1,0,0,0,0,0]):
            raise ValueError('Requires named direct unit-gear joint actuators')
    kp=model.actuator_gainprm[ids,0].copy();kv=-model.actuator_biasprm[ids,2].copy()
    if np.any(kp<=0) or not np.allclose(model.actuator_biasprm[ids,1],-kp):
        raise ValueError('Requires affine position servos')
    slopes=np.diff(angles,axis=0)/np.diff(times)[:,None]
    # 分段线性路径无经典连续加速度；用节点中心差分构造平滑近似前馈。
    accelerations=np.gradient(np.gradient(angles,times,axis=0),times,axis=0)
    data.qpos[robot.mq]=angles[0];data.ctrl[ids]=angles[0];mujoco.mj_forward(model,data)
    rows=[]
    for k in range(int(np.ceil((args.duration+1)/model.opt.timestep))):
        t=float(data.time);ref=interpolate(t,times,angles)
        # 使用本段线性关节路径的精确斜率；终点保持阶段速度为零。
        i=min(np.searchsorted(times,t,side='right')-1,len(slopes)-1)
        velocity=slopes[max(0,i)] if t<args.duration else np.zeros(7)
        command=ref+(kv/kp*velocity if method!='baseline' else 0)
        if method=='dynamics_ff':
            acc=interpolate(t,times,accelerations) if t<args.duration else np.zeros(7)
            q=robot.q(ref);v=np.zeros(7);a=np.zeros(7);v[robot.iv]=velocity;a[robot.iv]=acc
            torque=pin.rnea(robot.model,robot.data,q,v,a).copy()
            gravity=pin.computeGeneralizedGravity(robot.model,robot.data,q).copy()
            # XML 已有 gravcomp=1，因此减去重力；RNEA 已含 armature，不再重复加。
            command+=(torque-gravity)[robot.iv]/kp
        clipped=command.copy()
        for j,aid in enumerate(ids):
            if model.actuator_ctrllimited[aid]:clipped[j]=np.clip(command[j],*model.actuator_ctrlrange[aid])
        data.ctrl[ids]=clipped
        mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
        expected=reference_position(data.time,T.translation,args.scale,args.duration)
        actual=data.xpos[robot.body].copy();a=data.qpos[robot.mq].copy()
        # 同一物理时间比较三维误差；不移动时间轴来消除滞后。
        rows.append(dict(time_s=float(data.time),phase='draw' if data.time<=args.duration else 'hold',
            position_error_m=float(np.linalg.norm(actual-expected)),
            rotation_error_rad=float(np.linalg.norm(pin.log3(data.xmat[robot.body].reshape(3,3).T@T.rotation))),
            contact_count=int(data.ncon),command_clipped=bool(np.any(np.abs(command-clipped)>1e-12)),
            limit_margin_rad=float(np.min(np.minimum(a-robot.lower,robot.upper-a))),
            **{f'target_{c}':float(expected[j]) for j,c in enumerate('xyz')},
            **{f'actual_{c}':float(actual[j]) for j,c in enumerate('xyz')},
            **{f'command_{n}':float(clipped[j]) for j,n in enumerate(NAMES)},
            **{f'actual_{n}':float(a[j]) for j,n in enumerate(NAMES)},
            **{f'velocity_{n}':float(data.qvel[robot.mv[j]]) for j,n in enumerate(NAMES)},
            **{f'force_{n}':float(data.qfrc_actuator[robot.mv[j]]) for j,n in enumerate(NAMES)}))
        if not np.isfinite(a).all() or abs(data.time-(k+1)*model.opt.timestep)>1e-7:
            raise RuntimeError('Simulation diverged or reset')
    draw=[r for r in rows if r['phase']=='draw'];errors=np.array([r['position_error_m'] for r in draw])
    metrics=dict(rmse_mm=float(np.sqrt(np.mean(errors**2))*1000),max_error_mm=float(errors.max()*1000),
        p95_error_mm=float(np.quantile(errors,.95)*1000),max_rotation_deg=float(np.rad2deg(max(r['rotation_error_rad'] for r in draw))),
        min_limit_margin_rad=min(r['limit_margin_rad'] for r in rows),max_contacts=max(r['contact_count'] for r in rows),
        command_clipped_steps=sum(r['command_clipped'] for r in rows),
        max_actual_speed_rad_s=max(abs(r[f'velocity_{n}']) for r in rows for n in NAMES),
        max_actuator_force_Nm=max(abs(r[f'force_{n}']) for r in rows for n in NAMES))
    return rows,metrics,dict(kp=kp.tolist(),kv=kv.tolist(),timestep=model.opt.timestep)


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=root/'results/heart_comparison')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    robot=Robot(root/'flexiv_rizon4/flexiv_rizon4.xml');checks=robot.validate(42)
    # 预先固定工况，全部公开，不根据结果挑选最好的一组。
    dynamics_checks=check_dynamics(robot)
    cases=[('standard',10.,.004),('faster',6.,.004),('slower',14.,.004),('larger',10.,.005)]
    summary=[]; primary={};plan_checks=[]
    for name,duration,scale in cases:
        cfg=SimpleNamespace(samples=401,duration=duration,scale=scale)
        T,times,angles,nodes=plan(robot,cfg);dense=validate_dense(robot,cfg,T,times,angles)
        save(args.out/f'{name}_plan.csv',nodes)
        plan_checks.append(dict(case=name,max_interpolation_error_m=max(r['position_error_m'] for r in dense)))
        for method in ['baseline','velocity_ff','dynamics_ff']:
            rows,metrics,control=run(robot,T,times,angles,cfg,method)
            save(args.out/f'{name}_{method}.csv',rows)
            summary.append(dict(case=name,method=method,duration_s=duration,scale=scale,**metrics))
            if name=='standard':primary[method]=rows
            print(name,method,metrics,flush=True)
    save(args.out/'summary.csv',summary)
    record=dict(summary=summary,plan_checks=plan_checks,fk_checks=checks,dynamics_checks=dynamics_checks,control=control,
        pinocchio_version=pin.__version__,mujoco_version=mujoco.__version__,
        model_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
        formula='ctrl=q_ref+(kv/kp)*dq_ref+(RNEA(q_ref,dq_ref,ddq_ref)-g(q_ref))/kp; latter term only for dynamics_ff; existing control clipping retained',
        unchanged='Model, gains, gravcomp, actuator force bounds, duration, path, initialization and sample clock shared within each pair.',
        limitations='Piecewise reference velocity can jump; no global collision or hardware safety proof; baseline is previous project tracking, not an official heart demo.')
    (args.out/'summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False})
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    r=[r for r in primary['baseline'] if r['phase']=='draw']
    axs[0].plot([x['target_y'] for x in r],[x['target_z'] for x in r],'k--',label='同一目标')
    for method,label,color in [('baseline','原位置指令','#d98238'),('velocity_ff','速度前馈','#247f9e'),('dynamics_ff','速度＋Pinocchio动力学前馈','#258c65')]:
        r=[r for r in primary[method] if r['phase']=='draw']
        axs[0].plot([x['actual_y'] for x in r],[x['actual_z'] for x in r],color=color,label=label)
        axs[1].plot([x['time_s'] for x in r],[x['position_error_m']*1000 for x in r],color=color,label=label)
    axs[0].set(xlabel='世界 Y（m）',ylabel='世界 Z（m）',aspect='equal',title='10秒心形：实际轨迹')
    axs[1].set(xlabel='同一仿真时间（s）',ylabel='三维位置误差（mm）',title='不作时间平移的跟踪误差')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/'heart_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(cases))
    for method,offset,label,color in [('baseline',-.25,'原位置指令','#d98238'),('velocity_ff',0,'速度前馈','#247f9e'),('dynamics_ff',.25,'速度＋动力学前馈','#258c65')]:
        values=[r['rmse_mm'] for r in summary if r['method']==method]
        bars=ax.bar(x+offset,values,.25,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set(xticks=x,xticklabels=['标准 10s','加快 6s','减慢 14s','放大心形 10s'],ylabel='三维位置 RMSE（mm）',title='四组预设工况：同模型、同增益、同路径对照')
    ax.legend();ax.set_ylim(0,max(r['rmse_mm'] for r in summary)*1.22)
    fig.tight_layout();fig.savefig(args.out/'heart_cases.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
