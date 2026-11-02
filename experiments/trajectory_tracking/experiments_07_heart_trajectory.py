# 实验07：心形规划与跟踪仿真
# 功能：逐点求解固定姿态心形目标并插值，可选动力学仿真。
# 输出：results/heart_trajectory；规划误差与跟踪误差分别统计。

"""Rizon4 心形轨迹：Pinocchio 规划，可选 MuJoCo 执行器跟踪。

本文件基于 link7 和现有 Pinocchio IK 求解器重新实现。
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from experiments.common.paths import source_file
from pathlib import Path
import argparse
import csv
import json
import hashlib
import time
from contextlib import nullcontext
import numpy as np
import pinocchio as pin
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common.rizon4_analysis_common import Robot, NAMES
from experiments.inverse_kinematics.rizon4_ik import frame_pose, solve_ik, IKOptions


def smooth(u):
    # 五次时间律使参数在起止时速度和加速度为零，但不是等弧长匀速。
    u=np.clip(u,0.,1.)
    return u**3*(10-15*u+6*u*u)


def heart(theta,anchor,scale):
    """在世界 YZ 平面画心形，顶部凹点与起始位置重合。"""
    p=np.array(anchor,copy=True)
    # X 保持不变，在世界 YZ 平面生成心形。
    p[1]+=scale*16*np.sin(theta)**3
    p[2]+=scale*(13*np.cos(theta)-5*np.cos(2*theta)-2*np.cos(3*theta)-np.cos(4*theta)-5)
    return p


def reference_position(t,anchor,scale,duration):
    return heart(2*np.pi*smooth(t/duration),anchor,scale)


def save(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plan(robot,args):
    home=np.array([0.,0.,0.,1.57,0.,0.,0.])
    T=frame_pose(robot,home)
    options=IKOptions(position_tolerance=1e-6,rotation_tolerance=1e-5,max_iterations=150)
    times=np.linspace(0,args.duration,args.samples)
    angles=[];rows=[];current=home.copy()
    for i,t in enumerate(times):
        # 沿心形移动位置，同时保持 Home 姿态。
        target=pin.SE3(T.rotation,reference_position(t,T.translation,args.scale,args.duration))
        # 以前一点的解作为初值，优先保持相邻解的连续性。
        result=solve_ik(robot,target,current,'pose',options)
        if not result['success']:
            raise RuntimeError(f'Heart point {i}/{args.samples-1}: {result["status"]}; '
                               'reduce --scale or increase --samples; no successful plan exported')
        jump=float(np.max(np.abs(result['angles']-current)))
        # 检查相邻节点的最大关节跳变，过大则停止导出。
        if jump>0.15:
            raise RuntimeError(f'Joint jump {jump:.3f} rad at sample {i}; increase --samples')
        current=result['angles'].copy()
        angles.append(current)
        actual=frame_pose(robot,current)
        rows.append(dict(time_s=float(t),sample=i,
                         **{f'target_{c}_m':float(target.translation[j]) for j,c in enumerate('xyz')},
                         **{f'pin_{c}_m':float(actual.translation[j]) for j,c in enumerate('xyz')},
                         **{f'{n}_rad':float(current[j]) for j,n in enumerate(NAMES)},
                         position_error_m=result['position_error_m'],rotation_error_rad=result['rotation_error_rad'],
                         ik_iterations=result['iterations'],joint_step_max_rad=jump))
    return T,times,np.array(angles),rows


def interpolate(t,times,angles):
    # 关节角采用分段线性插值，不产生角度过冲，但速度可能跳变。
    return np.array([np.interp(t,times,angles[:,i]) for i in range(7)])


def validate_dense(robot,args,T,times,angles):
    rows=[]
    # 对每段插值加密检查；节点达标不保证节点之间也达标。
    for t in np.linspace(0,args.duration,4*(args.samples-1)+1):
        a=interpolate(t,times,angles)
        actual=frame_pose(robot,a)
        target=reference_position(t,T.translation,args.scale,args.duration)
        rows.append(dict(time_s=float(t),position_error_m=float(np.linalg.norm(actual.translation-target)),
                         rotation_error_rad=float(np.linalg.norm(pin.log3(actual.rotation.T@T.rotation)))))
    if max(r['position_error_m'] for r in rows)>1e-4 or max(r['rotation_error_rad'] for r in rows)>1e-3:
        raise RuntimeError('Interpolated path accuracy failed; increase --samples')
    return rows


def simulate(robot,args,T,times,angles):
    model=mujoco.MjModel.from_xml_path(str(robot.path));data=mujoco.MjData(model)
    # 按名称查找并核对位置执行器，避免把关节索引当控制索引。
    actuator_ids=[]
    for name in NAMES:
        aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        if aid<0 or model.actuator_trnid[aid,0]!=jid:
            raise ValueError(f'Expected official position actuator named {name}')
        actuator_ids.append(aid)
    # 只在初始化时设置 qpos；跟踪阶段通过执行器驱动。
    data.qpos[robot.mq]=angles[0]
    data.ctrl[actuator_ids]=angles[0]
    mujoco.mj_forward(model,data)
    context=nullcontext(None)
    if args.view:
        from mujoco import viewer as mjviewer
        context=mjviewer.launch_passive(model,data)
    rows=[];trace=[]
    with context as viewer:
        wall=time.perf_counter()
        total=args.duration+args.hold
        steps=int(np.ceil(total/model.opt.timestep))
        stride=max(1,round(.03/model.opt.timestep))
        if viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=T.translation
                viewer.cam.distance=1.6;viewer.cam.azimuth=0;viewer.cam.elevation=-10
        for k in range(steps):
            if viewer and not viewer.is_running():
                break
            lock=viewer.lock() if viewer else nullcontext()
            with lock:
                # 按当前仿真时间计算本步的关节角指令。
                command=interpolate(data.time,times,angles)
                data.ctrl[actuator_ids]=command
                # 动力学由执行器和模型决定，实际轨迹可偏离 IK 参考。
                mujoco.mj_step(model,data)
                mujoco.mj_forward(model,data)  # 使读取的位姿与步进后的关节角、时间一致
                target=reference_position(data.time,T.translation,args.scale,args.duration)
                actual=data.xpos[robot.body].copy()
                rot=data.xmat[robot.body].reshape(3,3)
                rows.append(dict(time_s=float(data.time),phase='draw' if data.time<=args.duration else 'hold',
                                 **{f'command_{n}_rad':float(command[j]) for j,n in enumerate(NAMES)},
                                 **{f'actual_{n}_rad':float(data.qpos[robot.mq[j]]) for j,n in enumerate(NAMES)},
                                 **{f'target_{c}_m':float(target[j]) for j,c in enumerate('xyz')},
                                 **{f'actual_{c}_m':float(actual[j]) for j,c in enumerate('xyz')},
                                 position_error_m=float(np.linalg.norm(actual-target)),
                                 rotation_error_rad=float(np.linalg.norm(pin.log3(rot.T@T.rotation))),
                                 contact_count=int(data.ncon)))
                if k%stride==0 and data.time<=args.duration:
                    trace.append(actual)
                if viewer and k%stride==0:
                    scene=viewer.user_scn;scene.ngeom=0
                    for start,end in zip(trace[:-1],trace[1:]):
                        if scene.ngeom>=scene.maxgeom:break
                        if np.linalg.norm(end-start)<1e-9:continue
                        geom=scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),np.array([1.,.1,.2,1.]))
                        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.0015,start,end)
                        scene.ngeom+=1
            if viewer and k%stride==0:
                viewer.sync()
                wait=data.time-(time.perf_counter()-wall)
                if wait>0:time.sleep(min(wait,.05))
    if not rows:raise RuntimeError('No physics samples recorded')
    return rows


def plot_heart_plan(out, planned, actual, times, rows, dense):
    """分开显示节点误差与区间峰值，避免密集插值振荡遮挡节点曲线。"""
    fig = plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], hspace=.48, wspace=.3)
    path_ax = fig.add_subplot(grid[:, 0])
    node_ax = fig.add_subplot(grid[0, 1])
    interp_ax = fig.add_subplot(grid[1, 1], sharex=node_ax)
    path_ax.plot(planned[:, 1], planned[:, 2], color='#adb5bd', lw=3, label='目标心形')
    path_ax.plot(actual[:, 1], actual[:, 2], '--', color='#2563a6', lw=1.3, label='节点正运动学')
    path_ax.set(xlabel='世界 Y（m）', ylabel='世界 Z（m）', aspect='equal', title='目标与 IK 节点轨迹')
    path_ax.legend(loc='best')
    node = np.array([float(r['position_error_m']) for r in rows])*1e6
    dense_t = np.array([float(r['time_s']) for r in dense])
    dense_e = np.array([float(r['position_error_m']) for r in dense])*1e6
    # 每个相邻IK节点区间保留密集检查的最大值；不使用平滑或均值隐藏峰值。
    bins = np.clip(np.searchsorted(times, dense_t, side='right')-1, 0, len(times)-2)
    peaks = np.array([dense_e[bins == i].max() if np.any(bins == i) else np.nan for i in range(len(times)-1)])
    centers = (times[:-1]+times[1:])/2
    node_ax.plot(times, node, color='#2563a6', lw=1.1)
    interp_ax.plot(centers, peaks, color='#df7f28', lw=1.6)
    for ax, title, x, y in [(node_ax, '① IK 节点位置误差', times, node),
                             (interp_ax, '② 关节插值后：各节点区间的采样最大误差', dense_t, dense_e)]:
        i = int(np.argmax(y))
        ax.scatter([x[i]], [y[i]], color='#b32624', s=24, zorder=5)
        ax.annotate(f'最大 {y[i]:.3f} μm', (x[i], y[i]), xytext=(8, -20), textcoords='offset points', fontsize=10)
        ax.set(title=title, ylabel='位置误差（μm）', ylim=(0, max(float(y.max())*1.22, .01)))
        ax.grid(alpha=.22)
        ax.set_xlim(times[0], times[-1])
    interp_ax.set_xlabel('规划时间（s）')
    fig.suptitle('心形轨迹规划精度：节点求解与关节插值分别检查', fontsize=15)
    fig.text(.5, .025, '1 μm = 0.001 mm；右侧两图纵轴范围不同。下图为区间采样峰值，非连续时间误差上界；不含动力学跟踪。', ha='center', fontsize=10)
    fig.subplots_adjust(top=.88, bottom=.12)
    fig.savefig(out/'heart_plan.png', dpi=180)
    plt.close(fig)


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description='Rizon4 heart: Pinocchio planning, optional MuJoCo physics')
    p.add_argument('--model',type=Path,default=root/'flexiv_rizon4/flexiv_rizon4.xml')
    p.add_argument('--out',type=Path,default=root/'results/heart_trajectory')
    p.add_argument('--scale',type=float,default=.004)
    p.add_argument('--samples',type=int,default=401)
    p.add_argument('--duration',type=float,default=10.)
    p.add_argument('--hold',type=float,default=1.)
    p.add_argument('--simulate',action='store_true',help='Run actual mj_step actuator tracking')
    p.add_argument('--view',action='store_true',help='Show physics viewer (implies --simulate)')
    args=p.parse_args()
    if (args.samples<21 or not np.isfinite([args.scale,args.duration,args.hold]).all()
        or args.scale<=0 or args.duration<=0 or args.hold<0):p.error('Invalid sampling, scale or timing')
    robot=Robot(args.model);checks=robot.validate(42)
    T,times,angles,rows=plan(robot,args)
    dense=validate_dense(robot,args,T,times,angles)
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'heart_plan.csv',rows);save(args.out/'interpolation_check.csv',dense)
    planned=np.array([[r[f'target_{c}_m'] for c in 'xyz'] for r in rows])
    actual=np.array([[r[f'pin_{c}_m'] for c in 'xyz'] for r in rows])
    slopes=np.diff(angles,axis=0)/np.diff(times)[:,None]
    summary=dict(model=str(robot.path),model_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
                 pinocchio_version=pin.__version__,mujoco_version=mujoco.__version__,
                 samples=args.samples,duration_s=args.duration,scale=args.scale,
                 frame='link7 BODY; fixed home orientation; world YZ plane',
                 temporal_law='theta=2*pi*(10*u^3-15*u^4+6*u^5); not constant arc speed',
                 heart_width_m=float(np.ptp(planned[:,1])),heart_height_m=float(np.ptp(planned[:,2])),
                 knot_max_position_error_m=max(r['position_error_m'] for r in rows),
                 knot_max_rotation_error_rad=max(r['rotation_error_rad'] for r in rows),
                 interpolation_max_position_error_m=max(r['position_error_m'] for r in dense),
                 interpolation_max_rotation_error_rad=max(r['rotation_error_rad'] for r in dense),
                 max_adjacent_joint_step_rad=float(np.abs(np.diff(angles,axis=0)).max()),
                 max_joint_speed_estimate_rad_s=np.abs(slopes).max(axis=0).tolist(),
                 joint_closure_max_rad=float(np.max(np.abs(angles[-1]-angles[0]))),
                 minimum_limit_margin_rad=float(np.min(np.minimum(angles-robot.lower,robot.upper-angles))),
                 fk_checks=checks,collision_free_certified=False,
                 code_sha256={n:hashlib.sha256(source_file(n).read_bytes()).hexdigest()
                              for n in ['experiments_07_heart_trajectory.py','rizon4_ik.py','rizon4_analysis_common.py']})
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    plot_heart_plan(args.out, planned, actual, times, rows, dense)
    fig,ax=plt.subplots(figsize=(10,5))
    for i in range(7):ax.plot(times,angles[:,i],label=NAMES[i])
    ax.set(xlabel='规划时间（s）',ylabel='关节角（rad）',title='Pinocchio 连续初值 IK 关节参考')
    ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(args.out/'heart_joints.png',dpi=170);plt.close(fig)
    if args.simulate or args.view:
        sim=simulate(robot,args,T,times,angles);save(args.out/'heart_tracking.csv',sim)
        # 只对画线阶段统计同步跟踪误差，不混入末端保持阶段。
        draw=[r for r in sim if r['phase']=='draw']
        error=np.array([r['position_error_m'] for r in draw])
        summary['physics']=dict(draw_samples=len(draw),draw_rmse_m=float(np.sqrt(np.mean(error**2))),
                                draw_max_error_m=float(error.max()),max_contact_count=max(r['contact_count'] for r in sim),
                                end_time_s=sim[-1]['time_s'],completed=sim[-1]['time_s']>=args.duration+args.hold-1e-8,
                                actual_min_limit_margin_rad=float(min(min(float(r[f'actual_{n}_rad'])-robot.lower[i],robot.upper[i]-float(r[f'actual_{n}_rad'])) for r in sim for i,n in enumerate(NAMES))))
        fig,axes=plt.subplots(1,2,figsize=(11,5))
        axes[0].plot(planned[:,1],planned[:,2],color='gray',label='目标')
        axes[0].plot([r['actual_y_m'] for r in draw],[r['actual_z_m'] for r in draw],color='crimson',label='MuJoCo 实际')
        axes[0].set(xlabel='世界 Y（m）',ylabel='世界 Z（m）',aspect='equal',title='动力学画线轨迹')
        axes[1].plot([r['time_s'] for r in draw],error*1000)
        axes[1].set(xlabel='仿真时间（s）',ylabel='实际位置误差（mm）',title='位置执行器跟踪误差')
        axes[0].legend();fig.tight_layout();fig.savefig(args.out/'heart_tracking.png',dpi=170);plt.close(fig)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
