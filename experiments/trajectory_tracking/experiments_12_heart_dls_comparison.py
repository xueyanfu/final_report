# 实验12：同一心形轨迹的基础DLS与扩展DLS对照。
# paired使用相同参考初值隔离求解器差异；sequential分别热启动评价实际规划。
# 输出CSV、参数记录与对照图；不包含动力学控制和样条改进。
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,csv,json,time,hashlib
from dataclasses import asdict
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.inverse_kinematics.run_06_reference import EvaluationRobot,run,VARIANTS,violation
from experiments.inverse_kinematics.rizon4_ik import IKOptions,frame_pose,task,solve_ik
from experiments.trajectory_tracking.experiments_07_heart_trajectory import reference_position
ROOT=Path(__file__).resolve().parents[2]

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=401)
    p.add_argument('--scale',type=float,default=.004)
    p.add_argument('--duration',type=float,default=10.)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,default=ROOT/'results/heart_dls_comparison')
    args=p.parse_args()
    if args.samples<2 or args.repeats<1 or not np.isfinite([args.scale,args.duration]).all() or min(args.scale,args.duration)<=0:p.error('Invalid arguments')
    args.out.mkdir(parents=True,exist_ok=True)
    robot=EvaluationRobot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
    opts=IKOptions(position_tolerance=1e-6,rotation_tolerance=1e-5,max_iterations=150)
    home=np.array([0.,0,0,1.57,0,0,0]);T=frame_pose(robot,home)
    times=np.linspace(0,args.duration,args.samples)
    targets=[pin.SE3(T.rotation,reference_position(t,T.translation,args.scale,args.duration)) for t in times]
    rng=np.random.default_rng(42);rows=[];summaries=[];reference=home.copy();reference_starts=[];identity_difference=0.
    # 公共初值来自完整方法的参考路径；因此paired是条件性诊断，不是独立规划成功率。
    for target in targets:
        reference_starts.append(reference.copy())
        a,status,h,_=run(robot,target,reference,'pose',opts,VARIANTS['E_full'])
        if status!='success' or violation(robot,a)>1e-10:raise RuntimeError('Shared reference path failed; change task parameters')
        original=solve_ik(robot,target,reference,'pose',opts)
        identity_difference=max(identity_difference,float(np.max(np.abs(a-original['angles']))))
        reference=a
    assert identity_difference<1e-10
    # 两种实现都先预热；计时均包含相同历史记录和SVD成本。
    for key in ['A_fixed','E_full']:run(robot,targets[1],home,'pose',opts,VARIANTS[key])
    for protocol in ['paired','sequential']:
        current={k:home.copy() for k in ['A_fixed','E_full']};active={k:True for k in current};paths={k:[] for k in current}
        for i,target in enumerate(targets):
            keys=list(current);rng.shuffle(keys)
            for key in keys:
                if not active[key]:continue
                initial=reference_starts[i] if protocol=='paired' else current[key]
                durations=[]
                for repeat in range(args.repeats):
                    t0=time.perf_counter();a,status,h,evaluations=run(robot,target,initial,'pose',opts,VARIANTS[key]);durations.append((time.perf_counter()-t0)*1000)
                _,_,ep,er=task(robot,a,target,'pose',opts.rotation_weight,False)
                legal=violation(robot,a)<=1e-10;success=ep<=opts.position_tolerance and er<=opts.rotation_tolerance and legal
                jump=float(np.max(np.abs(a-current[key]))) if paths[key] else float(np.max(np.abs(a-home)))
                rows.append(dict(protocol=protocol,algorithm=key,node=i,time_s=float(times[i]),success=success,status=status,iterations=h[-1]['iteration'],position_error_m=ep,rotation_error_rad=er,final_violation_rad=violation(robot,a),path_violating_states=sum(x['limit_violation_rad']>1e-10 for x in h),adjacent_step_rad=jump if protocol=='sequential' else '',elapsed_median_ms=float(np.median(durations)),**{f'initial_q{j+1}':float(v) for j,v in enumerate(initial)},**{f'final_q{j+1}':float(v) for j,v in enumerate(a)}))
                paths[key].append(a.copy());current[key]=a
                # 独立规划首次失败后停止，不拼接或用另一方法的解继续。
                if protocol=='sequential' and not success:active[key]=False
        for key in current:
            g=[r for r in rows if r['protocol']==protocol and r['algorithm']==key]
            complete=len(g)==args.samples and all(r['success'] for r in g)
            dense_error=None
            if protocol=='sequential' and complete:
                angles=np.array(paths[key]);dense_error=0.
                for t in np.linspace(0,args.duration,4*(args.samples-1)+1):
                    a=np.array([np.interp(t,times,angles[:,j]) for j in range(7)])
                    dense_error=max(dense_error,float(np.linalg.norm(frame_pose(robot,a).translation-reference_position(t,T.translation,args.scale,args.duration))))
            summaries.append(dict(protocol=protocol,algorithm=key,attempted=len(g),successful=sum(r['success'] for r in g),total_nodes=args.samples,all_nodes_solved=complete,position_max_m=max(r['position_error_m'] for r in g),rotation_max_rad=max(r['rotation_error_rad'] for r in g),time_median_ms=float(np.median([r['elapsed_median_ms'] for r in g])),iterations_median=float(np.median([r['iterations'] for r in g])),max_adjacent_step_rad=max(r['adjacent_step_rad'] for r in g) if protocol=='sequential' else None,dense_position_max_m=dense_error))
    write(args.out/'nodes.csv',rows);write(args.out/'summary.csv',summaries)
    meta=dict(options=asdict(opts),samples=args.samples,scale=args.scale,duration=args.duration,repeats=args.repeats,seed=42,full_solver_original_max_difference=identity_difference,protocols={'paired':'same prior full-solver reference initial guess at each node; conditional comparison','sequential':'each solver uses own prior solution; stops at first invalid endpoint'},timing='median repeated solve time including common trace/SVD; not real-time control guarantee',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest())
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for key,label in [('A_fixed','基础固定阻尼'),('E_full','扩展DLS')]:
        g=[r for r in rows if r['protocol']=='sequential' and r['algorithm']==key]
        axs[0].plot([r['time_s'] for r in g],[r['position_error_m']*1e6 for r in g],label=label)
        axs[1].plot([r['time_s'] for r in g],[r['adjacent_step_rad'] for r in g],label=label)
        g=[r for r in rows if r['protocol']=='paired' and r['algorithm']==key]
        axs[2].plot([r['time_s'] for r in g],[r['iterations'] for r in g],label=label)
    for ax,title,ylabel in zip(axs,['独立热启动：节点误差','独立热启动：相邻节点变化','共同初值：迭代次数'],['位置误差（μm）','最大单关节变化（rad）','迭代次数']):ax.set(title=title,xlabel='规划时间（s）',ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/'heart_dls_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
