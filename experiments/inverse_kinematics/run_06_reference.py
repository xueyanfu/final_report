# 实验06：固定阻尼DLS对照与组件消融
# 功能：在统一目标、初值和预算下比较8种策略，记录误差、耗时及超限。
# 输出：results/ik_comparison中的CSV、元数据与图表。

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
#消融实验
from pathlib import Path
import argparse, csv, hashlib, json, platform, time
from dataclasses import asdict
import numpy as np
import pinocchio as pin
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.common.rizon4_analysis_common import Robot, NAMES
from experiments.inverse_kinematics.rizon4_ik import IKOptions, task, frame_pose, solve_ik, check_task_derivatives

ROOT=Path(__file__).resolve().parents[2]
# λ 平方加到 JJᵀ；所有组共享第一个阻尼、权重、更新比例和停止阈值。
VARIANTS={
 'A_fixed':(False,False,False,False),
 'B_cap':(False,True,False,False),
 'C_candidates':(True,True,False,False),
 'D_search':(True,True,True,False),
 'E_full':(True,True,True,True),
 'minus_candidates':(False,True,True,True),
 'minus_cap':(True,False,True,True),
 'minus_search':(True,True,False,True),
}
LABELS=['A 固定阻尼','B 加步长限制','C 加阻尼候选','D 加回溯','E 完整方法','去阻尼候选','去步长限制','去回溯']

class EvaluationRobot(Robot):
    """仅实验：允许评估超限状态，不改变原 Robot 的限位保护。"""
    def q(self, angles):
        a=np.asarray(angles)
        if a.shape!=(7,) or not np.all(np.isfinite(a)):
            raise ValueError('Expected seven finite angles')
        q=pin.neutral(self.model); q[self.iq]=a
        return q

def violation(robot,a):
    return float(max(0,np.max(robot.lower-a),np.max(a-robot.upper)))

def run(robot,target,a0,mode,opts,flags):
    multi,cap,search,bounds=flags
    a=a0.copy(); history=[]; status='max_iterations'
    last=(0.,0.,0.); evaluations=0
    for i in range(opts.max_iterations+1):
        e,J,ep,er=task(robot,a,target,mode,opts.rotation_weight)
        evaluations+=1; cost=float(e@e/2)
        history.append(dict(iteration=i,position_error_m=ep,rotation_error_rad=er,cost=cost,
            sigma_min=float(np.linalg.svd(J,compute_uv=False)[-1]),
            limit_violation_rad=violation(robot,a),damping=last[0],alpha=last[1],step_inf_rad=last[2],
            **{n:float(v) for n,v in zip(NAMES,a)}))
        if ep<=opts.position_tolerance and (mode=='position' or er<=opts.rotation_tolerance):
            status='success';break
        if i==opts.max_iterations: break
        accepted=False
        for damping in (opts.damping if multi else opts.damping[:1]):
            delta=-J.T@np.linalg.solve(J@J.T+damping**2*np.eye(len(e)),e)
            if cap: delta*=min(1.,opts.max_joint_step/max(np.max(np.abs(delta)),1e-30))
            velocity=np.zeros(robot.model.nv);velocity[robot.iv]=delta
            for k in range(opts.backtracking_steps if search else 1):
                alpha=.5**k
                candidate=pin.integrate(robot.model,robot.q(a),alpha*velocity)[robot.iq]
                if bounds: candidate=np.clip(candidate,robot.lower,robot.upper)
                step=float(np.max(np.abs(candidate-a)))
                # 多阻尼以目标下降作为候选筛选；无筛选时不能让后续候选生效。
                if multi or search:
                    if step<1e-12: continue
                    trial=task(robot,candidate,target,mode,opts.rotation_weight,False)[0]
                    evaluations+=1
                    if float(trial@trial/2)>=cost: continue
                a=candidate;last=(float(damping),alpha,step);accepted=True;break
            if accepted: break
        if not accepted: status='stalled';break
    return a,status,history,evaluations

def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=12)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--max-iterations',type=int,default=250)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--out',type=Path,default=ROOT/'results/ik_comparison')
    p.add_argument('--plots-only', '--report-only', dest='report_only', action='store_true', help='Regenerate plots from saved results')
    args=p.parse_args()
    if args.report_only:
        def read(name):
            data=list(csv.DictReader((args.out/name).open(encoding='utf-8-sig')))
            for row in data:
                for k,v in row.items():
                    if v in ['True','False']: row[k]=v=='True'
                    elif v=='': row[k]=None
                    else:
                        try: row[k]=float(v) if any(c in v for c in '.eE') else int(v)
                        except ValueError: pass
            return data
        plot_results(args.out,read('solutions.csv'),read('iterations.csv'),read('summary.csv'),json.loads((args.out/'metadata.json').read_text()))
        return
    if min(args.samples,args.repeats)<1 or args.seed<0: p.error('Invalid count/seed')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    model=ROOT/'flexiv_rizon4/flexiv_rizon4.xml'
    strict=Robot(model);robot=EvaluationRobot(model);opts=IKOptions(max_iterations=args.max_iterations);opts.validate()
    checks={'model':strict.validate(args.seed),'derivatives':check_task_derivatives(strict,args.seed)}
    rng=np.random.default_rng(args.seed);span=robot.upper-robot.lower
    known=rng.uniform(robot.lower+.1*span,robot.upper-.1*span,(args.samples,7))
    home=np.zeros(7);home[3]=1.57; strict.q(home)
    cases=[];targets=[]
    for number,truth in enumerate(known):
        target=frame_pose(robot,truth)
        starts={'near':np.clip(truth+rng.normal(0,.15,7),robot.lower,robot.upper),
                'home':home.copy(),'random':rng.uniform(robot.lower,robot.upper)}
        targets.append(dict(target_id=number,reference=truth.tolist(),target=target.homogeneous.tolist(),
                            starts={k:v.tolist() for k,v in starts.items()}))
        for mode in ['position','pose']:
            for start,a0 in starts.items(): cases.append((number,mode,start,target,a0))
    (out/'cases.json').write_text(json.dumps(targets,indent=2),encoding='utf-8')
    # 预热每种策略；随机化每轮算法顺序，不改变目标和初值。
    for flags in VARIANTS.values():run(robot,cases[0][3],cases[0][4],cases[0][1],opts,flags)
    order_rng=np.random.default_rng(args.seed+10000)
    rows=[];traces=[];timings=[];parity=0.;repeat_error=0.
    for number,mode,start,target,a0 in cases:
        results={};times={k:[] for k in VARIANTS}
        for repeat in range(args.repeats):
            for key in order_rng.permutation(list(VARIANTS)):
                begin=time.perf_counter();res=run(robot,target,a0,mode,opts,VARIANTS[key]);elapsed=time.perf_counter()-begin
                times[key].append(elapsed)
                timings.append(dict(target_id=number,mode=mode,start=start,algorithm=key,repeat=repeat,seconds=elapsed))
                if key in results:
                    repeat_error=max(repeat_error,float(np.max(np.abs(res[0]-results[key][0]))))
                    assert res[1]==results[key][1] and len(res[2])==len(results[key][2])
                results[key]=res
        original=solve_ik(strict,target,a0,mode,opts)
        full=results['E_full'];parity=max(parity,float(np.max(np.abs(original['angles']-full[0]))))
        assert original['status']==full[1] and original['iterations']==len(full[2])-1
        assert np.allclose(original['angles'],full[0],rtol=0,atol=1e-12)
        for key,(a,status,h,evals) in results.items():
            last=h[-1];physical=status=='success';legal=violation(robot,a)<=1e-10
            row=dict(target_id=number,mode=mode,start=start,algorithm=key,status=status,
                task_success=physical,success=physical and legal,iterations=len(h)-1,
                position_error_m=last['position_error_m'],rotation_error_rad=last['rotation_error_rad'],
                time_median_ms=1000*float(np.median(times[key])),evaluations=evals,
                final_limit_violation_rad=violation(robot,a),
                violating_states=sum(s['limit_violation_rad']>1e-10 for s in h),
                max_limit_violation_rad=max(s['limit_violation_rad'] for s in h),
                sigma_min_final=last['sigma_min'],sigma_min_path=min(s['sigma_min'] for s in h),
                **{f'final_{n}':float(v) for n,v in zip(NAMES,a)})
            rows.append(row)
            traces.extend(dict(target_id=number,mode=mode,start=start,algorithm=key,**s) for s in h)
        print(f'{number+1}/{args.samples} {mode} {start} complete',flush=True)
    csvwrite(out/'solutions.csv',rows);csvwrite(out/'iterations.csv',traces);csvwrite(out/'timings.csv',timings)
    summary=[]
    for key in VARIANTS:
        for mode,start in [('all','all')]+[(m,s) for m in ['position','pose'] for s in ['near','home','random']]:
            group=[r for r in rows if r['algorithm']==key and (mode=='all' or (r['mode']==mode and r['start']==start))]
            success=[r for r in group if r['success']]
            entry=dict(algorithm=key,mode=mode,start=start,trials=len(group),successes=len(success),
                       success_rate=len(success)/len(group),task_successes=sum(r['task_success'] for r in group),
                       path_violating_cases=sum(r['violating_states']>0 for r in group),
                       final_violating_cases=sum(r['final_limit_violation_rad']>1e-10 for r in group),
                       violating_states=sum(r['violating_states'] for r in group))
            for metric in ['iterations','position_error_m','rotation_error_rad','time_median_ms','sigma_min_path']:
                entry[metric+'_all_median']=float(np.median([r[metric] for r in group]))
                entry[metric+'_success_median']=float(np.median([r[metric] for r in success])) if success else None
            summary.append(entry)
    csvwrite(out/'summary.csv',summary)
    checks.update(full_solver_max_angle_difference=parity,repeat_max_angle_difference=repeat_error)
    assert repeat_error==0
    meta=dict(seed=args.seed,samples=args.samples,repeats=args.repeats,options=asdict(opts),variants=VARIANTS,
        versions=dict(python=platform.python_version(),pinocchio=pin.__version__,mujoco=mujoco.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__),
        checks=checks,frame='link7 BODY origin in MJCF world',collision_checked=False,
        hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['experiments','flexiv_rizon4'] for f in (ROOT/folder).rglob('*') if f.is_file() and '__pycache__' not in str(f)},
        timing='perf_counter, includes common trace/SVD instrumentation, excludes plotting and IO; median of repeats; randomized order',
        source='https://github.com/stack-of-tasks/pinocchio/blob/devel/examples/inverse-kinematics.py')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    plot_results(out,rows,traces,summary,meta)

def plot_results(out,rows,traces,summary,meta):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    overall=[r for r in summary if r['mode']=='all'];keys=list(VARIANTS)
    fig,axs=plt.subplots(2,2,figsize=(14,10))
    for ax,field,title in zip(axs.flat,['success_rate','iterations_success_median','time_median_ms_all_median','path_violating_cases'],['满足任务且最终限位合法的成功率','成功案例迭代次数中位数','全部案例耗时中位数（ms）','迭代过程超限的案例数']):
        vals=[r[field] for r in overall];ax.bar(range(len(keys)),vals)
        ax.set_xticks(range(len(keys)),LABELS,rotation=25,ha='right');ax.set_title(title)
        for i,v in enumerate(vals):
            if v is not None: ax.text(i,v,f'{v:.3g}',ha='center',va='bottom')
    fig.tight_layout();fig.savefig(out/'06_1_3_comparison.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,mode,title in zip(axs,['position','pose'],['仅位置','完整位姿']):
        g=[r for r in summary if r['algorithm']=='E_full' and r['mode']==mode]
        ax.bar(['邻近','Home','随机'],[r['success_rate'] for r in g]);ax.set(title=title,ylim=(0,1.15),ylabel='成功率')
        for i,r in enumerate(g):ax.text(i,r['success_rate'],f"{r['successes']}/{r['trials']}",ha='center',va='bottom')
    fig.tight_layout();fig.savefig(out/'06_2_initialization.png',dpi=160);plt.close(fig)
    # 预先固定目标0、pose/random，避免挑选有利于某算法的展示案例。
    fig,axs=plt.subplots(1,3,figsize=(16,4.5))
    for key in keys[:5]:
        h=[s for s in traces if s['target_id']==0 and s['mode']=='pose' and s['start']=='random' and s['algorithm']==key]
        for ax,metric,title in zip(axs,['position_error_m','rotation_error_rad','sigma_min'],['实际位置误差（m）','实际旋转误差（rad）','加权残差 Jacobian 最小奇异值']):
            ax.semilogy([s['iteration'] for s in h],[max(s[metric],1e-15) for s in h],label=LABELS[keys.index(key)])
            ax.set(xlabel='迭代次数',title=title);ax.grid(alpha=.2)
    axs[0].axhline(1e-4,ls='--',color='black');axs[1].axhline(1e-3,ls='--',color='black');axs[2].legend(fontsize=8)
    fig.suptitle('固定示例：目标 0，完整位姿，随机初值；全部轨迹见 iterations.csv')
    fig.tight_layout();fig.savefig(out/'06_4_curves.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
