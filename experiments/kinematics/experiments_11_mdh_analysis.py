# 实验11：MDH模型独立验证
# 功能：用MDH表完成FK、约定错误对照、Jacobian及单参数扰动分析。
# 输出：results/mdh_analysis；与Pinocchio、MuJoCo、Pybotics比较。

"""MDH四部分实验：FK、错误约定、Jacobian、单参数扰动。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import mujoco
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from experiments.common.rizon4_analysis_common import Robot
from experiments.kinematics.rizon4_mdh import PARAMETERS, WORLD_T_0, fk, jacobian, pybotics_model

ROOT = Path(__file__).resolve().parents[2]


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def errors(a, b):
    return (float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
            float(np.linalg.norm(Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).as_rotvec())))


def make_cases(robot, samples, seed):
    home = np.array([0., 0, 0, 1.57, 0, 0, 0])
    cases = [('zero', np.zeros(7)), ('home', home)]
    span = robot.upper-robot.lower
    # 单关节正负扰动，以及逐关节靠近上下限；其他关节保持Home。
    for j in range(7):
        for sign in [-1, 1]:
            q = home.copy()
            q[j] += sign*.1*span[j]
            cases.append((f'joint{j+1}_{sign:+d}', q))
        for name, value in [('lower', robot.lower[j]+1e-4), ('upper', robot.upper[j]-1e-4)]:
            q = home.copy()
            q[j] = value
            cases.append((f'joint{j+1}_{name}', q))
    rng = np.random.default_rng(seed)
    for i, q in enumerate(rng.uniform(robot.lower+.01*span, robot.upper-.01*span, (samples, 7))):
        cases.append((f'random_{i}', q))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=ROOT/'results/mdh_analysis')
    args = parser.parse_args()
    if args.samples < 1 or args.seed < 0:
        parser.error('samples must be positive and seed nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    robot = Robot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
    external = pybotics_model()
    cases = make_cases(robot, args.samples, args.seed)
    save_csv(args.out/'configurations.csv', [dict(case=n, **{f'q{j+1}_rad':v for j,v in enumerate(q)}) for n,q in cases])
    save_csv(args.out/'mdh_parameters.csv', [dict(joint=i+1, a_m=r[0], alpha_rad=r[1], d_m=r[2], offset_rad=r[3]) for i,r in enumerate(PARAMETERS)])
    fk_rows, wrong_rows, jac_rows, perturb_rows = [], [], [], []
    for index, (name, q) in enumerate(cases):
        own = fk(q)
        pin.framesForwardKinematics(robot.model, robot.data, robot.q(q))
        ppose = robot.data.oMf[robot.frame].homogeneous.copy()
        robot.md.qpos[robot.mq] = q
        mujoco.mj_forward(robot.mj, robot.md)
        mpose = np.eye(4)
        mpose[:3, :3] = robot.md.xmat[robot.body].reshape(3,3)
        mpose[:3, 3] = robot.md.xpos[robot.body]
        for method, t in [('Pinocchio', ppose), ('MuJoCo', mpose), ('Pybotics', external.fk(q))]:
            ep, er = errors(own, t)
            fk_rows.append(dict(case=name, reference=method, position_error_m=ep, rotation_error_rad=er,
                                matrix_max_abs=float(np.max(np.abs(own-t)))))
        no_offset = PARAMETERS.copy()
        no_offset[:,3] = 0
        for variant, t in [('correct', own), ('missing_base', fk(q, base=np.eye(4))),
                           ('missing_offsets', fk(q, parameters=no_offset)), ('standard_DH_order', fk(q, standard=True))]:
            ep, er = errors(t, ppose)
            wrong_rows.append(dict(case=name, variant=variant, position_error_m=ep, rotation_error_rad=er))
        # Jacobian全量对照；中心差分仅依赖独立MDH函数。
        j = jacobian(q)
        jp = np.asarray(pin.computeFrameJacobian(robot.model, robot.data, robot.q(q), robot.frame,
                                                pin.LOCAL_WORLD_ALIGNED))[:, robot.iv]
        mjp, mjr = np.zeros((3,robot.mj.nv)), np.zeros((3,robot.mj.nv))
        mujoco.mj_jacBody(robot.mj, robot.md, mjp, mjr, robot.body)
        jm = np.vstack((mjp[:,robot.mv], mjr[:,robot.mv]))
        jf = np.empty((6,7))
        h = 1e-6
        for col in range(7):
            dq = np.zeros(7); dq[col] = h
            plus, minus = fk(q+dq), fk(q-dq)
            jf[:3,col] = (plus[:3,3]-minus[:3,3])/(2*h)
            jf[3:,col] = Rotation.from_matrix(plus[:3,:3]@minus[:3,:3].T).as_rotvec()/(2*h)
        for method, other in [('Pinocchio',jp),('MuJoCo',jm),('Pybotics',external.jacobian_world(q)),('central_difference',jf)]:
            jac_rows.append(dict(case=name, reference=method,
                                linear_max_abs=float(np.max(np.abs(j[:3]-other[:3]))),
                                angular_max_abs=float(np.max(np.abs(j[3:]-other[3:])))))
        # 固定其他MDH参数，仅改变一个参数。
        for col, kind, magnitude in [(0,'a',.001),(1,'alpha',np.deg2rad(.1)),(2,'d',.001),(3,'offset',np.deg2rad(.1))]:
            for link in range(7):
                changed = PARAMETERS.copy(); changed[link,col] += magnitude
                ep, er = errors(fk(q, parameters=changed), own)
                perturb_rows.append(dict(case=name, parameter=f'{kind}{link+1}', kind=kind,
                                         delta=magnitude, position_error_mm=ep*1000, rotation_error_deg=np.rad2deg(er)))
        if index % 200 == 0:
            print(f'MDH cases: {index+1}/{len(cases)}', flush=True)
    for filename, rows in [('fk_checks.csv',fk_rows),('convention_checks.csv',wrong_rows),
                           ('jacobian_checks.csv',jac_rows),('parameter_perturbations.csv',perturb_rows)]:
        save_csv(args.out/filename, rows)
    maxima = {}
    for method in ['Pinocchio','MuJoCo','Pybotics']:
        group = [r for r in fk_rows if r['reference']==method]
        maxima[method] = {k:max(r[k] for r in group) for k in ['position_error_m','rotation_error_rad','matrix_max_abs']}
    jac_max = {method:{k:max(r[k] for r in jac_rows if r['reference']==method)
                      for k in ['linear_max_abs','angular_max_abs']}
               for method in ['Pinocchio','MuJoCo','Pybotics','central_difference']}
    convention_summary = [{ 'variant':v, **{k:float(np.median([r[k] for r in wrong_rows if r['variant']==v]))
                            for k in ['position_error_m','rotation_error_rad']}}
                          for v in ['correct','missing_base','missing_offsets','standard_DH_order']]
    perturb_summary = []
    for kind in ['a','alpha','d','offset']:
        for link in range(7):
            group=[r for r in perturb_rows if r['parameter']==f'{kind}{link+1}']
            perturb_summary.append(dict(parameter=f'{kind}{link+1}',kind=kind,
                position_median_mm=float(np.median([r['position_error_mm'] for r in group])),
                position_max_mm=max(r['position_error_mm'] for r in group),
                rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in group]))))
    save_csv(args.out/'convention_summary.csv',convention_summary)
    save_csv(args.out/'parameter_summary.csv',perturb_summary)
    checks = {
        'fk_all_references_below_1e-10':all(v['matrix_max_abs']<1e-10 for v in maxima.values()),
        'jacobian_all_below_1e-7':all(max(v.values())<1e-7 for v in jac_max.values()),
        'missing_base_always_365mm':all(abs(r['position_error_m']-.365)<1e-10 for r in wrong_rows if r['variant']=='missing_base'),
        'rotation_matrices_valid':all(np.allclose(fk(q)[:3,:3].T@fk(q)[:3,:3],np.eye(3),atol=1e-12,rtol=0)
                                      and abs(np.linalg.det(fk(q)[:3,:3])-1)<1e-12 for _,q in cases),
    }
    meta = dict(seed=args.seed, random_count=args.samples, total_configurations=len(cases),
        convention='Craig: Rx(alpha) Tx(a) Rz(q+offset) Tz(d); world_T_0=Trans(0,0,0.365)',
        frame='MDH7 == link7 BODY origin; fixed base; identity tool',
        parameters=PARAMETERS.tolist(), world_T_0=WORLD_T_0.tolist(),
        fk_maxima=maxima,jacobian_maxima=jac_max,checks=checks,
        collision_checked=False,finite_difference_step_rad=1e-6,
        perturbation='one positive parameter at a time: a,d +1mm; alpha,offset +0.1deg; all other parameters fixed',
        versions=dict(python=platform.python_version(),numpy=np.__version__,pinocchio=pin.__version__,mujoco=mujoco.__version__,
                      pybotics=importlib.metadata.version('pybotics'),attrs=importlib.metadata.version('attrs')),
        hashes={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [Path(__file__),ROOT/'experiments/kinematics/rizon4_mdh.py',ROOT/'flexiv_rizon4/flexiv_rizon4.xml']})
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    if not all(checks.values()):
        raise RuntimeError(f'MDH validation failed: {checks}')
    plot(args.out, fk_rows, convention_summary, jac_rows, perturb_summary)
    print(json.dumps(dict(fk=maxima,jacobian=jac_max,checks=checks),indent=2))


def plot(out, fk_rows, conventions, jac_rows, perturb):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for method in ['Pinocchio','MuJoCo','Pybotics']:
        group=[r for r in fk_rows if r['reference']==method]
        for ax,key in zip(axes,['position_error_m','rotation_error_rad']):
            ax.semilogy(np.maximum([r[key] for r in group],1e-18),label=method,alpha=.7)
    for ax,label in zip(axes,['位置差（m）','姿态角差（rad）']):
        ax.set(xlabel='构型序号（非时间）',ylabel=label);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('用户MDH模型：与三个独立库对照（显示下限1e-18）')
    fig.tight_layout();fig.savefig(out/'11_1_fk_validation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=['完整MDH','漏基座偏移','漏角度偏置','误用标准DH顺序']
    for ax,key,factor,label in zip(axes,['position_error_m','rotation_error_rad'],[1000,180/np.pi],['位置差中位数（mm）','姿态差中位数（°）']):
        values=[r[key]*factor for r in conventions]
        bars=ax.bar(names,values,color=['#328777','#df9a41','#cc6767','#3677a8'])
        ax.bar_label(bars,labels=[f'{x:.3g}' for x in values],padding=3)
        ax.set_ylabel(label);ax.tick_params(axis='x',rotation=12);ax.margins(y=.2)
    fig.suptitle('同一参数表：错误使用坐标约定会发生什么？（与Pinocchio对照）')
    fig.tight_layout();fig.savefig(out/'11_2_conventions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,label in zip(axes,['linear_max_abs','angular_max_abs'],['线速度Jacobian最大元素差','角速度Jacobian最大元素差']):
        methods=['Pinocchio','MuJoCo','Pybotics','central_difference']
        vals=[max(r[key] for r in jac_rows if r['reference']==m) for m in methods]
        ax.bar(methods,np.maximum(vals,1e-18));ax.set_yscale('log');ax.set_ylabel(label);ax.tick_params(axis='x',rotation=12)
    fig.suptitle('MDH解析Jacobian：跨库与中心差分验证')
    fig.tight_layout();fig.savefig(out/'11_3_jacobian.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    groups=[['a','d'],['alpha','offset']]
    for ax,kinds,title in zip(axes,groups,['单个长度参数 +1 mm','单个角度参数 +0.1°']):
        for k,kind in enumerate(kinds):
            rows=[r for r in perturb if r['kind']==kind]
            ax.bar(np.arange(1,8)+(k-.5)*.34,[r['position_median_mm'] for r in rows],width=.34,label=kind)
        ax.set(title=title,xlabel='MDH参数行号',ylabel='末端位移中位数（mm）',xticks=range(1,8));ax.legend()
    fig.suptitle('单参数扰动：其余参数和当前关节角固定；非真机公差或标定结论')
    fig.tight_layout();fig.savefig(out/'11_4_parameter_sensitivity.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
