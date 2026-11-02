# 验证：实验06对照与消融结果
# 功能：复算保存解的误差及合法成功标签，检查采样顺序和代码哈希。
# 输出：实验06的verification.json；直接执行会写入验证记录。

from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from pathlib import Path
import csv,json,hashlib
import numpy as np
from experiments.inverse_kinematics.run_06_reference import ROOT,EvaluationRobot,violation
from experiments.inverse_kinematics.rizon4_ik import task

out=ROOT/'results/ik_comparison'
meta=json.loads((out/'metadata.json').read_text())
cases=json.loads((out/'cases.json').read_text())
rows=list(csv.DictReader((out/'solutions.csv').open(encoding='utf-8-sig')))
r=EvaluationRobot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
import pinocchio as pin
worst=0.
for row in rows:
    case=cases[int(row['target_id'])];T=np.array(case['target'])
    target=pin.SE3(T[:3,:3],T[:3,3])
    angles=np.array([float(row[f'final_joint{i}']) for i in range(1,8)])
    _,_,ep,er=task(r,angles,target,row['mode'],.2,False)
    worst=max(worst,abs(ep-float(row['position_error_m'])),abs(er-float(row['rotation_error_rad'])))
    expected=ep<=1e-4 and (row['mode']=='position' or er<=1e-3) and violation(r,angles)<=1e-10
    assert expected==(row['success']=='True')
assert worst<1e-12
# 独立重建原实验的目标和初值采样顺序。
rng=np.random.default_rng(meta['seed']);span=r.upper-r.lower
known=rng.uniform(r.lower+.1*span,r.upper-.1*span,(meta['samples'],7))
for truth,c in zip(known,cases):
    assert np.array_equal(truth,c['reference'])
    assert np.array_equal(np.clip(truth+rng.normal(0,.15,7),r.lower,r.upper),c['starts']['near'])
    assert np.array_equal(rng.uniform(r.lower,r.upper),c['starts']['random'])
# 校验随包清单，不依赖原电脑上的 F 盘目录。
manifest=json.loads((ROOT/'source_manifest.json').read_text(encoding='utf-8'))
unchanged={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in manifest['packaged_code_sha256'].items()}
assert all(unchanged.values())
assert len(rows)==meta['samples']*6*8
checks=dict(saved_result_count=len(rows),recomputed_error_max_difference=worst,
            original_sampling_exact=True,packaged_code_hashes_match=unchanged,
            run_checks=meta['checks'],final_source_sha256=hashlib.sha256((ROOT/'experiments/inverse_kinematics/run_06_reference.py').read_bytes()).hexdigest(),
            note='整理版仅调整入口和包内路径；历史实验元数据保持原样，最终代码哈希见source_manifest.json。')
(out/'verification.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(checks,indent=2,ensure_ascii=False))
