# 验证：MDH对实验06逆解的独立复核
# 功能：将已有最终关节角代入MDH链，重新判断误差与关节合法性。
# 输出：results/mdh_analysis中的复核CSV和JSON；不重新求解IK。

"""将现有IK最终角度代回独立MDH链，复核目标、实际误差和合法成功标签。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from experiments.kinematics.rizon4_mdh import fk
from experiments.common.rizon4_analysis_common import Robot

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/ik_comparison')
    p.add_argument('--out',type=Path,default=ROOT/'results/mdh_analysis')
    args=p.parse_args()
    meta=json.loads((args.results/'metadata.json').read_text(encoding='utf-8'))
    cases=json.loads((args.results/'cases.json').read_text(encoding='utf-8'))
    cases={c['target_id']:c for c in cases}
    with (args.results/'solutions.csv').open(encoding='utf-8-sig',newline='') as f:
        saved=list(csv.DictReader(f))
    robot=Robot(ROOT/'flexiv_rizon4/flexiv_rizon4.xml')
    records=[]
    target_error=max(float(np.max(np.abs(fk(c['reference'])-np.asarray(c['target'])))) for c in cases.values())
    for row in saved:
        angles=np.array([float(row[f'final_joint{i}']) for i in range(1,8)])
        actual=fk(angles)
        target=np.asarray(cases[int(row['target_id'])]['target'])
        ep=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
        er=float(np.linalg.norm(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).as_rotvec()))
        violation=float(max(0,np.max(robot.lower-angles),np.max(angles-robot.upper)))
        success=ep<=meta['options']['position_tolerance'] and (row['mode']=='position' or er<=meta['options']['rotation_tolerance']) and violation<=1e-10
        records.append(dict(target_id=row['target_id'],mode=row['mode'],start=row['start'],algorithm=row['algorithm'],
                            mdh_position_error_m=ep,mdh_rotation_error_rad=er,mdh_success=success,
                            label_matches=bool(success==(row['success']=='True')),
                            position_difference_m=abs(ep-float(row['position_error_m'])),
                            rotation_difference_rad=abs(er-float(row['rotation_error_rad']))))
    algorithms=sorted({r['algorithm'] for r in records})
    counts={a:sum(r['mdh_success'] for r in records if r['algorithm']==a) for a in algorithms}
    result=dict(saved_solution_count=len(saved),target_count=len(cases),
                target_matrix_max_difference=target_error,
                max_position_error_difference_m=max(r['position_difference_m'] for r in records),
                max_rotation_error_difference_rad=max(r['rotation_difference_rad'] for r in records),
                all_success_labels_match=all(r['label_matches'] for r in records),legal_success_counts=counts,
                note='Existing IK outputs independently checked with user MDH FK; solver was NOT replaced or rerun.',
                source_hashes={n:hashlib.sha256((args.results/n).read_bytes()).hexdigest() for n in ['metadata.json','cases.json','solutions.csv']})
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'ik_mdh_recheck.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (args.out/'ik_mdh_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(saved)==meta['samples']*6*8
    assert target_error<1e-10 and result['all_success_labels_match']
    assert max(result['max_position_error_difference_m'],result['max_rotation_error_difference_rad'])<1e-9
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
