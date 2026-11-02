# 实验15验证：从原始动力学记录重算指标，用独立MDH重放实际关节构型。
# 来源：项目rizon4_mdh、实验10控制记录；本文件新增实验15统计与同步检查。
# 输出JSON，不生成文档；这是保存结果复核，不是再次动力学运行。
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import hashlib
import json
import numpy as np
from experiments.kinematics.rizon4_mdh import fk
from experiments.trajectory_smoothing.smooth_trajectory import JointCurve
from experiments.trajectory_smoothing.segment_dls import SegmentCurve
from experiments.trajectory_tracking.experiments_07_heart_trajectory import reference_position

ROOT = Path(__file__).resolve().parents[2]


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'results/segment_tracking')
    args = parser.parse_args()
    folder = args.results
    meta = json.loads((folder/'metadata.json').read_text())
    for p, digest in {**meta['source_sha256'], **meta['source_inputs_sha256']}.items():
        require(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, f'Hash mismatch: {p}')
    require(hashlib.sha256((ROOT/'flexiv_rizon4/flexiv_rizon4.xml').read_bytes()).hexdigest() == meta['model_sha256'], 'Model mismatch')
    with (folder/'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        summaries = list(csv.DictReader(f))
    require(len(summaries) == 6, 'Expected six complete runs')
    checks = []
    for summary in summaries:
        name = f'{summary["case"]}_{summary["method"]}'
        d = np.genfromtxt(folder/f'{name}.csv', delimiter=',', names=True, dtype=None, encoding='utf-8-sig')
        time = d['time_s']
        require(len(d) == 5500 and abs(time[-1]-11) < 1e-8, 'Incomplete simulation')
        require(np.max(np.abs(np.diff(time)-meta['dt_s'])) < 1e-9, 'Time reset or missing samples')
        draw = d[d['phase'] == 'draw']
        xyz = np.column_stack([draw[f'actual_{c}']-draw[f'target_{c}'] for c in 'xyz'])
        errors = np.linalg.norm(xyz, axis=1)
        require(np.max(np.abs(errors-draw['position_error_m'])) < 1e-12, 'Stored error mismatch')
        rms = float(np.sqrt(np.mean(errors**2))*1e6)
        peak = float(errors.max()*1e6)
        require(abs(rms-float(summary['tracking_rmse_um'])) < 1e-8, 'RMSE mismatch')
        require(abs(peak-float(summary['tracking_max_um'])) < 1e-8, 'Peak mismatch')
        acc = np.column_stack([d[f'a_joint{j}'] for j in range(1, 8)])
        forces = np.column_stack([d[f'force_joint{j}'] for j in range(1, 8)])
        jerk = np.diff(acc, axis=0)/meta['dt_s']
        require(abs(np.sqrt(np.mean(jerk**2))-float(summary['discrete_jerk_rms_rad_s3'])) < 1e-9, 'Jerk mismatch')
        require(abs(np.abs(np.diff(forces, axis=0)).max()-float(summary['force_step_max_Nm'])) < 1e-9, 'Force-step mismatch')
        require(d['contacts'].max() == int(summary['max_contacts']), 'Contacts mismatch')
        require(np.count_nonzero(d['clipped']) == int(summary['clipped_steps']), 'Clipping mismatch')
        require(d['margin_rad'].min() > 0, 'Actual joint limit violated')
        with np.load(folder/f'{name}.npz', allow_pickle=False) as z:
            base = JointCurve(z['times'], z['angles'], 'quintic')
            curve = SegmentCurve(base, z['coefficients'])
            anchor, scale = z['translation'], float(z['scale'])
        mdh_error = reference_error = 0.
        for i in range(0, len(d), 10):
            row = d[i]
            q = np.array([row[f'q_joint{j}'] for j in range(1, 8)])
            actual = np.array([row[f'actual_{c}'] for c in 'xyz'])
            mdh_error = max(mdh_error, float(np.abs(fk(q)[:3, 3]-actual).max()))
            ref = fk(curve.evaluate(row['time_s'])[0])[:3, 3]
            target = reference_position(row['time_s'], anchor, scale, 10.)
            reference_error = max(reference_error, abs(np.linalg.norm(actual-ref)-row['actual_to_reference_m']),
                                  abs(np.linalg.norm(ref-target)-row['reference_to_target_m']))
        require(mdh_error < 1e-12 and reference_error < 1e-12, 'MDH/synchronization mismatch')
        checks.append(dict(run=name, rows=len(d), draw_rows=len(draw), tracking_rmse_um=rms,
                           mdh_actual_position_max_m=mdh_error, reference_decomposition_max_difference_m=float(reference_error)))
    result = dict(passed=True, runs=checks, replay_stride=10, source_hashes_checked=True,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  mdh_sha256=hashlib.sha256((ROOT/'experiments/kinematics/rizon4_mdh.py').read_bytes()).hexdigest(),
                  scope='Saved-data recomputation and independent kinematic replay; not a second dynamics run')
    (folder/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
