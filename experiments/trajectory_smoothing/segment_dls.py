# 实验14共享算法：端点二阶锁定的区段残差DLS。
# 固定已有IK节点，以区段多点位姿误差优化修正系数，保持节点q/dq/ddq。
# 提供解析导数、回溯、独立采样验收、关节限位及jerk预算。
"""相对逐点DLS的课程算法扩展，不宣称新的DLS数学算法或文献首创。

代码复用：本项目 rizon4_ik.task 的SE(3)残差及Jlog6 Jacobian，
其上游是Pinocchio官方示例（许可证：根目录PINOCCHIO_LICENSE.txt）：
https://github.com/stack-of-tasks/pinocchio/blob/f3b3953babbc65f9c91f294f956836ee5a6c9d0e/examples/inverse-kinematics.py
基准曲线复用本项目smooth_trajectory.JointCurve，调用SciPy而未修改库：
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.from_spline.html
SciPy许可证：references/SCIPY_LICENSE.txt。
相关研究（未复制其代码）：TORM https://sglab.kaist.ac.kr/TORM-auro/；
Hauser 2013 https://motion.cs.illinois.edu/papers/RSS2013_InterpolationOptimizationManifolds.pdf
本文件新增：锁定端点二阶导数的修正基、区段残差/系数Jacobian和验收流程。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PPoly
import pinocchio as pin
from experiments.inverse_kinematics.rizon4_ik import task
from experiments.trajectory_tracking.experiments_07_heart_trajectory import reference_position


@dataclass(frozen=True)
class SegmentOptions:
    profiles: int = 4
    probes: int = 9
    iterations: int = 2
    damping: float = .001
    rotation_weight: float = .2
    coefficient_step: float = .001
    max_jerk_rms_ratio: float = 1.05
    max_jerk_peak_ratio: float = 1.2

    def validate(self):
        if self.profiles not in (1, 2, 4) or self.probes < 5 or self.iterations < 1:
            raise ValueError('Invalid profiles/probes/iterations')
        values = [self.damping, self.rotation_weight, self.coefficient_step,
                  self.max_jerk_rms_ratio, self.max_jerk_peak_ratio]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Parameters must be finite and positive')
        if min(self.max_jerk_rms_ratio, self.max_jerk_peak_ratio) < 1:
            raise ValueError('Jerk budgets must permit the unmodified baseline')


def profiles(count):
    s = Polynomial([0, 1])
    bubble = 64*s**3*(1-s)**3
    return [bubble*(2*s-1)**k for k in range(count)]


class SegmentCurve:
    def __init__(self, base, coefficients):
        if base.kind != 'quintic':
            raise ValueError('Requires the experiment10 quintic base')
        self.base = base
        self.times, self.angles, self.duration = base.times, base.angles, base.duration
        self.coefficients = np.asarray(coefficients, dtype=float)
        if (self.coefficients.ndim != 3 or self.coefficients.shape[0] != len(self.times)-1
                or self.coefficients.shape[2] != 7 or not np.isfinite(self.coefficients).all()):
            raise ValueError('Expected finite coefficients: (intervals, profiles, 7)')
        self.basis = profiles(self.coefficients.shape[1])

    def values(self, times, derivative=0):
        ts = np.asarray(times, dtype=float)
        if np.any(ts < 0) or np.any(ts > self.duration):
            raise ValueError('Evaluation outside planning interval')
        i = np.clip(np.searchsorted(self.times, ts, side='right')-1, 0, len(self.times)-2)
        h = np.diff(self.times)[i]
        s = (ts-self.times[i])/h
        B = np.stack([p.deriv(derivative)(s) for p in self.basis], axis=-1)
        correction = np.einsum('...k,...kj->...j', B, self.coefficients[i])
        return self.base.spline(ts, derivative) + correction / np.asarray(h)[..., None]**derivative

    def evaluate(self, t):
        if t < 0:
            return self.angles[0].copy(), np.zeros(7), np.zeros(7)
        if t > self.duration:
            return self.angles[-1].copy(), np.zeros(7), np.zeros(7)
        return tuple(self.values(t, k) for k in range(3))


def base_margin(robot, base):
    """用分段多项式极值检查基准曲线，随后用系数上界保护修正曲线。"""
    margin = np.inf
    for j in range(7):
        spline = base.spline
        poly = PPoly.from_spline((spline.t, spline.c[:, j], spline.k))
        roots = poly.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= base.duration)]
        q = poly(np.r_[base.times, roots])
        margin = min(margin, float(np.min(q-robot.lower[j])), float(np.min(robot.upper[j]-q)))
    return margin


def segment_system(robot, T, base, index, C, fractions, scale, weight, jacobian=True):
    """链式Jacobian A=[b0*J, ..., bk*J]；向量按profile、joint顺序展开。"""
    bs = profiles(len(C))
    rows, matrices, ep, er = [], [], [], []
    h = base.times[index+1]-base.times[index]
    for s in fractions:
        t = base.times[index]+s*h
        B = np.array([b(s) for b in bs])
        q = base.spline(t)+B@C
        target = pin.SE3(T.rotation, reference_position(t, T.translation, scale, base.duration))
        residual, J, p, r = task(robot, q, target, 'pose', weight, jacobian)
        rows.append(residual)
        ep.append(p)
        er.append(r)
        if jacobian:
            matrices.append(np.hstack([b*J for b in B]))
    return (np.concatenate(rows), np.vstack(matrices) if jacobian else None,
            np.asarray(ep), np.asarray(er))


def correct_segments(robot, T, base, scale, options=None):
    opts = options or SegmentOptions()
    opts.validate()
    margin = base_margin(robot, base)
    if margin <= 0:
        raise ValueError('Baseline violates joint limits')
    C = np.zeros((len(base.times)-1, opts.profiles, 7))
    train = np.arange(1, opts.probes+1)/(opts.probes+1)
    # 检查点与训练点错开；最终评估和MDH复核另外使用更密网格。
    guard = (np.arange(32)+.5)/32
    history = []
    evaluations = 0
    for i in range(len(C)):
        for iteration in range(opts.iterations):
            r, A, _, _ = segment_system(robot, T, base, i, C[i], train, scale, opts.rotation_weight)
            old, _, old_ep, old_er = segment_system(robot, T, base, i, C[i], guard, scale, opts.rotation_weight, False)
            evaluations += len(train)+len(guard)
            # DLS求解变量是整段的系数，而非单个关节构型。
            delta = -np.linalg.solve(A.T@A+opts.damping**2*np.eye(C[i].size), A.T@r)
            delta *= min(1., opts.coefficient_step/max(float(np.abs(delta).max()), 1e-30))
            accepted = False
            for k in range(10):
                alpha = .5**k
                trial = C[i]+alpha*delta.reshape(C[i].shape)
                # |b_k|<=1，因此sum|c_k|给出整个区间的保守修正幅度界。
                if np.abs(trial).sum(axis=0).max() >= margin:
                    continue
                rr, _, pp, aa = segment_system(robot, T, base, i, trial, guard, scale, opts.rotation_weight, False)
                rt, _, _, _ = segment_system(robot, T, base, i, trial, train, scale, opts.rotation_weight, False)
                evaluations += len(train)+len(guard)
                if (rt@rt < r@r and rr@rr < old@old and
                        pp.max() <= old_ep.max()+1e-12 and aa.max() <= old_er.max()+1e-12):
                    C[i] = trial
                    accepted = True
                    break
            history.append(dict(segment=i, iteration=iteration, accepted=accepted,
                                guard_before_max_m=float(old_ep.max()),
                                guard_after_max_m=float(pp.max()) if accepted else float(old_ep.max()),
                                alpha=alpha if accepted else 0.))
            if not accepted:
                break
    # 全局缩减修正幅度，使有限采样的jerk开销处于预设预算内。
    grid = np.linspace(0, base.duration, 4001)
    base_jerk = base.spline(grid, 3)
    rms = np.sqrt(np.mean(base_jerk**2))
    peak = np.abs(base_jerk).max()
    factor = 1.
    for _ in range(21):
        curve = SegmentCurve(base, C*factor)
        jerk = curve.values(grid, 3)
        if (np.sqrt(np.mean(jerk**2)) <= opts.max_jerk_rms_ratio*rms+1e-12
                and np.abs(jerk).max() <= opts.max_jerk_peak_ratio*peak+1e-12):
            break
        factor *= .5
    else:
        factor = 0.
        curve = SegmentCurve(base, np.zeros_like(C))
    # 缩减后的非线性位姿误差重新检查；失败区段恢复为原曲线。
    for i in range(len(C)):
        old, _, op, ore = segment_system(robot, T, base, i, np.zeros_like(C[i]), guard, scale, opts.rotation_weight, False)
        rr, _, ep, er = segment_system(robot, T, base, i, curve.coefficients[i], guard, scale, opts.rotation_weight, False)
        evaluations += 2*len(guard)
        if rr@rr > old@old+1e-24 or ep.max() > op.max()+1e-12 or er.max() > ore.max()+1e-12:
            curve.coefficients[i] = 0.
    # 恢复区段后再次确认预算，不能把失效的局部接受拼成成功曲线。
    jerk = curve.values(grid, 3)
    if (np.sqrt(np.mean(jerk**2)) > opts.max_jerk_rms_ratio*rms+1e-12
            or np.abs(jerk).max() > opts.max_jerk_peak_ratio*peak+1e-12):
        curve = SegmentCurve(base, np.zeros_like(C))
        factor = 0.
    details = dict(global_scale=factor, pose_evaluations=evaluations,
                   jacobian_evaluations=sum(len(train) for _ in history),
                   modified_segments=int(np.count_nonzero(np.abs(curve.coefficients).max(axis=(1, 2)))),
                   conservative_joint_margin_rad=float(margin-np.abs(curve.coefficients).sum(axis=1).max()))
    return curve, history, details
