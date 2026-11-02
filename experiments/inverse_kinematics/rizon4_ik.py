# 共享求解器：实验06—10的逆运动学
# 功能：实现两类任务残差、加权DLS、候选阻尼、限步、回溯及限位。
# 输出：关节解、状态、实际误差及迭代记录；局部失败不证明不可达。


"""
Flexiv Rizon4 七自由度机械臂数值逆运动学求解器

参考代码：
Pinocchio 官方 inverse-kinematics.py
https://github.com/stack-of-tasks/pinocchio/blob/devel/examples/inverse-kinematics.py

基础方法：
采用官方示例中的 SE(3) 位姿误差、Jacobian 及阻尼最小二乘法（DLS）。

相对官方示例的项目扩展：

[改进1] 双任务 IK
支持仅位置 IK（3维）和完整位姿 IK（6维），
并对位置误差和姿态误差设置独立收敛条件。

[改进2] 多阻尼候选策略
由单一固定阻尼扩展为多个阻尼候选，
依次尝试不同阻尼系数，寻找能够降低目标函数的更新。

[改进3] 关节更新步长限制与回溯线搜索
限制单次最大关节更新量，并逐级减小步长，
降低大步长更新导致误差上升的风险。

[改进4] 关节限位处理
将候选关节角投影至模型允许范围，
避免输出超出关节限位的迭代结果。

[改进5] 位置与姿态误差加权
在完整位姿 IK 中设置旋转权重，
平衡不同量纲的平移残差与旋转残差。

[验证扩展] 数值导数与迭代记录
通过中心有限差分检查任务 Jacobian，
并记录每次迭代的误差、阻尼、步长及关节角。
说明：
上述改进属于已有数值优化方法的工程组合，
不宣称提出新的 DLS 数学算法。
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from dataclasses import dataclass
import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class IKOptions:
    max_iterations: int = 250
    position_tolerance: float = 1e-4  # m
    rotation_tolerance: float = 1e-3  # rad；仅位置任务不使用该阈值
    rotation_weight: float = 0.2  # m/rad；平衡位置与旋转残差
    max_joint_step: float = 0.2  # rad；单关节最大更新量
    damping: tuple = (1e-3, 1e-2, 1e-1, 1.0)
    backtracking_steps: int = 10

    def validate(self):
        if self.max_iterations < 0 or self.backtracking_steps < 1:
            raise ValueError('Invalid iteration/backtracking count')
        for value in (self.position_tolerance, self.rotation_tolerance,
                      self.rotation_weight, self.max_joint_step, *self.damping):
            if not np.isfinite(value) or value <= 0:
                raise ValueError('All tolerances, weights, steps and damping must be positive')
        if not self.damping:
            raise ValueError('At least one damping value is required')


def frame_pose(robot, angles):
    """返回 link7 原点的世界位姿；输入角度按 joint1～joint7 排列。"""
    pin.framesForwardKinematics(robot.model, robot.data, robot.q(angles))
    return robot.data.oMf[robot.frame].copy()


def task(robot, angles, target, mode, rotation_weight, jacobian=True):
    current = frame_pose(robot, angles)
    position_error = float(np.linalg.norm(current.translation-target.translation))
    rotation_error = float(np.linalg.norm(pin.log3(current.rotation.T@target.rotation)))
    # [改进1] 将官方完整位姿 IK 扩展为两种求解任务：
    # position：只约束末端三维位置；
    # pose：同时约束末端位置和姿态。
    # 用于研究增加姿态约束对收敛性能的影响。
    if mode == 'position':
        error = current.translation-target.translation
        J = robot.jacobian(angles) if jacobian else None
    elif mode == 'pose':
        relative = current.actInv(target)
        error = pin.log6(relative).vector.copy()
        if jacobian:
            local = pin.computeFrameJacobian(robot.model, robot.data, robot.q(angles),
                                             robot.frame, pin.LOCAL)
            # 修正 log6 残差的导数，使 Jacobian 与误差定义一致。
            J = (-pin.Jlog6(relative.inverse())@local)[:,robot.iv]
            # [改进5] 位置与姿态残差加权
            # 对旋转残差及其 Jacobian 同时加权，
            # 调节位置误差和姿态误差在优化中的相对影响。
            J[3:,:] *= rotation_weight
        else:
            J = None
        error[3:] *= rotation_weight
    else:
        raise ValueError('mode must be position or pose')
    return np.asarray(error), J, position_error, rotation_error


def solve_ik(robot, target, initial_angles, mode='pose', options=None):
    """求解局部 IK；迭代失败不代表目标不可达。

    target：link7 原点的世界位姿；initial_angles：按名称排序的七个角度（rad）。
    返回关节角、收敛状态、物理误差和迭代记录；不检查碰撞或生成运动时间律。
    """
    opts = options or IKOptions()
    opts.validate()
    a = np.asarray(initial_angles,dtype=float).copy()
    robot.q(a)  # 先检查初值是否合法，不静默截断初值
    R = np.asarray(target.rotation)
    if (not np.all(np.isfinite(target.homogeneous)) or
        not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or
        not np.isclose(np.linalg.det(R),1.0,atol=1e-7,rtol=0)):
        raise ValueError('Target must contain finite translation and a valid rotation')
    history=[]
    last_damping=last_alpha=last_step=0.0
    status='max_iterations'
    for iteration in range(opts.max_iterations+1):
        error,J,ep,er=task(robot,a,target,mode,opts.rotation_weight)
        cost=float(error@error/2)
        svals=np.linalg.svd(J,compute_uv=False)
        history.append(dict(iteration=iteration,position_error_m=ep,rotation_error_rad=er,
                            cost=cost,sigma_min=float(svals[-1]),damping=last_damping,
                            alpha=last_alpha,step_inf_rad=last_step,angles=a.copy()))
        # 依据实际位置和旋转误差判定成功，不以加权目标值代替。
        if ep<=opts.position_tolerance and (mode=='position' or er<=opts.rotation_tolerance):
            status='success'
            break
        if iteration==opts.max_iterations:
            break
        accepted=False
        # [改进2] 多阻尼候选 DLS
        # 不再只使用单一固定阻尼，而是依次尝试不同阻尼系数。
        # 通过候选步的目标函数值筛选可接受的更新。
        # 目的是提高局部迭代的稳定性。
        for damping in opts.damping:
            # 阻尼最小二乘步：抑制接近奇异构型时的过大更新。
            delta=-J.T@np.linalg.solve(J@J.T+damping*damping*np.eye(len(error)),error)
            # [改进3.1] 限制单次关节角更新幅度
            # 防止 DLS 在 Jacobian 接近奇异时产生过大的关节更新。
            delta *= min(1.0,opts.max_joint_step/max(np.max(np.abs(delta)),1e-30))
            # 将按关节名称排列的更新量写入模型速度索引。
            velocity=np.zeros(robot.model.nv)
            velocity[robot.iv]=delta
            for k in range(opts.backtracking_steps):
                # [改进3.2] 引入回溯线搜索
                # 将候选步长依次缩小为 1、1/2、1/4……
                # 只接受使加权残差目标函数下降的更新。
                alpha=0.5**k
                # 用 Pinocchio integrate 更新配置，再投影到合法角度范围。
                qnew=pin.integrate(robot.model,robot.q(a),alpha*velocity)
                # [改进4] 关节限位投影
                # 将每次候选更新约束在 Rizon4 的合法关节角范围内。
                # 避免数值求解过程中产生超限关节配置。
                # 不包含碰撞检测。
                candidate=np.clip(qnew[robot.iq],robot.lower,robot.upper)
                step=float(np.max(np.abs(candidate-a)))
                if step<1e-12:
                    continue
                trial,_,_,_=task(robot,candidate,target,mode,opts.rotation_weight,False)
                # 限位投影后重新计算目标值，只接受使目标值下降的步。
                if float(trial@trial/2)<cost:
                    a=candidate
                    last_damping,last_alpha,last_step=float(damping),alpha,step
                    accepted=True
                    break
            if accepted:
                break
        # 所有候选步均无改进时报告停滞，不据此判定目标不可达。
        if not accepted:
            status='stalled'
            break
    return dict(angles=a,success=status=='success',status=status,
                iterations=len(history)-1,position_error_m=history[-1]['position_error_m'],
                rotation_error_rad=history[-1]['rotation_error_rad'],history=history)


def check_task_derivatives(robot,seed=42):
    """用中心有限差分验证位置及完整位姿残差的 Jacobian。"""
    rng=np.random.default_rng(seed)
    span=robot.upper-robot.lower
    worst={'position':0.0,'pose':0.0}
    for _ in range(6):
        a=rng.uniform(robot.lower+.2*span,robot.upper-.2*span)
        target=frame_pose(robot,np.clip(a+rng.normal(0,.2,7),robot.lower,robot.upper))
        for mode in worst:
            _,J,_,_=task(robot,a,target,mode,.2)
            fd=np.zeros_like(J)
            for i in range(7):
                d=np.zeros(7); d[i]=1e-6
                plus=task(robot,a+d,target,mode,.2,False)[0]
                minus=task(robot,a-d,target,mode,.2,False)[0]
                fd[:,i]=(plus-minus)/(2e-6)
            worst[mode]=max(worst[mode],float(np.max(np.abs(J-fd))))
    if max(worst.values())>1e-6:
        raise RuntimeError(f'Residual derivative check failed: {worst}')
    return worst
