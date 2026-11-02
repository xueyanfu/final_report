# 轨迹模块：实验10
# 功能：封装样条位置和解析导数，精化IK节点及起止边界条件。
# 输出：统一关节位置、速度和加速度参考。

"""参考 SciPy make_interp_spline；增加 Rizon4 起止约束和统一的 q/dq/ddq 接口。

开源接口：https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
引入五次样条曲线；这里的改进是端点约束、精化 IK 及前馈导数的一致性。
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline,make_interp_spline
from experiments.inverse_kinematics.rizon4_ik import IKOptions,solve_ik,frame_pose
from experiments.trajectory_tracking.experiments_07_heart_trajectory import interpolate,reference_position


class JointCurve:
    def __init__(self,times,angles,kind):
        self.times=np.asarray(times);self.angles=np.asarray(angles);self.kind=kind
        self.duration=float(times[-1]);self.spline=None
        self.slopes=np.diff(angles,axis=0)/np.diff(times)[:,None]
        self.approx_acc=np.gradient(np.gradient(angles,times,axis=0),times,axis=0)
        if kind=='cubic':
            # 开源默认三次样条作为中间对照，不添加静止端点条件。
            self.spline=CubicSpline(times,angles,axis=0)
        elif kind=='quintic':
            # 与静止保持段衔接时，速度、加速度同时为零。
            boundary=[(1,np.zeros(7)),(2,np.zeros(7))]
            # 五次 B 样条插值。
            self.spline=make_interp_spline(times,angles,k=5,bc_type=(boundary,boundary),axis=0)
        elif kind!='linear':raise ValueError(kind)

    def evaluate(self,t):
        if t<0:return self.angles[0].copy(),np.zeros(7),np.zeros(7)
        if t>=self.duration:return self.angles[-1].copy(),np.zeros(7),np.zeros(7)
        if self.spline is not None:
            return tuple(np.asarray(self.spline(t,nu)) for nu in range(3))
        i=max(0,min(np.searchsorted(self.times,t,side='right')-1,len(self.slopes)-1))
        # 线性位置、分段常速、中心差分近似加速度。
        return interpolate(t,self.times,self.angles),self.slopes[i],interpolate(t,self.times,self.approx_acc)

    def continuity(self):
        """报告节点速度跳变及静止衔接误差；线性段的近似加速度不是真导数。"""
        if self.spline is None:
            jumps=np.vstack([self.slopes[0],np.diff(self.slopes,axis=0),-self.slopes[-1]])
            return dict(max_velocity_jump_rad_s=float(np.abs(jumps).max()),acceleration_continuous=False)
        knots=np.unique(self.spline.t) if hasattr(self.spline,'t') else self.spline.x
        knots=knots[(knots>0)&(knots<self.duration)]
        eps=1e-9
        dv=max(float(np.max(np.abs(self.spline(t-eps,1)-self.spline(t+eps,1)))) for t in knots)
        da=max(float(np.max(np.abs(self.spline(t-eps,2)-self.spline(t+eps,2)))) for t in knots)
        endpoint_v=max(float(np.abs(self.spline(t,1)).max()) for t in [0,self.duration])
        endpoint_a=max(float(np.abs(self.spline(t,2)).max()) for t in [0,self.duration])
        return dict(max_velocity_jump_rad_s=max(dv,endpoint_v),interior_acceleration_two_sided_difference=da,
                    endpoint_speed_rad_s=endpoint_v,endpoint_acceleration_rad_s2=endpoint_a,
                    continuity_probe_epsilon_s=eps,acceleration_continuous=endpoint_a<1e-8)


def refine_nodes(robot,T,times,scale):
    '''
        原代码在 1e-6 m 终止阈值的小误差在二次求导时会放大
        我先提高逆运动学（IK）在每个离散时间节点上求得的关节角解的精度，从而减小后续计算关节加速度时的数值误差。
    '''
    current=np.array([0.,0.,0.,1.57,0.,0.,0.]);angles=[]
    # 提高精度到 1e-10 m
    opts=IKOptions(position_tolerance=1e-10,rotation_tolerance=1e-10,max_iterations=150)
    for t in times:
        target=pin.SE3(T.rotation,reference_position(t,T.translation,scale,times[-1]))
        result=solve_ik(robot,target,current,'pose',opts)
        if not result['success']:raise RuntimeError(f'Refined IK failed at {t}: {result["status"]}')
        current=result['angles'];angles.append(current.copy())
    return np.array(angles)
