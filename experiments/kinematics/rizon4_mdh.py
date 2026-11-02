# MDH计算模块：实验11与逆解复核
# 功能：根据Craig MDH参数独立构造变换链及世界表达几何Jacobian。
# 输出：link7位姿和6×7 Jacobian；长度m、角度rad。

"""参数表的 Craig MDH 实现；与 XML 读取和 Pinocchio FK 独立。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import sys
import numpy as np

# 每行 a_(i-1), alpha_(i-1), d_i, theta_offset_i；长度m、角度rad。
PARAMETERS = np.array([
    [0, 0, 0, 0], [0, np.pi/2, .065, -np.pi],
    [0, np.pi/2, .395, -np.pi], [.020, -np.pi/2, .055, -np.pi],
    [.020, -np.pi/2, .385, -np.pi], [0, np.pi/2, .103, np.pi/2],
    [.110, np.pi/2, .015, 0],
])
WORLD_T_0 = np.eye(4)
WORLD_T_0[2, 3] = .365


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    t = np.eye(4)
    if axis == 'x':
        t[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == 'z':
        t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    else:
        raise ValueError('Only x/z rotations are used')
    return t


def translation(x=0., z=0.):
    t = np.eye(4)
    t[:3, 3] = [x, 0., z]
    return t


def chain(q, parameters=PARAMETERS, base=WORLD_T_0, standard=False):
    """返回 world→MDH 0..7。standard仅用于故意错误的约定对照。"""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('Expected seven finite joint angles')
    frames = [np.array(base, copy=True)]
    for qi, (a, alpha, d, offset) in zip(q, parameters):
        fixed = rotation('x', alpha) @ translation(x=a)
        joint = rotation('z', qi+offset) @ translation(z=d)
        frames.append(frames[-1] @ (joint @ fixed if standard else fixed @ joint))
    return np.array(frames)


def fk(q, parameters=PARAMETERS, base=WORLD_T_0, standard=False):
    return chain(q, parameters, base, standard)[-1]


def jacobian(q):
    """世界表达几何Jacobian，前3行线速度、后3行角速度。

    Craig MDH 的转轴是当前第i系z轴，不能套用标准DH的上一系z轴。
    原点可沿同一轴移动，故可使用完成A_i后的MDH原点。
    """
    frames = chain(q)
    p = frames[-1, :3, 3]
    axes = frames[1:, :3, 2]
    origins = frames[1:, :3, 3]
    return np.vstack((np.cross(axes, p-origins).T, axes.T))


def pybotics_model():
    """直接调用未修改的Pybotics；其参数列序是alpha,a,theta,d。"""
    local = Path(__file__).resolve().parents[2]/'.deps'
    if local.is_dir():
        sys.path.insert(0, str(local))
    try:
        from pybotics.robot import Robot
        from pybotics.kinematic_chain import MDHKinematicChain
    except ImportError as exc:
        raise ImportError('Install requirements-mdh.txt into the active environment') from exc
    return Robot(MDHKinematicChain(PARAMETERS[:, [1, 0, 3, 2]]),
                 world_frame=WORLD_T_0.copy())
