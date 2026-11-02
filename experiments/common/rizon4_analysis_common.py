# 共享模型接口：实验03—11
# 功能：按关节名称映射两库索引，读取限位并交叉核对FK与Jacobian。
# 输出：Robot接口、通用参数解析及实验元数据。

"""共享模型接口：名称映射、限位核对、Pinocchio FK 与位置 Jacobian。"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import mujoco
import pinocchio as pin

NAMES = [f"joint{i}" for i in range(1, 8)]


def parser(description, folder, samples):
    root = Path(__file__).resolve().parents[2]
    default = root / 'flexiv_rizon4/flexiv_rizon4.xml'
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", type=Path, default=default)
    p.add_argument("--out", type=Path, default=root / "results" / folder)
    p.add_argument("--samples", type=int, default=samples)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--view-index", type=int, default=None,
                   help="Optional zero-based sample to show in existing viewer")
    return p


class Robot:
    def __init__(self, path):
        self.path = path.resolve(strict=True)
        # 采用固定基座，保留 MJCF 中的固定变换。
        self.model = pin.buildModelFromMJCF(str(self.path))
        self.data = self.model.createData()
        self.mj = mujoco.MjModel.from_xml_path(str(self.path))
        self.md = mujoco.MjData(self.mj)
        if (self.model.nq, self.model.nv, self.mj.nq, self.mj.nv) != (7, 7, 7, 7):
            raise ValueError("Requires a fixed-base, exactly seven-hinge robot")
        # 分别保存两库的位置与速度索引，不能混用。
        self.iq, self.iv, self.mq, self.mv, bounds = [], [], [], [], []
        for name in NAMES:
            jid = self.model.getJointId(name)
            mid = mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_JOINT, name)
            if not (0 < jid < self.model.njoints) or mid < 0:
                raise ValueError(f"Missing joint {name}")
            joint = self.model.joints[jid]
            if (joint.nq, joint.nv) != (1, 1):
                raise ValueError(f"Expected scalar revolute joint: {name}")
            if self.mj.jnt_type[mid] != mujoco.mjtJoint.mjJNT_HINGE or not self.mj.jnt_limited[mid]:
                raise ValueError(f"Expected limited hinge: {name}")
            self.iq.append(joint.idx_q)
            self.iv.append(joint.idx_v)
            self.mq.append(int(self.mj.jnt_qposadr[mid]))
            self.mv.append(int(self.mj.jnt_dofadr[mid]))
            bounds.append(self.mj.jnt_range[mid].copy())
        # 限位取自编译后的 MuJoCo 模型，再与 Pinocchio 核对。
        self.lower, self.upper = np.asarray(bounds).T
        if not np.all(np.isfinite(bounds)) or not np.all(self.lower < self.upper):
            raise ValueError("Non-finite or invalid limits")
        if not (np.allclose(self.lower, self.model.lowerPositionLimit[self.iq], atol=1e-9, rtol=0)
                and np.allclose(self.upper, self.model.upperPositionLimit[self.iq], atol=1e-9, rtol=0)):
            raise ValueError("Pinocchio/MuJoCo joint limits disagree")
        if len(set(self.iq)) != 7 or len(set(self.iv)) != 7:
            raise ValueError("Joint mapping is not one-to-one")
        # 选取 link7 的 BODY 原点，不是质心或工具 TCP。
        self.frame = self.model.getFrameId("link7", pin.FrameType.BODY)
        self.body = mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_BODY, "link7")
        if self.frame >= self.model.nframes or self.body < 0:
            raise ValueError("Missing link7 body frame")

    def q(self, angles):
        angles = np.asarray(angles)
        if angles.shape != (7,) or not np.all(np.isfinite(angles)):
            raise ValueError("Expected seven finite angles in joint1..joint7 order")
        if np.any(angles < self.lower) or np.any(angles > self.upper):
            raise ValueError("Configuration exceeds joint limits")
        q = pin.neutral(self.model)
        # 外部七个角度按名称顺序传入，这里转换为模型配置顺序。
        q[self.iq] = angles
        return q

    def position(self, angles):
        pin.framesForwardKinematics(self.model, self.data, self.q(angles))
        return self.data.oMf[self.frame].translation.copy()

    def jacobian(self, angles):
        # 原点在 link7，坐标轴与世界系对齐，得到世界位置的导数。
        spatial = pin.computeFrameJacobian(self.model, self.data, self.q(angles),
                                           self.frame, pin.LOCAL_WORLD_ALIGNED)
        # 前三行是线速度，后三行是角速度；按关节速度索引取列。
        return np.asarray(spatial)[:3, self.iv].copy()

    def validate(self, seed):
        """用 MuJoCo 和中心有限差分交叉检查 FK 与位置 Jacobian。"""
        rng = np.random.default_rng(seed)
        errors = np.zeros(4)
        span = self.upper - self.lower
        for angles in rng.uniform(self.lower + .1*span, self.upper - .1*span, (12, 7)):
            pos = self.position(angles)
            rot = self.data.oMf[self.frame].rotation.copy()
            jac = self.jacobian(angles)
            self.md.qpos[self.mq] = angles
            mujoco.mj_forward(self.mj, self.md)
            jp, jr = np.zeros((3, 7)), np.zeros((3, 7))
            mujoco.mj_jacBody(self.mj, self.md, jp, jr, self.body)
            fd = np.zeros((3, 7))
            # 每次只微扰一个关节，其他关节不动，用中心差分核对一列。
            for i in range(7):
                d = np.zeros(7)
                d[i] = 1e-6
                fd[:, i] = (self.position(angles+d)-self.position(angles-d))/(2e-6)
            errors = np.maximum(errors, [np.max(np.abs(pos-self.md.xpos[self.body])),
                np.max(np.abs(rot-self.md.xmat[self.body].reshape(3, 3))),
                np.max(np.abs(jac-jp[:, self.mv])), np.max(np.abs(jac-fd))])
        if np.any(errors > 1e-6):
            raise RuntimeError(f"Frame/mapping/Jacobian validation failed: {errors}")
        return dict(zip(["fk_position_max_abs_m", "rotation_max_abs", "mj_jacobian_max_abs", "finite_difference_max_abs"], errors.tolist()))


def prepare(args):
    if args.samples < 2 or args.seed < 0:
        raise ValueError("samples >= 2 and seed >= 0 required")
    if args.view_index is not None and not 0 <= args.view_index < args.samples:
        raise ValueError("view-index must be in [0, samples)")
    robot = Robot(args.model)
    checks = robot.validate(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print("Validation:", checks)
    return robot, checks


def save_metadata(args, robot, checks, **extra):
    import matplotlib
    record = dict(model=str(robot.path), sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
                  seed=args.seed, samples=args.samples, joint_names=NAMES,
                  q_indices=robot.iq, velocity_indices=robot.iv,
                  limits_rad=np.column_stack((robot.lower, robot.upper)).tolist(),
                  frame="link7 BODY origin; MJCF world coordinates; fixed base",
                  reference="LOCAL_WORLD_ALIGNED; linear rows 0:3",
                  versions=dict(python=platform.python_version(), pinocchio=pin.__version__,
                                mujoco=mujoco.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
                  validation=checks, collision_filter=False, **extra)
    # 记录模型及代码哈希，便于追溯每次运行所用版本。
    record["xml_hashes"] = {str(p.relative_to(robot.path.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(robot.path.parent.rglob("*.xml"))}
    record["code_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob("*.py")}
    (args.out / "metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def view(args, robot, angles):
    if args.view_index is None:
        return
    import experiments.common.visualize_rizon4 as viewer
    # 显示时使用本次计算的同一模型，避免场景与计算不一致。
    viewer.SCENE_XML = robot.path
    viewer.show_configuration(robot.model, robot.q(angles[args.view_index]))
