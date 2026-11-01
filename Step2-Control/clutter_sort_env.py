import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import mujoco
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent / "Step1-Environment"))

from ur5e_clutter_workcell import (
    build_ur5e_clutter_spec,
    SORTING_STATIONS,
    BIN_CENTER,
    OBJECT_NAMES,
)

from stacking_env import CONTROL_FREQUENCY
HOME_TCP = np.array([0.15, -0.28, 0.84])
AXES_SEED = np.array([-0.45, -1.35, 1.65, -1.85, -1.57, 0.0])

R_TOP_DOWN = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])



@dataclass
class ClutterEpisodeSpec:
    pick_obj: str  # "pen" | "bottle" | "tape" | "tissue"
    target_station: str  # "pen_tray" | "bottle_zone" | "tape_bin" | "tissue_spot"
    instruction: str
    bin_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    bin_quats: Dict[str, np.ndarray] = field(default_factory=dict)


# 各オブジェクトの許容把持半径と高さ許容値
OBJ_GRASP_PROFILES = {
    "pen": {"radius_xy": 0.060, "dz_min": -0.025, "dz_max": 0.045},
    "bottle": {"radius_xy": 0.050, "dz_min": -0.020, "dz_max": 0.090},
    "tape": {"radius_xy": 0.055, "dz_min": -0.025, "dz_max": 0.045},
    "tissue": {"radius_xy": 0.065, "dz_min": -0.030, "dz_max": 0.055},
}

# 仕分け成功判定用の許容距離 (各ステーション中心からのXY平面距離 & 高さ)
STATION_SUCCESS_TOLERANCES = {
    "pen_tray": {"dxy_max": 0.060, "z_min": 0.705, "z_max": 0.750},
    "bottle_zone": {"dxy_max": 0.060, "z_min": 0.705, "z_max": 0.820},  # ボトル全高130mmを考慮
    "tape_bin": {"dxy_max": 0.065, "z_min": 0.705, "z_max": 0.755},
    "tissue_spot": {"dxy_max": 0.080, "z_min": 0.705, "z_max": 0.760},  # 卓上マット外寸130mm x 180mmを考慮
}


class ClutterSortEnv:
    """UR5e ばら積みピッキング・仕分け整理シミュレーション環境"""

    def __init__(self, image_size: int = 512):
        self.image_size = image_size
        self.spec_builder = build_ur5e_clutter_spec()
        self.model = self.spec_builder.compile()
        self.data = mujoco.MjData(self.model)

        m = self.model
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        self.wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        self.obj_bodies = {obj: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{obj}_obj") for obj in OBJECT_NAMES}
        self.obj_qposadr = {}
        self.obj_dofadr = {}
        for obj in OBJECT_NAMES:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_joint")
            self.obj_qposadr[obj] = m.jnt_qposadr[j]
            self.obj_dofadr[obj] = m.jnt_dofadr[j]

        # サイトマーカーの透明化 (カメラ画像に影響を与えないようにする)
        m.site_rgba[:, 3] = 0.0

        self.substeps = max(1, int(round(1.0 / CONTROL_FREQUENCY / m.opt.timestep)))
        self.renderer = None
        self.attached: Optional[str] = None
        self.attach_offset = np.zeros(3)
        self.gripper_state = 0.0  # 0.0 = 開, 1.0 = 閉
        self.current_spec: Optional[ClutterEpisodeSpec] = None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def obj_pos(self, obj_name: str) -> np.ndarray:
        return self.data.xpos[self.obj_bodies[obj_name]].copy()

    def _set_obj_pose(self, obj_name: str, pos: np.ndarray, quat: np.ndarray):
        position_address = self.obj_qposadr[obj_name]
        self.data.qpos[position_address:position_address + 3] = pos
        self.data.qpos[position_address + 3:position_address + 7] = quat
        dadr = self.obj_dofadr[obj_name]
        self.data.qvel[dadr:dadr + 6] = 0.0

    def _set_gripper_joints(self, closed: bool):
        grip_closed_cmd = 0.048
        self.data.ctrl[6:8] = grip_closed_cmd if closed else 0.0

    def solve_ik(self, target_pos: np.ndarray, axes_init: np.ndarray, max_iters: int = 80, tol: float = 1e-4) -> np.ndarray:
        """手首真下向きで目標TCPへIKを解く"""
        m, d = self.model, self.data
        axes_backup = d.qpos[:6].copy()
        d.qpos[:6] = axes_init
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        rot_w = 3.5

        for _ in range(max_iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = target_pos - d.site_xpos[self.site_id]
            R = d.xmat[self.wrist_id].reshape(3, 3)
            err_r = 0.5 * sum(np.cross(R[:, k], R_TOP_DOWN[:, k]) for k in range(3))
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 5e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6] * rot_w])
            e = np.concatenate([err_p, err_r * rot_w])
            axes_delta = np.linalg.lstsq(J, e, rcond=1e-3)[0]
            d.qpos[:6] += np.clip(axes_delta, -0.25, 0.25)

        axes_result = d.qpos[:6].copy()
        d.qpos[:6] = axes_backup
        mujoco.mj_kinematics(m, d)
        return axes_result

    def _try_attach(self):
        """把持判定: TCPが指定対象物に近い場合に把持吸着を有効化"""
        if self.current_spec is None:
            return
        target = self.current_spec.pick_obj
        prof = OBJ_GRASP_PROFILES[target]

        p_tcp = self.tcp_pos()
        p_obj = self.obj_pos(target)
        dxy = np.linalg.norm(p_tcp[:2] - p_obj[:2])
        dz = p_tcp[2] - p_obj[2]

        if dxy < prof["radius_xy"] and (prof["dz_min"] <= dz <= prof["dz_max"]):
            self.attached = target
            self.attach_offset = p_obj - p_tcp

    def reset(self, spec: Optional[ClutterEpisodeSpec] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if spec is None:
            spec = sample_clutter_episode(rng)
        self.current_spec = spec

        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)

        axes_home = self.solve_ik(HOME_TCP, AXES_SEED, max_iters=200)
        d.qpos[:6] = axes_home
        d.ctrl[:6] = axes_home
        self.gripper_state = 0.0
        self._set_gripper_joints(False)
        self.attached = None

        for obj in OBJECT_NAMES:
            self._set_obj_pose(obj, spec.bin_positions[obj], spec.bin_quats[obj])

        mujoco.mj_forward(m, d)

        # 物理シミュレーションを回してばら積みオブジェクトを着地・安定化
        for _ in range(120):
            d.qpos[:6] = axes_home
            d.qvel[:6] = 0.0
            d.ctrl[:6] = axes_home
            mujoco.mj_step(m, d)

        return self.get_state()

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[:6], [self.gripper_state]]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """action: [6関節目標角, グリッパー開閉指令(>0.5で閉)]"""
        m, d = self.model, self.data
        axes_target = np.asarray(action[:6], dtype=np.float64)
        want_closed = action[6] > 0.5

        if want_closed != bool(self.gripper_state):
            self.gripper_state = float(want_closed)
            self._set_gripper_joints(want_closed)
            if not want_closed:
                self.attached = None
        if want_closed and self.attached is None:
            self._try_attach()

        axes_start = d.qpos[:6].copy()
        for k in range(self.substeps):
            alpha = (k + 1) / self.substeps
            d.qpos[:6] = axes_start + alpha * (axes_target - axes_start)
            d.qvel[:6] = 0.0
            d.ctrl[:6] = d.qpos[:6]
            if self.attached is not None:
                mujoco.mj_kinematics(m, d)
                self._set_obj_pose(
                    self.attached,
                    d.site_xpos[self.site_id] + self.attach_offset,
                    self.current_spec.bin_quats.get(self.attached, np.array([1.0, 0.0, 0.0, 0.0])),
                )
            mujoco.mj_step(m, d)

        success = self.check_success()
        return self.get_state(), float(success), False, {"success": success}

    def render_overhead(self) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self.renderer.update_scene(self.data, camera="overhead_camera")
        return self.renderer.render()

    def render_wrist(self) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self.renderer.update_scene(self.data, camera="wrist_camera")
        return self.renderer.render()

    def check_success(self) -> bool:
        """仕分け目標ステーションに対象物が正しく搬送・リリースされたかを判定"""
        if self.current_spec is None or self.attached is not None:
            return False

        obj = self.current_spec.pick_obj
        st = self.current_spec.target_station
        p_obj = self.obj_pos(obj)
        p_target = SORTING_STATIONS[st]["target_pos"]
        tol = STATION_SUCCESS_TOLERANCES[st]

        dxy = np.linalg.norm(p_obj[:2] - p_target[:2])
        z_ok = (tol["z_min"] <= p_obj[2] <= tol["z_max"])

        return bool(dxy <= tol["dxy_max"] and z_ok)



DEFAULT_CLUTTER_TEMPLATES = {
    "pen": [
        "pick up the pen and place it in the pen tray",
        "put the pen into the pen organizer tray",
        "sort the pen from the bin into the pen tray",
    ],
    "bottle": [
        "pick up the plastic bottle and put it into the bottle recycling zone",
        "sort the plastic bottle into the recycling area",
        "move the bottle from the bin to the bottle zone",
    ],
    "tape": [
        "pick up the packing tape and place it into the tape bin",
        "put the tape into the tape storage bin",
        "sort the roll of tape into the tool tray",
    ],
    "tissue": [
        "pick up the tissue box and place it on the tissue station",
        "put the tissue box onto the desk tissue spot",
        "sort the tissue box into the designated tissue station",
    ],
}

OBJ_TO_STATION = {
    "pen": "pen_tray",
    "bottle": "bottle_zone",
    "tape": "tape_bin",
    "tissue": "tissue_spot",
}


def sample_clutter_episode(rng: np.random.Generator) -> ClutterEpisodeSpec:
    """ばら積みコンテナ内に4つのオブジェクトをランダム配置し、仕分けタスクを生成"""
    pick_obj = rng.choice(OBJECT_NAMES)
    target_station = OBJ_TO_STATION[pick_obj]
    instruction = rng.choice(DEFAULT_CLUTTER_TEMPLATES[pick_obj])

    # ばら積み配置 (BIN_CENTER = [-0.38, -0.06, 0.64])
    # Y方向スロットにジッターを加える
    y_slots = rng.permutation([-0.14, -0.05, 0.04, 0.13])

    bin_positions = {}
    bin_quats = {}

    for i, obj in enumerate(OBJECT_NAMES):
        x = BIN_CENTER[0] + rng.uniform(-0.04, 0.04)
        y = BIN_CENTER[1] + (y_slots[i] + rng.uniform(-0.02, 0.02))

        if obj == "pen":
            # ペンは横倒し + Z軸まわりのランダム回転 (±30度)
            angle = rng.uniform(-0.5, 0.5)
            # base quat [0.7071, 0.7071, 0, 0] に Z回転を合成
            cz, sz = np.cos(angle / 2.0), np.sin(angle / 2.0)
            pen_quat = np.array([0.7071 * cz, 0.7071 * cz, 0.7071 * sz, -0.7071 * sz])
            pen_quat /= np.linalg.norm(pen_quat)
            bin_positions[obj] = np.array([x, y, 0.665])
            bin_quats[obj] = pen_quat
        elif obj == "bottle":
            # ペットボトルは直立 + 微小傾斜
            tilt = rng.uniform(-0.05, 0.05, size=2)
            bottle_quat = np.array([1.0, tilt[0], tilt[1], 0.0])
            bottle_quat /= np.linalg.norm(bottle_quat)
            bin_positions[obj] = np.array([x, y, 0.70])
            bin_quats[obj] = bottle_quat
        else:
            # テープとティッシュ箱は平置き
            z = 0.665 if obj == "tape" else 0.680
            bin_positions[obj] = np.array([x, y, z])
            bin_quats[obj] = np.array([1.0, 0.0, 0.0, 0.0])

    return ClutterEpisodeSpec(
        pick_obj=pick_obj,
        target_station=target_station,
        instruction=instruction,
        bin_positions=bin_positions,
        bin_quats=bin_quats,
    )
