import sys
from pathlib import Path
from dataclasses import dataclass, field

import mujoco
import numpy as np

from ur5e_workcell import build_ur5e_workcell_spec

COLORS = ["red", "blue", "yellow"]
BLOCK_SIZE = 0.04
BLOCK_HALF = BLOCK_SIZE / 2
TABLE_Z = 0.705 


CONTROL_FREQUENCY = 30

# 作業台上の置き場所エリア (中心 xy, 指示文での呼び方)
# 向きは俯瞰カメラから見た向き (手前 = -y, 右 = +x)
TABLE_REGIONS = {
    "center": (np.array([0.18, -0.04]), "between the two L markers"),
    "left":   (np.array([0.03, -0.08]), "on the left side of the table"),
    "right":  (np.array([0.36, -0.08]), "on the right side of the table"),
    "front":  (np.array([0.20, -0.24]), "at the front of the table"),
    "back":   (np.array([0.22,  0.10]), "at the back of the table"),
}
# 置き場所エリアの中心から、X・Y方向にそれぞれ最大±3cmずらす。
TARGET_RANDOM_OFFSET = 0.03

# 手首 (wrist_3_link) の目標姿勢: フランジ(+Y)が真下、+Z がワールド +Y、指はワールド X 方向に開閉
R_TOP_DOWN = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
])
HOME_TCP = np.array([-0.05, -0.05, 0.90])
AXES_SEED = np.array([-1.2, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent / "Step1-Environment"))

@dataclass
class EpisodeSpec:
    """1エピソードの設定 (乱数から作る)"""
    kind: str
    pick_color: str
    tray_xy: dict
    stacked_blosks: list = field(default_factory=list) 
    table_xy: np.ndarray = None
    region: str = "center"
    instruction: str = ""


class StackingEnv:
    def __init__(self, image_size=256):
        self.model = build_ur5e_workcell_spec().compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
        # 補助用の目印 (緑の目標点・赤い TCP 球) はカメラに写さない (誤った手がかりになるため)
        for name in ["tcp_site", "stack_target_site"]:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                m.site_rgba[sid, 3] = 0.0
        self.wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
        self.block_body = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{c}_block") for c in COLORS}
        self.block_qadr = {}
        self.block_dadr = {}
        for c in COLORS:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{c}_block_joint")
            self.block_qadr[c] = m.jnt_qposadr[j]
            self.block_dadr[c] = m.jnt_dofadr[j]
        self.substeps = max(1, int(round(1.0 / CONTROL_FREQUENCY / m.opt.timestep)))
        self.image_size = image_size
        self.renderer = None
        self.attached = None
        self.attach_offset = np.zeros(3)
        self.gripper_state = 0.0
        self.spec = None

    # ------------------------------------------------------------------ 基本
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def tcp_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def block_pos(self, color):
        return self.data.xpos[self.block_body[color]].copy()

    def _set_block(self, color, pos):
        a = self.block_qadr[color]
        self.data.qpos[a:a + 3] = pos
        self.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
        d = self.block_dadr[color]
        self.data.qvel[d:d + 6] = 0.0

    def _set_gripper_joints(self, closed):
        grip_closed_cmd = 0.012
        self.data.ctrl[6:8] = grip_closed_cmd if closed else 0.0

    # ------------------------------------------------------------------ IK
    def solve_ik(self, target_pos, axes_init, max_iters=60, tol=1e-4):
        """手首を真下に向けたまま TCP を target_pos へ (6自由度 DLS)。data の腕関節は元に戻す"""
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
            axes_delta = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d.qpos[:6] += np.clip(axes_delta, -0.1, 0.1) * 0.6
        axes = d.qpos[:6].copy()
        d.qpos[:6] = axes_backup
        mujoco.mj_forward(m, d)
        return axes

    # ------------------------------------------------------------------ リセット
    def reset(self, spec: EpisodeSpec, rng: np.random.Generator):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.attached = None
        self.gripper_state = 0.0
        self._set_gripper_joints(False)
        self.spec = spec

        # ブロック配置
        for c, xy in spec.tray_xy.items():
            self._set_block(c, np.array([xy[0], xy[1], 0.665]))
        for i, c in enumerate(spec.stacked_blosks):
            z = TABLE_Z + BLOCK_HALF + 2 * BLOCK_HALF * i
            self._set_block(c, np.array([spec.table_xy[0], spec.table_xy[1], z]))

        # 腕: ホーム付近 (少しランダム) に垂直姿勢で置く
        home = HOME_TCP + rng.uniform([-0.04, -0.04, -0.03], [0.04, 0.04, 0.03])
        axes_home = self.solve_ik(home, AXES_SEED, max_iters=300)
        d.qpos[:6] = axes_home
        d.ctrl[:6] = axes_home
        mujoco.mj_forward(m, d)

        # ブロックを落ち着かせる
        for _ in range(150):
            d.qpos[:6] = axes_home
            d.qvel[:6] = 0.0
            mujoco.mj_step(m, d)
        return self.get_state()

    # ------------------------------------------------------------------ 1ステップ
    def get_state(self):
        return np.concatenate([self.data.qpos[:6], [self.gripper_state]]).astype(np.float32)

    def step(self, action):
        """action = [目標関節6, グリッパー指令(>0.5で閉)] を 1/CONTROL_FREQUENCY 秒かけて実行"""
        m, d = self.model, self.data
        axes_target = np.asarray(action[:6], dtype=np.float64)
        want_closed = action[6] > 0.5

        # グリッパーの開閉
        if want_closed != bool(self.gripper_state):
            self.gripper_state = float(want_closed)
            self._set_gripper_joints(want_closed)
            if not want_closed:
                self.attached = None
        if want_closed and self.attached is None:
            self._try_attach()

        # 腕は目標関節角へ直線補間で動かす (URSim の servoJ 追従に相当)
        axes_start = d.qpos[:6].copy()
        for k in range(self.substeps):
            a = (k + 1) / self.substeps
            d.qpos[:6] = axes_start + a * (axes_target - axes_start)
            d.qvel[:6] = 0.0
            d.ctrl[:6] = d.qpos[:6]
            if self.attached is not None:
                mujoco.mj_kinematics(m, d)
                self._set_block(self.attached, d.site_xpos[self.site_id] + self.attach_offset)
            mujoco.mj_step(m, d)
        return self.get_state()

    def _try_attach(self):
        """指先付近 (物理指開き幅 ±3.8cm 以内、高さが指パッド範囲内) にあるブロックを固定する"""
        tcp = self.tcp_pos()
        best, best_dist = None, 1e9
        for c in COLORS:
            p = self.block_pos(c)
            dxy = np.linalg.norm(p[:2] - tcp[:2])
            dz = tcp[2] - p[2]          # ブロック中心は TCP より少し下
            if dxy < 0.036 and -0.015 < dz < 0.048 and dxy < best_dist:
                best, best_dist = c, dxy
        if best is not None:
            self.attached = best
            p = self.block_pos(best)
            # xy のずれは吸収して真下に固定 (見た目の自然さ優先)
            self.attach_offset = np.array([0.0, 0.0, p[2] - tcp[2]])

    # ------------------------------------------------------------------ 画像
    def render(self, camera):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    # ------------------------------------------------------------------ 成否判定
    def check_success(self):
        s = self.spec
        p = self.block_pos(s.pick_color)
        a = self.block_qadr[s.pick_color]
        quat = self.data.qpos[a + 3:a + 7]
        upright = abs(quat[0]) > 0.95       # ほぼ傾いていない
        if self.attached is not None:
            return False
        if s.kind == "place":
            # TARGET_RANDOM_OFFSET (±3cm) はカメラ上で不可視のためエリア中心/目標点から 6cm 以内ならエリア内配置成功
            region_center = TABLE_REGIONS[s.region][0]
            ok_xy = (np.linalg.norm(p[:2] - s.table_xy) < 0.055) or (np.linalg.norm(p[:2] - region_center) < 0.055)
            ok_z = abs(p[2] - (TABLE_Z + BLOCK_HALF)) < 0.010
        else:
            top = self.block_pos(s.stacked_blosks[-1])
            ok_xy = np.linalg.norm(p[:2] - top[:2]) < 0.025
            ok_z = abs(p[2] - (top[2] + 2 * BLOCK_HALF)) < 0.012
        # 土台のブロックが崩れていないこと
        base_ok = True
        for i, c in enumerate(s.stacked_blosks):
            base_position = self.block_pos(c)
            base_ok &= np.linalg.norm(base_position[:2] - s.table_xy) < 0.025
            base_ok &= abs(base_position[2] - (TABLE_Z + BLOCK_HALF + 2 * BLOCK_HALF * i)) < 0.010
        return bool(ok_xy and ok_z and upright and base_ok)


# ---------------------------------------------------------------------- エピソード生成
PLACE_TEMPLATES = [
    "pick up the {c} block and place it {r}",
    "put the {c} block {r}",
    "move the {c} block from the tray and put it {r}",
]
STACK_TEMPLATES = [
    "stack the {c} block on the {t} block",
    "put the {c} block on top of the {t} block",
    "place the {c} block onto the {t} block",
]


def sample_tray_positions(colors, rng):
    """トレイ内にブロックを並べる (指が壁や隣のブロックに当たらない間隔を確保)"""
    for _ in range(1000):
        center = np.array([-0.38, -0.08])
        xy = {c: center + rng.uniform([-0.06, -0.15], [0.06, 0.15]) for c in colors}
        cs = list(colors)
        ok = all(abs(xy[a][1] - xy[b][1]) >= 0.065 for i, a in enumerate(cs) for b in cs[i + 1:])
        if ok:
            return xy
    raise RuntimeError("tray sampling failed")


def sample_episode(rng: np.random.Generator) -> EpisodeSpec:
    kind_roll = rng.random()
    colors = list(rng.permutation(COLORS))
    region = rng.choice(list(TABLE_REGIONS.keys()))
    center, phrase = TABLE_REGIONS[region]
    table_xy = center + rng.uniform(-TARGET_RANDOM_OFFSET, TARGET_RANDOM_OFFSET, size=2)

    if kind_roll < 1 / 3:
        # place: 作業台は空、トレイに3個
        pick = colors[0]
        spec = EpisodeSpec("place", pick, sample_tray_positions(colors, rng), [], table_xy, region)
        spec.instruction = rng.choice(PLACE_TEMPLATES).format(c=pick, r=phrase)
    else:
        n_base = 1 if kind_roll < 2 / 3 else 2
        base = colors[:n_base]
        rest = colors[n_base:]
        pick = rest[0]
        spec = EpisodeSpec("stack", pick, sample_tray_positions(rest, rng), base, table_xy, region)
        spec.instruction = rng.choice(STACK_TEMPLATES).format(c=pick, t=base[-1])
    return spec
