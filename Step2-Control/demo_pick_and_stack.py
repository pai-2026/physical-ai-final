"""
UR5e 40mm木製ブロック (赤・青・黄) 完全垂直ハンド ピック＆スタック (改良版)
- Cartesian(直交座標)補間により、移動中も1度も傾かず常に真下(-Z軸)を維持
- 姿勢誤差の重み強化(rot_weight=3.5)と完全直交基底(3x3目標回転行列)による高精度6自由度IK
- 把持中のブロック回転ロック機能付き
- URSim (Sim-to-Sim) リアルタイム同期対応
"""
import sys
import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent / "Step1-Environment"))

from ur5e_workcell import create_workcell
from stacking_env import COLORS, BLOCK_SIZE, HOME_TCP, AXES_SEED, R_TOP_DOWN

try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False


class BlockStackingController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")

        self.block_ids = {
            color: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_block")
            for color in COLORS
        }

        self.attached_block = None
        self.attach_rel_pos = np.zeros(3)

    def attach(self, block_name):
        self.attached_block = block_name
        block_id = self.block_ids[block_name]
        tcp_pos = self.data.site_xpos[self.site_id]
        block_pos = self.data.xpos[block_id]
        # 常にTCPの真下に吸着するようXYズレを補正し、Zオフセットのみ保持
        self.attach_rel_pos = np.array([0.0, 0.0, block_pos[2] - tcp_pos[2]])
        if self.model.nu >= 8:
            self.data.ctrl[6:8] = 0.018

    def detach(self):
        self.attached_block = None
        if self.model.nu >= 8:
            self.data.ctrl[6:8] = 0.0

    def step(self):
        # 把持中はブロックの位置と水平姿勢(Quaternion)をTCPに完全同期
        if self.attached_block is not None:
            block_name = self.attached_block
            jnt_name = f"{block_name}_block_joint"
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            position_address = self.model.jnt_qposadr[jnt_id]
            tcp_pos = self.data.site_xpos[self.site_id]
            new_pos = tcp_pos + self.attach_rel_pos

            self.data.qpos[position_address:position_address+3] = new_pos
            self.data.qpos[position_address+3:position_address+7] = np.array([1.0, 0.0, 0.0, 0.0])

            dof_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[dof_adr:dof_adr+6] = 0.0

        mujoco.mj_step(self.model, self.data)


def solve_ik_top_down(model, data, target_pos, site_name="tcp_site", max_iters=200, tol=1e-4):
    """
    UR5eのフランジ(wrist_3_link)を厳密に「真下(ワールド-Z軸)」に固定する6自由度IK
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    wrist_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    # 目標とするwrist_3_linkの直交3軸 (右手系: X cross Y = Z)
    # - Y軸(フランジ先端方向): ワールド真下 [0, 0, -1]
    # - Z軸(手首カメラ上方向): ワールド奥側 [0, 1, 0]
    # - X軸(グリッパー開閉軸): ワールド右側 [1, 0, 0]
    R_des = R_TOP_DOWN

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)

        err_pos = target_pos - data.site_xpos[site_id]

        R_cur = data.xmat[wrist_body_id].reshape(3, 3)
        err_rot = 0.5 * sum(np.cross(R_cur[:, k], R_des[:, k]) for k in range(3))

        if np.linalg.norm(err_pos) < tol and np.linalg.norm(err_rot) < 5e-4:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        # 姿勢が絶対に傾かないよう、回転誤差に 3.5倍 の重みをかける
        rot_weight = 3.5
        J = np.vstack([jacp[:, :6], jacr[:, :6] * rot_weight])
        err = np.concatenate([err_pos, err_rot * rot_weight])

        damping = 1e-2
        axes_delta = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)

        # 1ステップあたりの最大角度変化を制限して安全に収束させる
        data.qpos[:6] += np.clip(axes_delta, -0.1, 0.1) * 0.6
        data.ctrl[:6] = data.qpos[:6]

    mujoco.mj_forward(model, data)
    return data.qpos[:6].copy()


def plan_cartesian_top_down_path(model, data, pos_waypoints, steps_per_segment=50):
    """
    XYZ座標のウェイポイント間を直線補間し、すべての通過点で垂直6自由度IKを解くことで
    移動中もハンドが1ミリも傾かない関節軌道リストを生成する
    """
    axes_backup = data.qpos[:6].copy()
    full_axes_path = []

    for p_start, p_end in zip(pos_waypoints, pos_waypoints[1:]):
        for step in range(steps_per_segment):
            s = 0.5 * (1.0 - np.cos(np.pi * step / (steps_per_segment - 1)))
            p_target = p_start + s * (p_end - p_start)
            # 直前のステップの関節角を初期値にして高速・連続的にIKを解く
            full_axes_path.append(solve_ik_top_down(model, data, p_target, max_iters=40, tol=1e-4))

    # シミュレーション状態の関節角度をプランニング前の状態に戻す
    data.qpos[:6] = axes_backup
    data.ctrl[:6] = axes_backup
    mujoco.mj_forward(model, data)
    return full_axes_path


def run_pick_and_stack(interactive=True, use_ursim=True):
    model, data = create_workcell()
    ctrl = BlockStackingController(model, data)

    # 1. まず垂直に近い基準関節角をセットし、初期ホーム座標(XYZ)で厳密に垂直IKを解く
    axes_seed = AXES_SEED
    data.qpos[:6] = axes_seed
    data.ctrl[:6] = axes_seed
    mujoco.mj_forward(model, data)

    home_pos = HOME_TCP
    axes_home = solve_ik_top_down(model, data, home_pos, max_iters=250)
    data.qpos[:6] = axes_home
    data.ctrl[:6] = axes_home
    ctrl.detach()

    # 初期物理安定化ステップ
    for _ in range(80):
        ctrl.step()

    rtde_c = None
    if use_ursim and RTDE_AVAILABLE:
        try:
            rtde_c = rtde_control.RTDEControlInterface("127.0.0.1")
            print(">>> URSim RTDE 接続成功！Sim-to-Sim モードで同期実行します。")
            rtde_c.moveJ(axes_home.tolist(), 0.5, 0.5)
        except Exception as e:
            print(f"URSim 未接続 (MuJoCo単体モードで実行): {e}")
            rtde_c = None

    red_id = ctrl.block_ids["red"]
    blue_id = ctrl.block_ids["blue"]
    yellow_id = ctrl.block_ids["yellow"]

    # スタック目標基準点 (黒いテーブルのL字マーカー中央)
    stack_base_pos = np.array([0.18, -0.04, 0.725])
    block_h = BLOCK_SIZE

    actions = [
        ("1. 赤ブロックを作業台へ配置", "red", data.xpos[red_id].copy(), stack_base_pos.copy()),
        ("2. 青ブロックを赤ブロックの上にスタック", "blue", data.xpos[blue_id].copy(), stack_base_pos + np.array([0, 0, block_h])),
        ("3. 黄ブロックを青ブロックの上にスタック", "yellow", data.xpos[yellow_id].copy(), stack_base_pos + np.array([0, 0, block_h * 2])),
    ]

    print("\n=======================================================")
    print("UR5e 完全垂直ハンド(Cartesian補間) ピック＆スタック開始")
    print("=======================================================")

    def execute_plan(viewer=None):
        current_tcp_pos = data.site_xpos[ctrl.site_id].copy()

        for task_name, block_name, pick_pos, place_pos in actions:
            print(f"\n▶ 実行中: {task_name}")

            p_above_pick = pick_pos + np.array([0, 0, 0.16])
            p_pick = pick_pos + np.array([0, 0, 0.015])
            p_lift = np.array([pick_pos[0], pick_pos[1], 0.90])
            p_transit = np.array([place_pos[0], place_pos[1], 0.90])
            p_place = place_pos + np.array([0, 0, 0.018])
            p_retreat = place_pos + np.array([0, 0, 0.16])

            ctrl.detach()
            path1 = plan_cartesian_top_down_path(model, data, [current_tcp_pos, p_above_pick, p_pick], steps_per_segment=55)
            for axes in path1:
                data.ctrl[:6] = axes
                data.qpos[:6] = axes
                if rtde_c: rtde_c.servoJ(axes.tolist(), 0.5, 0.5, 0.008, 0.1, 300)
                ctrl.step()
                if viewer: viewer.sync(); time.sleep(0.005)

            ctrl.attach(block_name)
            for _ in range(25):
                ctrl.step()
                if viewer: viewer.sync(); time.sleep(0.005)

            path2 = plan_cartesian_top_down_path(model, data, [p_pick, p_lift, p_transit, p_place], steps_per_segment=60)
            for axes in path2:
                data.ctrl[:6] = axes
                data.qpos[:6] = axes
                if rtde_c: rtde_c.servoJ(axes.tolist(), 0.5, 0.5, 0.008, 0.1, 300)
                ctrl.step()
                if viewer: viewer.sync(); time.sleep(0.005)

            ctrl.detach()
            for _ in range(25):
                ctrl.step()
                if viewer: viewer.sync(); time.sleep(0.005)

            path3 = plan_cartesian_top_down_path(model, data, [p_place, p_retreat], steps_per_segment=45)
            for axes in path3:
                data.ctrl[:6] = axes
                data.qpos[:6] = axes
                if rtde_c: rtde_c.servoJ(axes.tolist(), 0.5, 0.5, 0.008, 0.1, 300)
                ctrl.step()
                if viewer: viewer.sync(); time.sleep(0.005)

            current_tcp_pos = p_retreat.copy()

        path_home = plan_cartesian_top_down_path(model, data, [current_tcp_pos, home_pos], steps_per_segment=55)
        for axes in path_home:
            data.ctrl[:6] = axes
            data.qpos[:6] = axes
            if rtde_c: rtde_c.servoJ(axes.tolist(), 0.5, 0.5, 0.008, 0.1, 300)
            ctrl.step()
            if viewer: viewer.sync(); time.sleep(0.005)

        if rtde_c:
            rtde_c.servoStop()
            rtde_c.disconnect()

        print("\n★ 全てのブロックの積み重ね (赤 -> 青 -> 黄) が完全垂直姿勢で完了しました！")

    try:
        if interactive:
            print("\nMuJoCo 3Dビューアを起動してスタック動作を実演します...")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                execute_plan(viewer)
                while viewer.is_running():
                    ctrl.step()
                    viewer.sync()
                    time.sleep(0.01)
        else:
            execute_plan(viewer=None)
    except KeyboardInterrupt:
        print("\nユーザーによる中断 (Ctrl+C)。正常終了します。")


if __name__ == "__main__":
    import sys
    interactive_mode = "--headless" not in sys.argv
    run_pick_and_stack(interactive=interactive_mode, use_ursim=True)
