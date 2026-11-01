import os
import mujoco
import numpy as np
import cv2
from robot_descriptions.ur5e_mj_description import MJCF_PATH

def build_ur5e_workcell_spec():
    """
    撮影画像をもとにした作業環境を MjSpec で構築する。
    - UR5e本体 (mujoco_menagerie 公式)
    - 40mmブロック把持用パラレルグリッパー (開閉アクチュエータ付き)
    - 手首カメラ (指先と把持対象を常時捉える画角)
    - トレイ内部: 40mm四方の木製ブロック3個 (赤・青・黄)
    - メイン作業台 (黒マット天板 + L字マーカー2個 + スタック目標エリア)
    - サイドテーブル (木目天板 + 水色パーツトレイ)
    - 俯瞰カメラ (写真の視点を再現)
    - 上部リングライト (白飛び抑制設定)
    """
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    spec.modelname = "ur5e_block_stacking_workcell"
    spec.visual.global_.offwidth = 1024
    spec.visual.global_.offheight = 1024

    # UR5eのベース位置・向き
    # 架台高さ Z=0.70m, 奥 Y=0.35m, X=-0.05m
    ur_base = spec.body("base")
    if ur_base is not None:
        ur_base.pos = [-0.05, 0.35, 0.70]
        ur_base.quat = [0.7071068, 0, 0, -0.7071068]

    # 手首(wrist_3_link): 2指パラレルグリッパー + 手首カメラ + TCPサイト
    wrist = spec.body("wrist_3_link")
    if wrist is not None:
        # 手首マウントカメラ (Intel RealSense D435相当)
        # グリッパーの指先と把持対象を斜め上からクリアに見下ろす
        wrist.add_camera(
            name="wrist_camera",
            pos=[0.0, 0.06, 0.07],
            quat=[0.683, 0.683, -0.183, 0.183],
            fovy=62
        )

        # グリッパーベースボディ
        gripper_base = wrist.add_body(name="gripper_base", pos=[0.0, 0.04, 0.0])
        # グリッパー本体ハウジング
        gripper_base.add_geom(
            name="gripper_housing",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.02, 0.0],
            size=[0.042, 0.025, 0.025],
            rgba=[0.18, 0.18, 0.20, 1.0],
            mass=0.35
        )

        # 左指 (スライドジョイントで外側/内側に開閉)
        # 初期開き幅: 中心から X=-0.038m (両指間隔 76mm, 40mmブロックを余裕でカバー)
        finger_l = gripper_base.add_body(name="finger_left", pos=[-0.038, 0.055, 0.0])
        finger_l.add_joint(
            name="finger_left_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[0.0, 0.022],
            damping=5.0
        )
        finger_l.add_geom(
            name="finger_l_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.025, 0.0],
            size=[0.005, 0.028, 0.015],
            rgba=[0.25, 0.25, 0.28, 1.0],
            mass=0.04,
            friction=[2.0, 0.01, 0.0002]
        )
        # 先端ラバーパッド (高摩擦)
        finger_l.add_geom(
            name="finger_l_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.004, 0.035, 0.0],
            size=[0.002, 0.016, 0.014],
            rgba=[0.85, 0.65, 0.10, 1.0],
            friction=[2.5, 0.01, 0.0002]
        )

        # 右指 (スライドジョイント)
        finger_r = gripper_base.add_body(name="finger_right", pos=[0.038, 0.055, 0.0])
        finger_r.add_joint(
            name="finger_right_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[-1, 0, 0],
            range=[0.0, 0.022],
            damping=5.0
        )
        finger_r.add_geom(
            name="finger_r_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.025, 0.0],
            size=[0.005, 0.028, 0.015],
            rgba=[0.25, 0.25, 0.28, 1.0],
            mass=0.04,
            friction=[2.0, 0.01, 0.0002]
        )
        # 先端ラバーパッド (高摩擦)
        finger_r.add_geom(
            name="finger_r_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.004, 0.035, 0.0],
            size=[0.002, 0.016, 0.014],
            rgba=[0.85, 0.65, 0.10, 1.0],
            friction=[2.5, 0.01, 0.0002]
        )

        # 指先中心 (TCP: Tool Center Point)
        wrist.add_site(
            name="tcp_site",
            pos=[0.0, 0.13, 0.0],
            size=[0.010, 0.010, 0.010],
            rgba=[1.0, 0.1, 0.1, 0.8]
        )

    # グリッパーアクチュエータの追加 (左右対称制御)
    act_l = spec.add_actuator(name="gripper_left", trntype=mujoco.mjtTrn.mjTRN_JOINT, target="finger_left_joint")
    act_l.ctrlrange = [0.0, 0.022]
    act_l.set_to_position(kp=400.0, kv=20.0)

    act_r = spec.add_actuator(name="gripper_right", trntype=mujoco.mjtTrn.mjTRN_JOINT, target="finger_right_joint")
    act_r.ctrlrange = [0.0, 0.022]
    act_r.set_to_position(kp=400.0, kv=20.0)

    world = spec.worldbody

    # 床面
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.5, 2.5, 0.05],
        rgba=[0.38, 0.38, 0.40, 1.0]
    )

    # 天井リングライト (白飛び防止)
    world.add_light(
        name="ring_light",
        pos=[0.10, 0.0, 1.50],
        dir=[0, 0, -1],
        diffuse=[0.65, 0.65, 0.65],
        specular=[0.10, 0.10, 0.10],
        ambient=[0.30, 0.30, 0.30]
    )
    # 環境補助ライト
    world.add_light(
        name="ambient_fill",
        pos=[-0.5, -0.5, 1.8],
        dir=[0.3, 0.3, -0.9],
        diffuse=[0.30, 0.30, 0.30],
        specular=[0.05, 0.05, 0.05]
    )

    # 俯瞰カメラ (Overhead Camera: 写真の視点)
    world.add_camera(
        name="overhead_camera",
        pos=[0.25, -0.75, 1.30],
        xyaxes=[0.8944, 0.4472, 0.0, -0.2857, 0.5715, 0.7693],
        fovy=58
    )

    # UR5e用アルミフレーム架台
    mount = world.add_body(name="alu_robot_stand", pos=[-0.05, 0.35, 0.35])
    mount.add_geom(
        name="stand_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.14, 0.14, 0.35],
        rgba=[0.75, 0.75, 0.78, 1.0]
    )

    # 右側: 黒いマットのメイン作業台 (アルミフレーム + 黒天板)
    black_table = world.add_body(name="black_worktable", pos=[0.20, -0.06, 0.35])
    black_table.add_geom(
        name="black_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.34],
        size=[0.34, 0.34, 0.015],
        rgba=[0.12, 0.12, 0.13, 1.0]
    )
    black_table.add_geom(
        name="black_frame",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0],
        size=[0.32, 0.32, 0.34],
        rgba=[0.72, 0.73, 0.75, 1.0]
    )

    # L字テープマーカー 1 (奥側)
    black_table.add_geom(
        name="l_marker_1_h",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.04, 0.08, 0.356],
        size=[0.045, 0.010, 0.001],
        rgba=[0.32, 0.32, 0.34, 1.0]
    )
    black_table.add_geom(
        name="l_marker_1_v",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.075, 0.115, 0.356],
        size=[0.010, 0.045, 0.001],
        rgba=[0.32, 0.32, 0.34, 1.0]
    )

    # L字テープマーカー 2 (手前側)
    black_table.add_geom(
        name="l_marker_2_h",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.04, -0.08, 0.356],
        size=[0.045, 0.010, 0.001],
        rgba=[0.32, 0.32, 0.34, 1.0]
    )
    black_table.add_geom(
        name="l_marker_2_v",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.075, -0.045, 0.356],
        size=[0.010, 0.045, 0.001],
        rgba=[0.32, 0.32, 0.34, 1.0]
    )

    # 作業台上のスタック目標エリア (L字マーカーで囲まれたエリア)
    black_table.add_site(
        name="stack_target_site",
        pos=[-0.02, 0.02, 0.36],
        size=[0.015, 0.015, 0.002],
        rgba=[0.2, 0.9, 0.3, 0.5]
    )

    # 左側: 木目調サイドテーブル
    wood_table = world.add_body(name="wood_side_table", pos=[-0.38, -0.08, 0.62])
    wood_table.add_geom(
        name="wood_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.20, 0.28, 0.015],
        rgba=[0.76, 0.58, 0.38, 1.0]
    )
    wood_table.add_geom(
        name="wood_legs",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -0.31],
        size=[0.18, 0.26, 0.30],
        rgba=[0.45, 0.45, 0.48, 1.0]
    )

    # 水色のパーツトレイ (底面 + 4方の壁)
    bin_body = world.add_body(name="blue_parts_bin", pos=[-0.38, -0.08, 0.64])
    bin_color = [0.65, 0.82, 0.88, 1.0]
    bin_body.add_geom(name="bin_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0], size=[0.13, 0.20, 0.005], rgba=bin_color)
    bin_body.add_geom(name="bin_w_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.122, 0, 0.035], size=[0.008, 0.20, 0.035], rgba=bin_color)
    bin_body.add_geom(name="bin_w_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.122, 0, 0.035], size=[0.008, 0.20, 0.035], rgba=bin_color)
    bin_body.add_geom(name="bin_w_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -0.192, 0.035], size=[0.13, 0.008, 0.035], rgba=bin_color)
    bin_body.add_geom(name="bin_w_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.192, 0.035], size=[0.13, 0.008, 0.035], rgba=bin_color)

    # ==========================================
    # トレイ内部: 40mm四方の赤・青・黄の木製ブロック
    # 40mm = size=[0.02, 0.02, 0.02] (MuJoCoは半サイズ指定)
    # 木製比重約0.6 -> 質量 38g = 0.038kg
    # ==========================================
    block_half_size = [0.02, 0.02, 0.02]
    block_mass = 0.038
    block_friction = [1.5, 0.01, 0.0002]  # スタック時に崩れにくい安定した摩擦

    # 1. 赤色ブロック (Red Block)
    red_b = world.add_body(name="red_block", pos=[-0.38, -0.15, 0.67])
    red_b.add_freejoint(name="red_block_joint")
    red_b.add_geom(
        name="red_block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=block_half_size,
        rgba=[0.88, 0.22, 0.22, 1.0],
        mass=block_mass,
        friction=block_friction
    )

    # 2. 青色ブロック (Blue Block)
    blue_b = world.add_body(name="blue_block", pos=[-0.38, -0.08, 0.67])
    blue_b.add_freejoint(name="blue_block_joint")
    blue_b.add_geom(
        name="blue_block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=block_half_size,
        rgba=[0.20, 0.42, 0.88, 1.0],
        mass=block_mass,
        friction=block_friction
    )

    # 3. 黄色ブロック (Yellow Block)
    yellow_b = world.add_body(name="yellow_block", pos=[-0.38, -0.01, 0.67])
    yellow_b.add_freejoint(name="yellow_block_joint")
    yellow_b.add_geom(
        name="yellow_block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=block_half_size,
        rgba=[0.92, 0.80, 0.18, 1.0],
        mass=block_mass,
        friction=block_friction
    )

    return spec

def create_workcell():
    """コンパイル済みの model, data を取得し写真の初期姿勢を設定"""
    spec = build_ur5e_workcell_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    # 写真のUR5eの初期関節角度 (ラジアン)
    # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    initial_qpos = np.array([-0.45, -1.35, 1.65, -1.85, -1.57, 0.0])
    data.qpos[:6] = initial_qpos
    data.ctrl[:6] = initial_qpos
    mujoco.mj_forward(model, data)

    return model, data

if __name__ == "__main__":
    model, data = create_workcell()
    print("UR5e Block Stacking Workcell compiled successfully!")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of cameras: {model.ncam}")
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  Camera {i}: {cam_name}")
