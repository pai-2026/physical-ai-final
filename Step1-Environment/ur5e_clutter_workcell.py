"""
UR5e Clutter Bin-Picking & Sorting Workcell
==========================================
ばら積みピッキング・仕分け整理用 MuJoCo 環境モデル。
- ロボット: UR5e + 万能パラレルグリッパー (0〜86mm 開閉ストローク)
- ピッキングエリア: ばら積みコンテナ (Tote Bin)
- ばら積み対象物:
  1. ペン (Pen / Marker): ペン軸 + キャップ + クリップ
  2. ペットボトル (Plastic Bottle): ボトル本体 + ラベル + ネック + キャップ
  3. ガムテープ (Packing Tape): クラフト紙巻き芯 + テープ外周
  4. ティッシュ箱 (Tissue Box): 直方体パッケージ + 上部取り出し口 & ティッシュペーパー
- 仕分け整理エリア (4つの専用ステーション):
  1. ペントレイ (Pen Organizer Tray)
  2. ペットボトル回収エリア (Bottle Recycling Coaster)
  3. ガムテープ・工具トレイ (Tape Storage Bin)
  4. ティッシュ置き場 (Tissue Station)
- カメラ: 俯瞰カメラ (Overhead) + 手首カメラ (Wrist)
"""

import os
import mujoco
import numpy as np
from robot_descriptions.ur5e_mj_description import MJCF_PATH

OBJECT_NAMES = ["pen", "bottle", "tape", "tissue"]

# 仕分け目標ステーションの座標定義 (メイン作業台上の4箇所: すべてUR5e可達範囲内)
SORTING_STATIONS = {
    "pen_tray": {
        "target_pos": np.array([0.09, 0.01, 0.72]),  # 奥・左
        "name_ja": "ペントレイ",
        "description": "pen organizer tray",
        "accepted_obj": "pen",
    },
    "bottle_zone": {
        "target_pos": np.array([0.21, 0.01, 0.72]),  # 奥・右
        "name_ja": "ペットボトル回収ゾーン",
        "description": "bottle recycling zone",
        "accepted_obj": "bottle",
    },
    "tape_bin": {
        "target_pos": np.array([0.09, -0.12, 0.72]),  # 手前・左
        "name_ja": "ガムテープトレイ",
        "description": "tape storage bin",
        "accepted_obj": "tape",
    },
    "tissue_spot": {
        "target_pos": np.array([0.21, -0.12, 0.72]),  # 手前・右
        "name_ja": "ティッシュ置き場",
        "description": "tissue station",
        "accepted_obj": "tissue",
    },
}

# ばら積みコンテナ中心座標
BIN_CENTER = np.array([-0.38, -0.06, 0.64])


def build_ur5e_clutter_spec():
    """MjSpec を用いてばら積みピッキング・仕分け環境を構築"""
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    spec.modelname = "ur5e_clutter_sorting_workcell"
    spec.visual.global_.offwidth = 1024
    spec.visual.global_.offheight = 1024

    # 1. UR5e ベース位置
    ur_base = spec.body("base")
    if ur_base is not None:
        ur_base.pos = [-0.05, 0.35, 0.70]
        ur_base.quat = [0.7071068, 0, 0, -0.7071068]

    # 2. 手首 (wrist_3_link): 広ストローク万能パラレルグリッパー (0〜96mm) + 手首カメラ
    wrist = spec.body("wrist_3_link")
    if wrist is not None:
        # 手首カメラ (指先とワークをしっかり画角に収める)
        wrist.add_camera(
            name="wrist_camera",
            pos=[0.0, 0.06, 0.08],
            quat=[0.683, 0.683, -0.183, 0.183],
            fovy=64,
        )

        # グリッパー基部
        gripper_base = wrist.add_body(name="gripper_base", pos=[0.0, 0.04, 0.0])
        gripper_base.add_geom(
            name="gripper_housing",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.02, 0.0],
            size=[0.050, 0.025, 0.025],
            rgba=[0.18, 0.18, 0.20, 1.0],
            mass=0.35,
        )

        # 指先可動範囲: 外側 48mm (全開 96mm) から 内側 0mm (全閉)
        # 左指
        finger_l = gripper_base.add_body(name="finger_left", pos=[-0.048, 0.055, 0.0])
        finger_l.add_joint(
            name="finger_left_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[0.0, 0.048],  # 0: 全開(96mm), 0.048: 全閉(0mm)
            damping=5.0,
        )
        finger_l.add_geom(
            name="finger_l_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.025, 0.0],
            size=[0.005, 0.030, 0.015],
            rgba=[0.25, 0.25, 0.28, 1.0],
            mass=0.04,
            friction=[2.5, 0.01, 0.0002],
        )
        # 高摩擦ラバーパッド
        finger_l.add_geom(
            name="finger_l_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.004, 0.035, 0.0],
            size=[0.002, 0.020, 0.014],
            rgba=[0.90, 0.60, 0.10, 1.0],
            friction=[3.0, 0.02, 0.0005],
        )

        # 右指
        finger_r = gripper_base.add_body(name="finger_right", pos=[0.048, 0.055, 0.0])
        finger_r.add_joint(
            name="finger_right_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[-1, 0, 0],
            range=[0.0, 0.048],
            damping=5.0,
        )
        finger_r.add_geom(
            name="finger_r_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.025, 0.0],
            size=[0.005, 0.030, 0.015],
            rgba=[0.25, 0.25, 0.28, 1.0],
            mass=0.04,
            friction=[2.5, 0.01, 0.0002],
        )
        # 高摩擦ラバーパッド
        finger_r.add_geom(
            name="finger_r_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.004, 0.035, 0.0],
            size=[0.002, 0.020, 0.014],
            rgba=[0.90, 0.60, 0.10, 1.0],
            friction=[3.0, 0.02, 0.0005],
        )

        # TCP サイト
        wrist.add_site(
            name="tcp_site",
            pos=[0.0, 0.135, 0.0],
            size=[0.010, 0.010, 0.010],
            rgba=[1.0, 0.1, 0.1, 0.8],
        )

    # グリッパーアクチュエータ
    act_l = spec.add_actuator(name="gripper_left", trntype=mujoco.mjtTrn.mjTRN_JOINT, target="finger_left_joint")
    act_l.ctrlrange = [0.0, 0.048]
    act_l.set_to_position(kp=500.0, kv=25.0)

    act_r = spec.add_actuator(name="gripper_right", trntype=mujoco.mjtTrn.mjTRN_JOINT, target="finger_right_joint")
    act_r.ctrlrange = [0.0, 0.048]
    act_r.set_to_position(kp=500.0, kv=25.0)

    world = spec.worldbody

    # 床面
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.5, 2.5, 0.05],
        rgba=[0.38, 0.38, 0.40, 1.0],
    )

    # 天井リングライト
    world.add_light(
        name="ring_light",
        pos=[0.0, 0.0, 1.60],
        dir=[0, 0, -1],
        diffuse=[0.70, 0.70, 0.70],
        specular=[0.10, 0.10, 0.10],
        ambient=[0.35, 0.35, 0.35],
    )
    # 環境補助ライト
    world.add_light(
        name="ambient_fill",
        pos=[-0.5, -0.6, 1.8],
        dir=[0.3, 0.3, -0.9],
        diffuse=[0.35, 0.35, 0.35],
        specular=[0.05, 0.05, 0.05],
    )

    # 俯瞰カメラ (Overhead Camera): ばら積みコンテナと仕分け作業台の両方を明瞭に捉える
    world.add_camera(
        name="overhead_camera",
        pos=[0.20, -0.80, 1.35],
        xyaxes=[0.8944, 0.4472, 0.0, -0.2857, 0.5715, 0.7693],
        fovy=60,
    )

    # ロボット架台
    mount = world.add_body(name="alu_robot_stand", pos=[-0.05, 0.35, 0.35])
    mount.add_geom(
        name="stand_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.14, 0.14, 0.35],
        rgba=[0.75, 0.75, 0.78, 1.0],
    )

    # ==========================================
    # 右側: 仕分け整理用メイン作業台 (アルミフレーム + マット天板)
    # ==========================================
    worktable = world.add_body(name="sorting_worktable", pos=[0.20, -0.06, 0.35])
    worktable.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.34],
        size=[0.34, 0.34, 0.015],
        rgba=[0.14, 0.14, 0.16, 1.0],
    )
    worktable.add_geom(
        name="table_frame",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0],
        size=[0.32, 0.32, 0.34],
        rgba=[0.70, 0.71, 0.73, 1.0],
    )

    # ----------------------------------------------------
    # 仕分けステーション 1: ペントレイ (奥・左)
    # ----------------------------------------------------
    pen_tray = world.add_body(name="pen_tray_body", pos=[0.09, 0.01, 0.71])
    # トレイ底面
    pen_tray.add_geom(
        name="pt_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0],
        size=[0.045, 0.085, 0.004],
        rgba=[0.20, 0.25, 0.38, 1.0],  # ネイビーブルー
    )
    # トレイ外枠
    pen_tray.add_geom(name="pt_w_l", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.041, 0, 0.010], size=[0.004, 0.085, 0.010], rgba=[0.15, 0.20, 0.32, 1.0])
    pen_tray.add_geom(name="pt_w_r", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.041, 0, 0.010], size=[0.004, 0.085, 0.010], rgba=[0.15, 0.20, 0.32, 1.0])
    pen_tray.add_geom(name="pt_w_f", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -0.081, 0.010], size=[0.045, 0.004, 0.010], rgba=[0.15, 0.20, 0.32, 1.0])
    pen_tray.add_geom(name="pt_w_b", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.081, 0.010], size=[0.045, 0.004, 0.010], rgba=[0.15, 0.20, 0.32, 1.0])
    pen_tray.add_site(name="pen_tray_site", pos=[0, 0, 0.010], size=[0.015, 0.015, 0.002], rgba=[0.3, 0.5, 0.9, 0.6])

    # ----------------------------------------------------
    # 仕分けステーション 2: ペットボトル回収エリア (奥・右)
    # ----------------------------------------------------
    bottle_zone = world.add_body(name="bottle_zone_body", pos=[0.21, 0.01, 0.707])
    # 丸型リサイクルコースター
    bottle_zone.add_geom(
        name="bz_coaster",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0],
        size=[0.050, 0.003],
        rgba=[0.18, 0.55, 0.32, 1.0],  # リサイクルグリーン
    )
    # コースター縁取りリング
    bottle_zone.add_geom(
        name="bz_ring",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.004],
        size=[0.052, 0.002],
        rgba=[0.30, 0.75, 0.45, 0.7],
    )
    bottle_zone.add_site(name="bottle_zone_site", pos=[0, 0, 0.008], size=[0.015, 0.015, 0.002], rgba=[0.2, 0.8, 0.3, 0.6])

    # ----------------------------------------------------
    # 仕分けステーション 3: ガムテープトレイ (手前・左)
    # ----------------------------------------------------
    tape_bin = world.add_body(name="tape_bin_body", pos=[0.09, -0.12, 0.71])
    # 正方形トレイ底面
    tape_bin.add_geom(
        name="tb_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0],
        size=[0.058, 0.058, 0.004],
        rgba=[0.75, 0.45, 0.18, 1.0],  # アンバー/オレンジ調
    )
    # トレイ外枠
    tape_bin.add_geom(name="tb_w_l", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.054, 0, 0.010], size=[0.004, 0.058, 0.010], rgba=[0.60, 0.35, 0.12, 1.0])
    tape_bin.add_geom(name="tb_w_r", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.054, 0, 0.010], size=[0.004, 0.058, 0.010], rgba=[0.60, 0.35, 0.12, 1.0])
    tape_bin.add_geom(name="tb_w_f", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -0.054, 0.010], size=[0.058, 0.004, 0.010], rgba=[0.60, 0.35, 0.12, 1.0])
    tape_bin.add_geom(name="tb_w_b", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.054, 0.010], size=[0.058, 0.004, 0.010], rgba=[0.60, 0.35, 0.12, 1.0])
    tape_bin.add_site(name="tape_bin_site", pos=[0, 0, 0.010], size=[0.015, 0.015, 0.002], rgba=[0.9, 0.6, 0.2, 0.6])

    # ----------------------------------------------------
    # 仕分けステーション 4: ティッシュ置き場 (手前・右)
    # ----------------------------------------------------
    tissue_spot = world.add_body(name="tissue_spot_body", pos=[0.21, -0.12, 0.707])
    # 卓上マット/枠
    tissue_spot.add_geom(
        name="ts_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0],
        size=[0.065, 0.090, 0.003],
        rgba=[0.35, 0.55, 0.58, 1.0],  # ターコイズ/ペールティール
    )
    # 枠線ガイド
    tissue_spot.add_geom(
        name="ts_border",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.003],
        size=[0.062, 0.086, 0.001],
        rgba=[0.60, 0.82, 0.85, 0.9],
    )
    tissue_spot.add_site(name="tissue_spot_site", pos=[0, 0, 0.008], size=[0.015, 0.015, 0.002], rgba=[0.4, 0.8, 0.8, 0.6])

    # ==========================================
    # 左側: サイドテーブル + ばら積みトートコンテナ (Pick Bin)
    # ==========================================
    side_table = world.add_body(name="clutter_side_table", pos=[-0.38, -0.06, 0.62])
    side_table.add_geom(
        name="st_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.22, 0.30, 0.015],
        rgba=[0.76, 0.58, 0.38, 1.0],
    )
    side_table.add_geom(
        name="st_legs",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, -0.31],
        size=[0.20, 0.28, 0.30],
        rgba=[0.45, 0.45, 0.48, 1.0],
    )

    # ばら積み用大型トートコンテナ (外寸: 300mm x 440mm, 深さ 70mm)
    tote_body = world.add_body(name="clutter_tote_bin", pos=BIN_CENTER)
    tote_color = [0.82, 0.84, 0.88, 1.0]  # ライトグレーの工業用トート
    tote_body.add_geom(name="tote_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, 0], size=[0.14, 0.21, 0.006], rgba=tote_color)
    tote_body.add_geom(name="tote_wall_l", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.132, 0, 0.040], size=[0.008, 0.21, 0.040], rgba=tote_color)
    tote_body.add_geom(name="tote_wall_r", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.132, 0, 0.040], size=[0.008, 0.21, 0.040], rgba=tote_color)
    tote_body.add_geom(name="tote_wall_f", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -0.202, 0.040], size=[0.14, 0.008, 0.040], rgba=tote_color)
    tote_body.add_geom(name="tote_wall_b", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.202, 0.040], size=[0.14, 0.008, 0.040], rgba=tote_color)

    # =========================================================================
    # ばら積み対象物 (4種): ペン, ペットボトル, ガムテープ, ティッシュ箱
    # =========================================================================

    # ----------------------------------------------------
    # 1. ペン (Pen / Marker): 直径14mm, 長さ130mm, 質量22g (横倒しで配置)
    # ----------------------------------------------------
    pen_body = world.add_body(name="pen_obj", pos=[-0.38, -0.16, 0.66], quat=[0.7071068, 0.7071068, 0, 0])
    pen_body.add_freejoint(name="pen_joint")
    # ペン軸 (黒色)
    pen_body.add_geom(
        name="pen_shaft",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, -0.015, 0],
        size=[0.007, 0.050],  # 半径7mm, 半長50mm (長さ100mm)
        rgba=[0.12, 0.12, 0.14, 1.0],
        mass=0.016,
        friction=[2.5, 0.01, 0.0002],
    )
    # ペンキャップ (鮮やかな青色)
    pen_body.add_geom(
        name="pen_cap",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0.050, 0],
        size=[0.008, 0.020],  # 半径8mm, 長さ40mm
        rgba=[0.15, 0.45, 0.90, 1.0],
        mass=0.006,
        friction=[2.5, 0.01, 0.0002],
    )
    # クリップ部
    pen_body.add_geom(
        name="pen_clip",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.009, 0.050, 0],
        size=[0.002, 0.015, 0.002],
        rgba=[0.85, 0.85, 0.90, 1.0],
        mass=0.001,
    )

    # ----------------------------------------------------
    # 2. ペットボトル (Plastic Bottle): 直径46mm, 全高130mm, 質量35g
    # ----------------------------------------------------
    bottle_body = world.add_body(name="bottle_obj", pos=[-0.38, -0.06, 0.68])
    bottle_body.add_freejoint(name="bottle_joint")
    # 下部本体 (透明感のある薄い水色)
    bottle_body.add_geom(
        name="bottle_lower",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, -0.025],
        size=[0.023, 0.040],  # 直径46mm, 長さ80mm
        rgba=[0.80, 0.92, 0.98, 0.85],
        mass=0.022,
        friction=[2.0, 0.01, 0.0002],
    )
    # ラベル帯 (視認性の高い白・緑ラベル)
    bottle_body.add_geom(
        name="bottle_label",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, -0.010],
        size=[0.0233, 0.020],
        rgba=[0.25, 0.68, 0.38, 1.0],
        mass=0.005,
    )
    # 上部テーパー肩口
    bottle_body.add_geom(
        name="bottle_shoulder",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.025],
        size=[0.016, 0.010],
        rgba=[0.82, 0.93, 0.98, 0.85],
        mass=0.006,
    )
    # ネック & キャップ (赤色キャップ)
    bottle_body.add_geom(
        name="bottle_cap",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.045],
        size=[0.013, 0.010],
        rgba=[0.88, 0.20, 0.20, 1.0],
        mass=0.003,
        friction=[2.5, 0.01, 0.0002],
    )

    # ----------------------------------------------------
    # 3. ガムテープ (Packing Tape Roll): 外径64mm, 内径36mm, 厚み24mm, 質量45g
    # ----------------------------------------------------
    tape_body = world.add_body(name="tape_obj", pos=[-0.38, 0.04, 0.67])
    tape_body.add_freejoint(name="tape_joint")
    # クラフトテープ本体 (茶色ロール)
    tape_body.add_geom(
        name="tape_outer",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0],
        size=[0.032, 0.012],  # 外径64mm, 厚さ24mm
        rgba=[0.72, 0.50, 0.30, 1.0],  # クラフト紙ブラウン
        mass=0.038,
        friction=[2.2, 0.01, 0.0002],
    )
    # 内側紙管芯 (薄いベージュ)
    tape_body.add_geom(
        name="tape_core",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0],
        size=[0.018, 0.0121],
        rgba=[0.88, 0.78, 0.65, 1.0],
        mass=0.007,
    )

    # ----------------------------------------------------
    # 4. ティッシュ箱 (Tissue Box): 幅56mm, 奥行き90mm, 高さ36mm, 質量35g
    # ----------------------------------------------------
    tissue_body = world.add_body(name="tissue_obj", pos=[-0.38, 0.14, 0.68])
    tissue_body.add_freejoint(name="tissue_joint")
    # 箱本体 (淡いパステルミントグリーン)
    tissue_body.add_geom(
        name="tissue_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0],
        size=[0.028, 0.045, 0.018],  # 56mm x 90mm x 36mm
        rgba=[0.60, 0.82, 0.78, 1.0],
        mass=0.033,
        friction=[2.0, 0.01, 0.0002],
    )
    # 上部取り出し口の楕円枠
    tissue_body.add_geom(
        name="tissue_slit",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0, 0, 0.0182],
        size=[0.012, 0.0005],
        rgba=[0.45, 0.68, 0.65, 1.0],
    )
    # 引き出された白いティッシュペーパー
    tissue_body.add_geom(
        name="tissue_paper",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.023],
        size=[0.014, 0.022, 0.004],
        rgba=[0.98, 0.98, 0.98, 1.0],
        mass=0.002,
    )

    return spec


def create_clutter_workcell():
    """コンパイル済みの model, data を取得し初期姿勢を設定"""
    spec = build_ur5e_clutter_spec()
    model = spec.compile()
    data = mujoco.MjData(model)

    initial_qpos = np.array([-0.45, -1.35, 1.65, -1.85, -1.57, 0.0])
    data.qpos[:6] = initial_qpos
    data.ctrl[:6] = initial_qpos
    mujoco.mj_forward(model, data)

    return model, data


if __name__ == "__main__":
    model, data = create_clutter_workcell()
    print("✅ UR5e Clutter Sorting Workcell compiled successfully!")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of cameras: {model.ncam}")
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  Camera {i}: {cam_name}")
