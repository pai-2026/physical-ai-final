# Step 1: MuJoCo ワークセル環境の構築 (Step1-Environment)

UR5e によるピック＆プレース学習を行うための 3D シミュレータ環境。
環境画像をGemini Argonに入れ、Mujocoの環境を自動生成作成

## 構成概要

- **ロボット**: Universal Robots UR5e
- **対象ワーク**: 3色立方体ブロック（赤・青・黄、一辺 40mm）
- **カメラ（2視点・512×512）**:
  - `overhead_camera`: ワークセル全体を見下ろす俯瞰視点
  - `wrist_camera`: ハンド手元の近接視点
- **実装方式**: `mujoco.MjSpec`（Python）による動的 XML (MJCF) 生成

Sim環境の構築はAIに任せた。
環境の再現を確認しこの内容は手を付けていない