# UR5eを用いた社内勉強会の環境構築、学習
社内のVLA勉強会（及び自動化案件）で、VLAを用いたピック/プレースにVLAを用いて改善できないかの基礎研究を行う。
そこで使用するテーマをもとに、シミュレータ作成し小型VLAで学習を行い知見を得るする。

実機環境を模したシミュレータの構築から、データ収集、VLAの学習まで行う。
業務内容とかぶるため、提出内容はブロックスタッキングの簡易デモにする。

- 本番は顧客の物品の整列タスクだが、ここでは簡易なブロックに置き換えている。
- モデルも計算機の都合でSmolVLAにしているが、実際はPi0やGR00Tでやる予定。
- ブロックスタッキングのSmolVLAは学習済み。ACTなどの比較学習は未実施。

## 提出内容

- **Step 1: シミュレータ環境の構築 (`Step1-Environment/`)**
  - UR5e + 作業台を MuJoCo 上にモデリング。
  - Gemini4 Argonを使用して環境撮影から半自動作成（細かな調整はこのデモではしていない）
- **Step 2: 学習用の教師データ作成 (`Step2-Control/`)**
  - TCP姿勢を安定させるIKと、把持・搬送・退避の軌道を実装。
- **Step 3: データ収集 (`Step3-Dataset/`)**
  - 初期配置や指示を変え、成功した教示をLeRobot形式で保存。
  - 俯瞰・手首の2眼（512×512）と、関節状態・行動を30Hzで記録。
- **Step 4: 学習 (`Step4-Training/`)**
  - ブロック300エピソードでSmolVLAを学習。
  - ACTの比較用スクリプトとColabノートブックを準備。
- **Step 5: 動作確認 (`Step5-Evaluation/`)**
  - 業務・研究活動で使用しているWeb GUIを流用。
  - 言語指示を入力し、2眼映像でシミュレータ上の動作を確認。

## 自分で工夫した点

- 手先の傾き：IKの回転誤差に3.5倍の重みを設定。
- 把持時の反発：把持瞬間の相対位置を保持し、対象物を手先に追従。
- 搬送時の接触：水平移動の前に垂直リフトを追加。
- データ収集：配置・置き場所・積む順序を変え、成功例を選別して保存。

UR5e本体はMuJoCo Menagerie、保存・学習はLeRobotを利用。把持はシミュレータ上の簡易方式。

## 実行方法

Linux・Python 3.12、CUDA対応GPUを想定。ルートから実行する。初回はモデル取得のためネット接続とGit、ヘッドレス描画にはEGLが必要。

```bash
python3.12 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
export MUJOCO_GL=egl

# データ収集 → 学習 → GUI
.venv/bin/python Step3-Dataset/collect_stacking_dataset.py --episodes 300
.venv/bin/python Step4-Training/train_smolvla.py
.venv/bin/python Step5-Evaluation/web_gui.py --host 127.0.0.1
```

データ・重みは同梱せず、上記手順で生成する。既存の保存先がある場合は別名を指定する。詳しい仕様・操作は各StepのREADMEを参照。

## 動作確認

教示動作、データ保存・再読込、1ステップ学習とモデル保存を確認。自律評価は1試行を完走したが、把持に失敗（成功0/1）。複数試行での性能評価は未実施。


## AI利用

実装・環境構築・資料作成の補助にAIを使用。
