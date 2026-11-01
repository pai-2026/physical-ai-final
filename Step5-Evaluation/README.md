# Step 5: 自律評価とWeb GUI

学習済みSmolVLAに言語指示を与え、シミュレータ上の動作を確認。

## 構成概要

- **Web GUI**: 指示入力、初期配置のリセット、自律実行・停止。
- **映像**: 俯瞰・手首の2視点と、関節・把持状態を表示。
- **バッチ評価**: 複数エピソードの成功率を集計し、動画を保存。
- **確認結果**: 既存512モデルの1試行は420ステップを完走したが、把持に失敗（成功0/1）。

Web GUIは業務・研究活動で使用しているものを、本課題の動作確認用に流用。
Step4で学習した重みを使用する。GUIは起動後に`http://localhost:5000`を開く。

```bash
.venv/bin/python Step5-Evaluation/web_gui.py --host 127.0.0.1
.venv/bin/python Step5-Evaluation/eval_smolvla.py --num_episodes 5
```
