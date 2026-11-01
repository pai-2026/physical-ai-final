# Step 3: データセット自動収集

Step2の教示軌道を使い、成功したエピソードをLeRobot形式で保存。

## 構成概要

- **多様化**: 初期配置・置き場所・積む順序・言語指示を変える。
- **記録**: 俯瞰・手首の2眼（512×512）、7次元の状態・行動、指示文を30Hzで保存。
- **保存形式**: MP4動画・Parquet・正規化用統計量。
- **選別**: 環境の成功判定を通過した教示のみを保存。

保存には公式LeRobotDataset APIを利用し、同期入力用のラッパーを実装した。
既存の保存先がある場合は、別の`--root`を指定する。

```bash
.venv/bin/python Step3-Dataset/collect_stacking_dataset.py --episodes 300
```
