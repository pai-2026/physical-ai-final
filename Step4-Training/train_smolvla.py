"""
UR5e ブロックスタッキング SmolVLA 学習実行スクリプト
Hugging Face LeRobot (v0.6.1) を使用して SmolVLA をファインチューニングする。
"""
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="UR5e SmolVLA Training Script")
    parser.add_argument("--steps", type=int, default=10000, help="総学習ステップ数 (デフォルト: 10,000)")
    parser.add_argument("--batch_size", type=int, default=8, help="バッチサイズ (デフォルト: 8, RTX 4070で約3.2GB VRAM)")
    parser.add_argument("--save_freq", type=int, default=2000, help="チェックポイント保存頻度 (デフォルト: 2,000)")
    parser.add_argument("--log_freq", type=int, default=50, help="ログ出力頻度 (デフォルト: 50)")
    parser.add_argument("--output_dir", type=str, default="outputs/smolvla_stacking_512", help="出力ディレクトリ")
    parser.add_argument("--pretrained_path", type=str, default="lerobot/smolvla_base", help="ベースモデル")
    parser.add_argument("--dataset_root", type=str, default="datasets/ur5e_block_stacking_512", help="データセットディレクトリ")
    parser.add_argument("--repo_id", type=str, default="local/ur5e_block_stacking_512", help="データセットID")
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    if output_path.exists():
        raise FileExistsError(f"保存先が既に存在します。別の --output_dir を指定してください: {output_path}")
    if not (Path(args.dataset_root) / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"データセットが見つかりません: {args.dataset_root}")

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--dataset.repo_id", args.repo_id,
        "--dataset.root", args.dataset_root,
        "--policy.type", "smolvla",
        "--policy.pretrained_path", args.pretrained_path,
        "--policy.push_to_hub", "false",
        "--output_dir", str(output_path),
        "--batch_size", str(args.batch_size),
        "--steps", str(args.steps),
        "--log_freq", str(args.log_freq),
        "--save_checkpoint", "true",
        "--save_freq", str(args.save_freq),
        "--wandb.enable", "false"
    ]

    print("=" * 60)
    print("🚀 SmolVLA 学習開始")
    print(f"  ・ベースモデル   : {args.pretrained_path}")
    print(f"  ・データセット   : {args.dataset_root}")
    print(f"  ・総ステップ数   : {args.steps:,} steps")
    print(f"  ・バッチサイズ   : {args.batch_size}")
    print(f"  ・保存頻度       : 毎 {args.save_freq:,} steps")
    print(f"  ・出力先         : {args.output_dir}")
    print("=" * 60)
    print("実行コマンド:")
    print(" ".join(cmd))
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 学習中にエラーが発生しました (終了コード: {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
