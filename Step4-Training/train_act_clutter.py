"""
UR5e Clutter Sorting ACT (Action Chunking with Transformers) Training Script
===========================================================================
ばら積みピッキング・仕分け整理データセットを用いて ACT ポリシーを学習するスクリプト。
Google Colab (RTX Pro 6000) やローカルPCで動作します。

使用例:
  python train_act_clutter.py --dataset_root ./datasets/ur5e_clutter_sorting_512 --batch_size 16 --steps 20000
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="UR5e Clutter Sorting ACT Training Script")
    parser.add_argument("--steps", type=int, default=20000, help="総学習ステップ数 (ACT推奨: 20,000〜30,000)")
    parser.add_argument("--batch_size", type=int, default=16, help="バッチサイズ (RTX 6000なら 16〜32 推奨)")
    parser.add_argument("--save_freq", type=int, default=5000, help="チェックポイント保存頻度")
    parser.add_argument("--log_freq", type=int, default=50, help="コンソールログ頻度")
    parser.add_argument("--output_dir", type=str, default="outputs/act_clutter_512", help="モデル保存先")
    parser.add_argument("--dataset_root", type=str, default="datasets/ur5e_clutter_sorting_512", help="データセットディレクトリ")
    parser.add_argument("--repo_id", type=str, default="local/ur5e_clutter_sorting_512", help="データセット識別子")
    parser.add_argument("--vision_backbone", type=str, default="resnet18", help="ビジョンバックボーン (resnet18 / resnet34 / resnet50)")
    parser.add_argument("--chunk_size", type=int, default=100, help="アクションチャンクサイズ (デフォルト: 100)")
    parser.add_argument("--wandb", action="store_true", help="Weights & Biases ログを有効化")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_root)
    info_path = dataset_path / "meta" / "info.json"

    if not info_path.is_file():
        print(f"❌ エラー: データセットが見つかりません: {info_path}", file=sys.stderr)
        print("  --dataset_root 引数で正しいパスを指定してください。", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_dir)
    if output_path.exists():
        raise FileExistsError(f"保存先が既に存在します。別の --output_dir を指定してください: {output_path}")

    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--dataset.repo_id", args.repo_id,
        "--dataset.root", str(dataset_path),
        "--policy.type", "act",
        "--policy.vision_backbone", args.vision_backbone,
        "--policy.chunk_size", str(args.chunk_size),
        "--policy.n_action_steps", str(args.chunk_size),
        "--policy.push_to_hub", "false",
        "--output_dir", str(output_path),
        "--batch_size", str(args.batch_size),
        "--steps", str(args.steps),
        "--log_freq", str(args.log_freq),
        "--save_checkpoint", "true",
        "--save_freq", str(args.save_freq),
        "--wandb.enable", "true" if args.wandb else "false",
    ]

    print("=" * 65)
    print("🤖 UR5e ばら積み仕分け ACT (Action Chunking with Transformers) 学習開始")
    print(f"  ・ビジョンバックボーン: {args.vision_backbone}")
    print(f"  ・アクションチャンク   : {args.chunk_size} steps")
    print(f"  ・データセット         : {dataset_path} (512x512 Dual-View)")
    print(f"  ・総ステップ数         : {args.steps:,} steps")
    print(f"  ・バッチサイズ         : {args.batch_size}")
    print(f"  ・保存頻度             : 毎 {args.save_freq:,} steps")
    print(f"  ・出力ディレクトリ     : {output_path}")
    print("=" * 65)
    print("実行コマンド:")
    print(" ".join(cmd))
    print("=" * 65 + "\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 65)
        print("🎉 ACT 学習が正常に完了しました！")
        print(f"  ・学習済みモデル保存場所: {output_path / 'checkpoints' / 'last' / 'pretrained_model'}")
        print("=" * 65)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 学習中にエラーが発生しました (終了コード: {e.returncode})", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
