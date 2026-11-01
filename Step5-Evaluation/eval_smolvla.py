"""
UR5e ブロックスタッキング SmolVLA 自律推論・評価スクリプト
学習済みチェックポイントを読み込み、MuJoCo環境で自律推論ロールアウトを実行し、成否判定・動画記録を行う。
"""
import os
if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.startswith("Step") else CURRENT_DIR
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Step1-Environment"))
sys.path.insert(0, str(PROJECT_ROOT / "Step2-Control"))

import cv2
import mujoco
import numpy as np
import torch

from stacking_env import StackingEnv, sample_episode
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


def parse_args():
    parser = argparse.ArgumentParser(description="UR5e SmolVLA Closed-Loop Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/smolvla_stacking_512/checkpoints/last/pretrained_model",
        help="読み込むチェックポイントディレクトリ",
    )
    parser.add_argument("--num_episodes", type=int, default=10, help="評価エピソード数")
    parser.add_argument("--max_steps", type=int, default=420, help="1エピソードの最大ステップ数 (30FPSで約14秒)")
    parser.add_argument("--save_video", action=argparse.BooleanOptionalAction, default=True, help="評価ロールアウト動画を保存")
    parser.add_argument("--video_dir", type=str, default="videos/eval_512", help="動画保存先ディレクトリ")
    parser.add_argument("--seed", type=int, default=2026, help="評価用乱数シード")
    parser.add_argument("--device", type=str, default="cuda", help="推論デバイス")
    return parser.parse_args()


def load_policy_and_processors(ckpt_path: Path, device: str):
    if not ckpt_path.exists():
        # last シンボリックリンクが切れている場合のフォールバック検索
        parent = ckpt_path.parent.parent if ckpt_path.parent.name == "last" else ckpt_path.parent
        checkpoints = sorted(parent.glob("[0-9]*"))
        if checkpoints:
            ckpt_path = checkpoints[-1] / "pretrained_model"
            print(f"指定チェックポイントが見つからないため最新の {ckpt_path} を使用します")
        else:
            raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")

    print(f"📥 チェックポイント読み込み中: {ckpt_path}")
    policy_cfg = SmolVLAConfig.from_pretrained(str(ckpt_path))
    policy_cfg.pretrained_path = str(ckpt_path)
    policy_cfg.device = device

    policy = SmolVLAPolicy.from_pretrained(str(ckpt_path))
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_path),
    )
    print("✅ ポリシーおよび前処理・後処理パイプラインの準備完了")
    return policy, preprocessor, postprocessor


def run_evaluation():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    video_dir = Path(args.video_dir)
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    policy, preprocessor, postprocessor = load_policy_and_processors(ckpt_path, args.device)
    img_feat = policy.config.input_features.get("observation.images.overhead")
    img_size = img_feat.shape[-1] if img_feat is not None else 512
    print(f"📷 入力カメラ解像度: {img_size}x{img_size}")
    env = StackingEnv(image_size=img_size)

    success_list = []
    print("\n" + "=" * 65)
    print(f"🎯 自律推論・評価開始 (全 {args.num_episodes} エピソード)")
    print("=" * 65)

    for ep_idx in range(args.num_episodes):
        rng = np.random.default_rng(args.seed + ep_idx * 77)
        spec = sample_episode(rng)
        state = env.reset(spec, rng)

        policy.reset()
        if hasattr(preprocessor, "reset"):
            preprocessor.reset()
        if hasattr(postprocessor, "reset"):
            postprocessor.reset()

        frames = []
        start_time = time.time()
        print(f"\n[Episode {ep_idx + 1}/{args.num_episodes}]")
        print(f"  指示文: \"{spec.instruction}\"")
        print(f"  種別  : {spec.kind} ({spec.pick_color} block)")

        was_picked = False
        min_pick_dist = 1e9
        target_xy = spec.table_xy if spec.kind == "place" else env.block_pos(spec.stacked_blosks[-1])[:2]

        for step in range(args.max_steps):
            img_oh_rgb = env.render("overhead_camera")
            img_wr_rgb = env.render("wrist_camera")

            if args.save_video:
                img_oh_bgr = cv2.cvtColor(img_oh_rgb, cv2.COLOR_RGB2BGR)
                img_wr_bgr = cv2.cvtColor(img_wr_rgb, cv2.COLOR_RGB2BGR)
                stacked = np.hstack([img_oh_bgr, img_wr_bgr])
                header = np.zeros((45, stacked.shape[1], 3), dtype=np.uint8)
                cv2.putText(
                    header,
                    f"Ep {ep_idx+1}: {spec.instruction}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    header,
                    f"Step: {step}/{args.max_steps} | Left: Overhead, Right: Wrist",
                    (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                frames.append(np.vstack([header, stacked]))

            # 画像は (3, H, W) float32 in [0, 1]
            raw_obs = {
                "observation.state": torch.from_numpy(state).float(),
                "observation.images.overhead": torch.from_numpy(img_oh_rgb).permute(2, 0, 1).float() / 255.0,
                "observation.images.wrist": torch.from_numpy(img_wr_rgb).permute(2, 0, 1).float() / 255.0,
                "task": spec.instruction,
            }

            obs_proc = preprocessor(raw_obs)

            with torch.no_grad():
                action_tensor = policy.select_action(obs_proc)

            # 後処理適用 (非正規化)
            action_np = postprocessor(action_tensor).squeeze(0).cpu().numpy()

            state = env.step(action_np)
            if env.attached == spec.pick_color:
                was_picked = True

            tcp = env.tcp_pos()
            b_pos = env.block_pos(spec.pick_color)
            dist_p = np.linalg.norm(tcp[:2] - b_pos[:2])
            if dist_p < min_pick_dist:
                min_pick_dist = dist_p

        success = env.check_success()
        success_list.append(success)
        elapsed = time.time() - start_time
        status_str = "✅ SUCCESS" if success else "❌ FAIL"
        final_block_xy = env.block_pos(spec.pick_color)[:2]
        dist_target = np.linalg.norm(final_block_xy - target_xy)
        if success:
            detail_str = " (把持成功 -> 目標地点へ完璧に配置/スタック)"
        elif was_picked:
            detail_str = f" (把持成功 -> 目標地点へ運搬, 目標位置誤差: {dist_target*100:.1f}cm)"
        else:
            detail_str = f" (トレイ内で空振り, 最接近距離: {min_pick_dist*100:.1f}cm)"
        print(f"  結果  : {status_str}{detail_str} (所要時間: {elapsed:.1f}秒)")

        if args.save_video and frames:
            vid_path = video_dir / f"eval_ep{ep_idx + 1:02d}_{'success' if success else 'fail'}.mp4"
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(vid_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30.0,
                (w, h),
            )
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"  動画  : {vid_path}")

    env.close()

    total = len(success_list)
    succ_count = sum(success_list)
    rate = succ_count / total * 100
    print("\n" + "=" * 65)
    print("📊 評価完了 サマリー")
    print(f"  総エピソード数 : {total}")
    print(f"  成功数         : {succ_count} / {total}")
    print(f"  成功率 (SR)    : {rate:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation()
