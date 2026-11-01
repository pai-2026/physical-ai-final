import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR.parent / "Step1-Environment"))
sys.path.insert(0, str(CURRENT_DIR.parent / "Step2-Control"))

if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import time

import cv2
import numpy as np

from stacking_env import (CONTROL_FREQUENCY, TABLE_Z, BLOCK_HALF,
                          StackingEnv, sample_episode)

CAMERAS = {"overhead": "overhead_camera", "wrist": "wrist_camera"}


def build_plan(env: StackingEnv, rng):
    """(TCP目標位置, グリッパー指令, 所要時間[s]) の区間リストを作る"""
    s = env.spec
    start = env.tcp_pos()
    pick = env.block_pos(s.pick_color)
    if s.kind == "place":
        place = np.array([s.table_xy[0], s.table_xy[1], TABLE_Z + BLOCK_HALF])
    else:
        place = env.block_pos(s.stacked_blosks[-1]) + np.array([0, 0, 2 * BLOCK_HALF])

    v_free = rng.uniform(0.18, 0.28)      # 空中移動の速さ [m/s]
    v_fine = rng.uniform(0.07, 0.11)      # 降下・上昇の速さ

    def seg(p0, p1, v, t_min=0.6):
        return max(t_min, np.linalg.norm(p1 - p0) / v)

    above_pick = pick + [0, 0, 0.13]
    grasp = pick + [0, 0, 0.015]
    approach_dist = 0.90
    lift = np.array([pick[0], pick[1], approach_dist])
    above_place = np.array([place[0], place[1], approach_dist])
    release = place + [0, 0, 0.018]
    retreat = place + [0, 0, 0.12]

    return [
        (start, above_pick, 0, seg(start, above_pick, v_free)),
        (above_pick, grasp, 0, seg(above_pick, grasp, v_fine)),
        (grasp, grasp, 1, 0.4),  # 閉じて待つ
        (grasp, lift, 1, seg(grasp, lift, v_fine)),
        (lift, above_place, 1, seg(lift, above_place, v_free)),
        (above_place, release, 1, seg(above_place, release, v_fine)),
        (release, release, 0, 0.4),  # 開いて待つ
        (release, retreat, 0, seg(release, retreat, v_fine)),
    ]


def expert_actions(env: StackingEnv, plan):
    """区間リストを CONTROL_FREQUENCY の行動列に変換するジェネレータ (毎ステップ IK)"""
    axes = env.data.qpos[:6].copy()
    for p0, p1, grip, dur in plan:
        n = max(1, int(round(dur * CONTROL_FREQUENCY)))
        for k in range(1, n + 1):
            s = 0.5 * (1 - np.cos(np.pi * k / n))
            target = p0 + s * (np.asarray(p1) - p0)
            axes = env.solve_ik(target, axes)
            yield np.concatenate([axes, [grip]]).astype(np.float32)
    # 最後に少し静止 (ブロックが落ち着くのを待つ)
    for _ in range(6):
        yield np.concatenate([axes, [0.0]]).astype(np.float32)


def run_episode(env: StackingEnv, rng):
    """1エピソードを実行し、フレーム列と成否を返す"""
    spec = sample_episode(rng)
    env.reset(spec, rng)
    plan = build_plan(env, rng)
    frames = []
    for action in expert_actions(env, plan):
        frame = {
            "observation.state": env.get_state(),
            "action": action,
        }
        for key, cam in CAMERAS.items():
            frame[f"observation.images.{key}"] = env.render(cam)
        frames.append(frame)
        env.step(action)
    return spec, frames, env.check_success()


def write_preview(path, frames, text):
    h = frames[0]["observation.images.overhead"].shape[0]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_FREQUENCY, (2 * h, h + 30))
    for f in frames:
        img = np.hstack([f["observation.images.overhead"], f["observation.images.wrist"]])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bar = np.zeros((30, 2 * h, 3), np.uint8)
        cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        vw.write(np.vstack([bar, img]))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=0, help="N エピソード試して videos/preview に動画保存")
    ap.add_argument("--episodes", type=int, default=100, help="保存する成功エピソード数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--root", default="datasets/ur5e_block_stacking_512")
    ap.add_argument("--repo-id", default="local/ur5e_block_stacking_512")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = StackingEnv(image_size=args.image_size)

    try:
        if args.preview:
            os.makedirs("videos/preview", exist_ok=True)
            n_ok = 0
            for i in range(args.preview):
                t0 = time.time()
                spec, frames, ok = run_episode(env, rng)
                n_ok += ok
                path = f"videos/preview/ep{i:02d}_{'ok' if ok else 'NG'}.mp4"
                write_preview(path, frames, f"{spec.instruction}  [{'OK' if ok else 'NG'}]")
                print(f"[{i}] {'OK' if ok else 'NG'}  {len(frames):3d} frames  "
                      f"{time.time() - t0:4.1f}s  '{spec.instruction}' -> {path}")
            print(f"成功 {n_ok}/{args.preview}")
            return

        from lerobot_writer import DatasetWriter   # LeRobot 導入後に使う
        writer = DatasetWriter(args.repo_id, args.root, CONTROL_FREQUENCY, args.image_size, list(CAMERAS))
        saved, tried = 0, 0
        t_start = time.time()
        while saved < args.episodes:
            tried += 1
            spec, frames, ok = run_episode(env, rng)
            if not ok:
                print(f"  skip (失敗) '{spec.instruction}'")
                continue
            writer.add_episode(frames, spec.instruction)
            saved += 1
            print(f"[{saved}/{args.episodes}] {len(frames)} frames  '{spec.instruction}'  "
                  f"(経過 {time.time() - t_start:.0f}s)")
        writer.finish()
        print(f"完了: 成功 {saved} / 試行 {tried}  保存先: {args.root}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
