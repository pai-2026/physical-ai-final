"""
UR5e SmolVLA 自律制御・実機検証 Web GUI アプリケーション
Flask + MuJoCo EGL + SmolVLA (LeRobot) によるリアルタイム制御コンソール
"""
import os
import sys
import time
import re
import queue
import threading
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name.startswith("Step") else CURRENT_DIR
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "Step1-Environment"))
sys.path.insert(0, str(PROJECT_ROOT / "Step2-Control"))

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, Response, request, jsonify

from stacking_env import (
    StackingEnv,
    EpisodeSpec,
    sample_tray_positions,
    COLORS,
    TABLE_REGIONS,
    TARGET_RANDOM_OFFSET,
    HOME_TCP,
    AXES_SEED,
    TABLE_Z,
    BLOCK_HALF,
)
from eval_smolvla import load_policy_and_processors

app = Flask(__name__, template_folder=str(CURRENT_DIR / "templates"))

DEFAULT_CHECKPOINT = "outputs/smolvla_stacking_512/checkpoints/last/pretrained_model"


def load_task_catalog():
    """Step3で生成したデータセットから指示文を読み込む"""
    parquet_path = Path("datasets/ur5e_block_stacking_512/meta/tasks.parquet")
    if not parquet_path.exists():
        return []
    df = pd.read_parquet(parquet_path)
    tasks = df.index.tolist()
    catalog = []
    for t in tasks:
        t_lower = t.lower()
        if "stack" in t_lower or "on top of" in t_lower or "onto" in t_lower:
            cat = "stack"
        elif "between the two l markers" in t_lower:
            cat = "place_center"
        elif "left side" in t_lower:
            cat = "place_left"
        elif "right side" in t_lower:
            cat = "place_right"
        elif "front" in t_lower:
            cat = "place_front"
        elif "back" in t_lower:
            cat = "place_back"
        else:
            cat = "other"
        catalog.append({"text": t, "category": cat})
    return catalog


TASK_CATALOG = load_task_catalog()


def parse_instruction_to_spec(instruction: str, base_option: str = "auto", rng: np.random.Generator = None) -> EpisodeSpec:
    """任意の指示文からMuJoCoのEpisodeSpecを自動構成する"""
    if rng is None:
        rng = np.random.default_rng()

    inst = instruction.lower().strip()

    stack_match = re.search(
        r"(stack|put|place)\s+(?:the\s+)?(red|blue|yellow)\s+block\s+(?:on\s+top\s+of|onto|on)\s+(?:the\s+)?(red|blue|yellow)\s+block",
        inst,
    )
    if stack_match:
        pick_color = stack_match.group(2)
        target_color = stack_match.group(3)
        other_colors = [c for c in COLORS if c not in (pick_color, target_color)]
        other_color = other_colors[0] if other_colors else "yellow"

        if base_option == "1":
            n_base = 1
        elif base_option == "2":
            n_base = 2
        else:
            n_base = 1 if rng.random() < 0.6 else 2

        if n_base == 1:
            base = [target_color]
            rest = [pick_color, other_color]
        else:
            base = [other_color, target_color]
            rest = [pick_color]

        region = rng.choice(list(TABLE_REGIONS.keys()))
        center, _ = TABLE_REGIONS[region]
        table_xy = center + rng.uniform(-TARGET_RANDOM_OFFSET, TARGET_RANDOM_OFFSET, size=2)
        tray_xy = sample_tray_positions(rest, rng)

        return EpisodeSpec(
            kind="stack",
            pick_color=pick_color,
            tray_xy=tray_xy,
            stacked_blosks=base,
            table_xy=table_xy,
            region=region,
            instruction=instruction,
        )

    colors_found = [c for c in COLORS if c in inst]
    pick_color = colors_found[0] if colors_found else "red"

    region = "center"
    for r_key, (_, phrase) in TABLE_REGIONS.items():
        if phrase in inst or r_key in inst:
            region = r_key
            break
    if "markers" in inst or "middle" in inst:
        region = "center"
    elif "left" in inst:
        region = "left"
    elif "right" in inst:
        region = "right"
    elif "front" in inst:
        region = "front"
    elif "back" in inst:
        region = "back"

    center, _ = TABLE_REGIONS[region]
    table_xy = center + rng.uniform(-TARGET_RANDOM_OFFSET, TARGET_RANDOM_OFFSET, size=2)
    tray_xy = sample_tray_positions(COLORS, rng)

    return EpisodeSpec(
        kind="place",
        pick_color=pick_color,
        tray_xy=tray_xy,
        stacked_blosks=[],
        table_xy=table_xy,
        region=region,
        instruction=instruction,
    )


class SimulationWorker(threading.Thread):
    """MuJoCoシミュレータ、SmolVLA推論、映像合成を管理する専用ワーカースレッド"""

    def __init__(self, ckpt_path=DEFAULT_CHECKPOINT, device="cuda"):
        super().__init__(daemon=True)
        self.device = device
        self.ckpt_path = Path(ckpt_path)
        self.cmd_queue = queue.Queue()
        self.lock = threading.Lock()

        self.env = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.current_spec = None
        self.current_state = None
        self.is_running = False
        self.stop_requested = False
        self.step_count = 0
        self.max_steps = 420
        self.speed = 1.0  # 1.0 = 30 FPS, 2.0 = 60 FPS, 0.0 = unlimited
        self.success = False
        self.was_picked = False
        self.start_time = 0.0
        self.latest_frame = None
        self.log_counter = 0

        self.status = {
            "state": "idle",
            "step": 0,
            "max_steps": self.max_steps,
            "tcp": [0.0, 0.0, 0.0],
            "gripper": 0.0,
            "attached": None,
            "dist_target": 0.0,
            "success": False,
            "elapsed_time": 0.0,
            "recent_logs": [],
        }

    def add_log(self, msg: str):
        with self.lock:
            self.log_counter += 1
            log_item = {"id": f"log_{self.log_counter}_{time.time()}", "msg": msg}
            self.status["recent_logs"] = [log_item]

    def run(self):
        print(f"🚀 SimulationWorker 初期化中 (Device: {self.device})...")
        self.policy, self.preprocessor, self.postprocessor = load_policy_and_processors(self.ckpt_path, self.device)
        img_feat = self.policy.config.input_features.get("observation.images.overhead")
        img_size = img_feat.shape[-1] if img_feat is not None else 512
        self.env = StackingEnv(image_size=img_size)

        default_inst = "stack the yellow block on the blue block"
        rng = np.random.default_rng(2026)
        self.current_spec = parse_instruction_to_spec(default_inst, "1", rng)
        self.current_state = self.env.reset(self.current_spec, rng)
        self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

        self._render_and_update_frame("IDLE (準備完了)")
        self.add_log(f"環境とモデルが起動しました。タスク: \"{default_inst}\"")
        print("✅ SimulationWorker 起動完了")

        while True:
            try:
                while not self.cmd_queue.empty():
                    cmd = self.cmd_queue.get_nowait()
                    self._handle_command(cmd)
            except Exception as e:
                print(f"Command error: {e}")

            if self.is_running:
                t_step_start = time.time()
                self._step_rollout()

                if self.speed > 0:
                    target_dt = (1.0 / 30.0) / self.speed
                    elapsed = time.time() - t_step_start
                    if target_dt > elapsed:
                        time.sleep(target_dt - elapsed)
            else:
                time.sleep(0.04)

    def _handle_command(self, cmd):
        c_type = cmd.get("type")
        if c_type == "reset":
            inst = cmd.get("instruction", "stack the yellow block on the blue block")
            base_opt = cmd.get("base_option", "auto")
            seed = int(time.time() * 1000) % 100000
            rng = np.random.default_rng(seed)

            self.current_spec = parse_instruction_to_spec(inst, base_opt, rng)
            self.current_state = self.env.reset(self.current_spec, rng)

            self.policy.reset()
            if hasattr(self.preprocessor, "reset"):
                self.preprocessor.reset()
            if hasattr(self.postprocessor, "reset"):
                self.postprocessor.reset()

            self.is_running = False
            self.stop_requested = False
            self.step_count = 0
            self.success = False
            self.was_picked = False
            self.start_time = 0.0

            self._update_telemetry("idle")
            self._render_and_update_frame("IDLE (配置完了)")
            self.add_log(f"シーン再配置完了: \"{inst}\" ({self.current_spec.kind})")

        elif c_type == "run":
            if not self.is_running:
                inst = cmd.get("instruction")
                if not inst and self.current_spec:
                    inst = self.current_spec.instruction
                elif not inst:
                    inst = "stack the yellow block on the blue block"

                # 前回の実行が終了しているか指示文が変わっている場合は自動でシーンを再配置
                need_scene_reset = (
                    self.step_count >= self.max_steps
                    or self.success
                    or self.current_spec is None
                    or (inst != self.current_spec.instruction)
                )

                if need_scene_reset:
                    seed = int(time.time() * 1000) % 100000
                    rng = np.random.default_rng(seed)
                    self.current_spec = parse_instruction_to_spec(inst, "auto", rng)
                    self.current_state = self.env.reset(self.current_spec, rng)
                else:
                    self.current_spec.instruction = inst

                self.policy.reset()
                if hasattr(self.preprocessor, "reset"):
                    self.preprocessor.reset()
                if hasattr(self.postprocessor, "reset"):
                    self.postprocessor.reset()

                self.is_running = True
                self.stop_requested = False
                self.step_count = 0
                self.success = False
                self.was_picked = False
                self.start_time = time.time()
                self.add_log(f"自律推論実行を開始しました: \"{inst}\" (最大 {self.max_steps} ステップ)")

        elif c_type == "stop":
            self.is_running = False
            self.stop_requested = True
            self._update_telemetry("stopped")
            self._render_and_update_frame("STOPPED (一時停止)")
            self.add_log("自律実行を停止しました。")

        elif c_type == "home":
            self.is_running = False
            self._smooth_home()
            self._update_telemetry("idle")
            self._render_and_update_frame("IDLE (ホーム復帰)")
            self.add_log("ロボットアームがホーム位置に復帰しました。")

        elif c_type == "switch_model":
            ckpt = Path(cmd.get("checkpoint"))
            if ckpt.exists():
                self.is_running = False
                self.ckpt_path = ckpt
                self.policy, self.preprocessor, self.postprocessor = load_policy_and_processors(self.ckpt_path, self.device)
                img_feat = self.policy.config.input_features.get("observation.images.overhead")
                img_size = img_feat.shape[-1] if img_feat is not None else 512
                if self.env.image_size != img_size:
                    if self.env.renderer is not None:
                        self.env.renderer.close()
                        self.env.renderer = None
                    self.env.image_size = img_size
                self.add_log(f"モデル切替完了: {ckpt.parent.parent.parent.name} ({img_size}x{img_size})")
            else:
                self.add_log(f"チェックポイントが存在しません: {ckpt}")

        elif c_type == "set_speed":
            self.speed = float(cmd.get("speed", 1.0))

    def _smooth_home(self):
        """アームを滑らかにホーム姿勢へ移動"""
        m, d = self.env.model, self.env.data
        axes_current = d.qpos[:6].copy()
        axes_home = self.env.solve_ik(HOME_TCP, AXES_SEED, max_iters=200)
        for i in range(25):
            alpha = (i + 1) / 25.0
            d.qpos[:6] = (1 - alpha) * axes_current + alpha * axes_home
            d.qvel[:6] = 0.0
            d.ctrl[:6] = d.qpos[:6]
            mujoco.mj_step(m, d)
        self.current_state = self.env.get_state()

    def _step_rollout(self):
        """SmolVLAの1ステップ推論とMuJoCo環境更新"""
        if self.step_count >= self.max_steps:
            self.is_running = False
            self.success = self.env.check_success()
            status_str = "SUCCESS" if self.success else "TIMEOUT"
            self._update_telemetry("success" if self.success else "stopped")
            self._render_and_update_frame(status_str)
            self.add_log(f"最大ステップ数到達: 判定 = {status_str}")
            return

        spec = self.current_spec
        img_oh = self.env.render("overhead_camera")
        img_wr = self.env.render("wrist_camera")

        raw_obs = {
            "observation.state": torch.from_numpy(self.current_state).float(),
            "observation.images.overhead": torch.from_numpy(img_oh).permute(2, 0, 1).float() / 255.0,
            "observation.images.wrist": torch.from_numpy(img_wr).permute(2, 0, 1).float() / 255.0,
            "task": spec.instruction,
        }

        obs_proc = self.preprocessor(raw_obs)
        with torch.no_grad():
            action_tensor = self.policy.select_action(obs_proc)
        final_action = self.postprocessor(action_tensor).squeeze(0).cpu().numpy()

        self.current_state = self.env.step(final_action)
        self.step_count += 1

        if self.env.attached == spec.pick_color and not self.was_picked:
            self.was_picked = True
            self.add_log(f"[{self.step_count}st] 対象ブロック ({spec.pick_color}) を把持しました！")

        if self.was_picked and self.env.attached is None and self.current_state[6] < 0.5:
            self.add_log(f"[{self.step_count}st] 目標位置でブロックをリリースしました。")

        if self.env.check_success():
            self.success = True
            self.is_running = False
            self._update_telemetry("success")
            self._render_and_update_frame("SUCCESS (成功 🎉)")
            self.add_log(f"🎉 目標達成！{spec.kind.upper()} 成功 (Step: {self.step_count})")
            return

        self._update_telemetry("running")
        self._render_and_update_frame("RUNNING")

    def _update_telemetry(self, state_str):
        spec = self.current_spec
        tcp = self.env.tcp_pos()
        target_xy = spec.table_xy if spec.kind == "place" else self.env.block_pos(spec.stacked_blosks[-1])[:2]
        b_pos = self.env.block_pos(spec.pick_color)
        dist_t = np.linalg.norm(b_pos[:2] - target_xy)

        with self.lock:
            self.status["state"] = state_str
            self.status["step"] = self.step_count
            self.status["max_steps"] = self.max_steps
            self.status["tcp"] = [float(tcp[0]), float(tcp[1]), float(tcp[2])]
            self.status["gripper"] = float(self.current_state[6])
            self.status["attached"] = self.env.attached
            self.status["dist_target"] = float(dist_t)
            self.status["success"] = self.success
            self.status["elapsed_time"] = time.time() - self.start_time if self.start_time > 0 else 0.0

    def _render_and_update_frame(self, banner_status: str):
        """俯瞰・手首カメラの映像を合成し、HUDオーバーレイを描画してJPEGエンコード"""
        img_oh = self.env.render("overhead_camera")
        img_wr = self.env.render("wrist_camera")

        img_oh_bgr = cv2.cvtColor(img_oh, cv2.COLOR_RGB2BGR)
        img_wr_bgr = cv2.cvtColor(img_wr, cv2.COLOR_RGB2BGR)

        combined = np.hstack([img_oh_bgr, img_wr_bgr])
        # 鮮明表示のため 768x384 にリサイズ
        h_target, w_target = 384, 768
        combined = cv2.resize(combined, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

        cv2.putText(combined, "OVERHEAD VIEW (俯瞰)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined, "WRIST VIEW (手首カメラ)", (w_target // 2 + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(combined, (w_target // 2, 0), (w_target // 2, h_target), (45, 55, 72), 1)

        header = np.zeros((52, w_target, 3), dtype=np.uint8)
        spec = self.current_spec
        inst_text = f"TASK: \"{spec.instruction}\"" if spec else "TASK: --"
        if len(inst_text) > 75:
            inst_text = inst_text[:72] + "..."

        cv2.putText(header, inst_text, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        status_color = (0, 220, 100) if "SUCCESS" in banner_status else ((0, 200, 255) if "RUNNING" in banner_status else (160, 160, 160))
        tcp = self.env.tcp_pos()
        grip_str = "CLOSED" if self.current_state[6] > 0.5 else "OPEN"
        att_str = f" [Attached: {self.env.attached}]" if self.env.attached else ""
        sub_text = f"STATE: {banner_status} | Step: {self.step_count}/{self.max_steps} | Gripper: {grip_str}{att_str} | TCP: ({tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f})"
        cv2.putText(header, sub_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

        if self.success:
            cv2.rectangle(combined, (0, 0), (w_target - 1, h_target - 1), (0, 220, 100), 4)

        final_frame = np.vstack([header, combined])
        ret, buf = cv2.imencode(".jpg", final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            with self.lock:
                self.latest_frame = buf.tobytes()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def get_status(self):
        with self.lock:
            return dict(self.status)

    def request_reset(self, instruction, base_option="auto"):
        self.cmd_queue.put({"type": "reset", "instruction": instruction, "base_option": base_option})

    def request_run(self, instruction=None):
        self.cmd_queue.put({"type": "run", "instruction": instruction})

    def request_stop(self):
        self.cmd_queue.put({"type": "stop"})

    def request_home(self):
        self.cmd_queue.put({"type": "home"})

    def request_switch_model(self, ckpt_path):
        self.cmd_queue.put({"type": "switch_model", "checkpoint": ckpt_path})

    def set_speed(self, speed):
        self.cmd_queue.put({"type": "set_speed", "speed": speed})


worker = SimulationWorker()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def mjpeg_generator():
        while True:
            frame = worker.get_latest_frame()
            if frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/tasks")
def api_tasks():
    return jsonify({"tasks": TASK_CATALOG})


@app.route("/api/status")
def api_status():
    return jsonify(worker.get_status())


@app.route("/api/action/reset", methods=["POST"])
def api_reset():
    data = request.get_json(silent=True) or {}
    instruction = data.get("instruction", "stack the yellow block on the blue block")
    base_option = data.get("base_option", "auto")
    worker.request_reset(instruction, base_option)
    time.sleep(0.1)
    spec = worker.current_spec
    return jsonify({
        "status": "ok",
        "kind": spec.kind if spec else "unknown",
        "pick_color": spec.pick_color if spec else "unknown",
        "instruction": instruction,
    })


@app.route("/api/action/run", methods=["POST"])
def api_run():
    data = request.get_json(silent=True) or {}
    instruction = data.get("instruction")
    worker.request_run(instruction)
    return jsonify({"status": "ok"})


@app.route("/api/action/stop", methods=["POST"])
def api_stop():
    worker.request_stop()
    return jsonify({"status": "ok"})


@app.route("/api/action/home", methods=["POST"])
def api_home():
    worker.request_home()
    return jsonify({"status": "ok"})


@app.route("/api/action/switch_model", methods=["POST"])
def api_switch_model():
    data = request.get_json(silent=True) or {}
    ckpt = data.get("checkpoint", DEFAULT_CHECKPOINT)
    worker.request_switch_model(ckpt)
    return jsonify({"status": "ok"})


@app.route("/api/action/set_speed", methods=["POST"])
def api_set_speed():
    data = request.get_json(silent=True) or {}
    speed = float(data.get("speed", 1.0))
    worker.set_speed(speed)
    return jsonify({"status": "ok"})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UR5e SmolVLA Web GUI")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server host")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    args = parser.parse_args()

    worker.device = args.device
    worker.start()

    print("\n" + "=" * 65)
    print(f"🤖 UR5e SmolVLA Web GUI 起動中")
    print(f"🔗 ブラウザでアクセス: http://localhost:{args.port}")
    print("=" * 65 + "\n")

    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
