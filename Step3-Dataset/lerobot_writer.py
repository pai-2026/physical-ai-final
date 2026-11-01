"""
LeRobot データセット保存用ラッパークラス
Hugging Face LeRobot (v0.6.1) の公式フォーマットでエピソードを追記保存する。
"""
import os
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DatasetWriter:
    def __init__(self, repo_id: str, root: str, fps: int, image_size: int, camera_names: list[str]):
        self.repo_id = repo_id
        self.root = root
        self.fps = fps
        self.image_size = image_size
        self.camera_names = camera_names

        features = {
            "observation.state": {"dtype": "float32", "shape": (7,)},
            "action": {"dtype": "float32", "shape": (7,)},
        }
        for cam in camera_names:
            features[f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": (3, image_size, image_size),
                "names": ["channels", "height", "width"],
            }

        if os.path.exists(root):
            raise FileExistsError(f"保存先が既に存在します。別の --root を指定してください: {root}")

        print(f"LeRobot データセットを初期化中: repo_id={repo_id}, root={root}")
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            use_videos=True,
            image_writer_threads=4,
        )

    def add_episode(self, frames: list[dict], instruction: str):
        """1エピソード分のフレーム列と指示テキストを追加して保存"""
        for f in frames:
            frame_dict = {
                "observation.state": f["observation.state"].astype(np.float32),
                "action": f["action"].astype(np.float32),
                "task": instruction,
            }
            for cam in self.camera_names:
                # MuJoCo の (H, W, C) から LeRobot の (C, H, W) へ転置
                frame_dict[f"observation.images.{cam}"] = f[f"observation.images.{cam}"].transpose(2, 0, 1)

            self.dataset.add_frame(frame_dict)

        self.dataset.save_episode()

    def finish(self):
        """全エピソードのエンコードとメタデータを確定"""
        print("データセットをファイナライズ中...")
        self.dataset.finalize()
        print(f"データセット保存完了！総エピソード数: {self.dataset.num_episodes}, 総フレーム数: {len(self.dataset)}")
