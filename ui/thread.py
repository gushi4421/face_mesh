"""该模块封装视频处理线程.

作为中间桥梁，连接 core 算法层与 ui 呈现层.
"""

import cv2 as cv
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.detector import FaceMeshDetector
from core.processor import ImageProcessor
from utils.camera_utils import open_camera


class VideoThread(QThread):
    """后台线程，执行摄像头捕获、美颜与人脸检测."""

    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.running = True
        self.params = {
            "max_faces": config["face-mesh"]["initial_max_faces"],
            "draw_mode": config["face-mesh"]["draw_mode"],
            "draw_on_left": config["face-mesh"]["draw_on_left"],
            "smoothing": 0,
        }

    def update_params(self, new_params: dict):
        """更新线程内的运行参数."""
        self.params.update(new_params)

    def run(self):
        """线程主循环."""
        cap = open_camera()
        detector = FaceMeshDetector(self.cfg["face-mesh"]["model_path"], 5)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)

            # 1. 美颜
            frame = ImageProcessor.apply_skin_smoothing(frame, self.params["smoothing"])

            # 2. 检测
            left, right = detector.find_face_mesh(
                frame,
                draw_mode=self.params["draw_mode"],
                draw_on_left=self.params["draw_on_left"],
            )

            # 3. 拼接并发送 (hstack 会保留 4 通道)
            combined = np.hstack((left, right))
            self.frame_signal.emit(combined)

        cap.release()
        detector.close()

    def stop(self):
        self.running = False
        self.wait()
