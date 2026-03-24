"""该模块负责后台视频流的采集与处理线程.

作为中间件，将核心算法应用到摄像头捕获的每一帧，并推送到 UI 层.
"""

import cv2 as cv
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from core.detector import FaceMeshDetector
from core.processor import ImageProcessor
from utils.camera_utils import open_camera


class VideoThread(QThread):
    """执行“捕获-处理-渲染”闭环的后台线程."""

    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.running = True
        # 初始化完整的业务参数
        self.params = {
            "draw_mode": "mesh",
            "draw_on_left": True,
            "saturation": 1.0,
            "sharpness": 0.0,
            "smoothing": 0,
            "brighten": 1.0,
        }

    def update_parameters(self, new_params: dict):
        """实时更新线程内的运行参数."""
        self.params.update(new_params)

    def run(self):
        """线程执行入口."""
        cap = open_camera()
        detector = FaceMeshDetector(self.cfg["face-mesh"]["model_path"], 2)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 预处理：镜像翻转
            frame = cv.flip(frame, 1)

            # 2. 核心算法：应用全套美颜滤镜
            # 调用封装好的静态方法，保持代码整洁
            beauty_frame = ImageProcessor.apply_all_filters(frame, self.params)

            # 3. 核心算法：应用 MediaPipe 检测与 4 通道渲染
            left, right = detector.find_face_mesh(
                beauty_frame,
                draw_mode=self.params["draw_mode"],
                draw_on_left=self.params["draw_on_left"],
            )

            # 4. 拼接并推送到主界面 (保持 4 通道)
            combined = np.hstack((left, right))
            self.frame_signal.emit(combined)

        cap.release()
        detector.close()

    def stop(self):
        self.running = False
        self.wait()
