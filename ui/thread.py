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
        """初始化线程。"""
        super().__init__()
        self.cfg = config
        self.running = True
        # 初始化运行参数
        self.params = {
            "max_faces": config['face-mesh'].get('initial_max_faces', 2),
            "draw_mode": config['face-mesh'].get('draw_mode', 'mesh'),
            "draw_on_left": config['face-mesh'].get('draw_on_left', True),
            "smoothing": 0,
            "brighten": 1.0,
            "saturation": 1.0,
            "sharpness": 0.0
        }

    def update_parameters(self, new_params: dict):
        """实时更新参数字典。"""
        self.params.update(new_params)

    def run(self):
        """线程主循环。"""
        cap = open_camera()
        
        # 核心逻辑：使用当前参数中的 max_faces 初始化检测器
        # 注意：如果用户在运行中修改了脸数，通常需要重启线程或重新初始化 detector 才能生效
        detector = FaceMeshDetector(
            self.cfg['face-mesh']['model_path'], 
            self.params['max_faces']
        )
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            frame = cv.flip(frame, 1)

            # 1. 执行全套美颜算法
            frame = ImageProcessor.apply_all_filters(frame, self.params)
            
            # 2. 执行 AI 推理与 4 通道绘制
            left, right = detector.find_face_mesh(
                frame, 
                draw_mode=self.params['draw_mode'],
                draw_on_left=self.params['draw_on_left']
            )
            
            # 3. 拼接并发送结果
            combined = np.hstack((left, right))
            self.frame_signal.emit(combined)
            
        cap.release()
        detector.close()

    def stop(self):
        """安全停止。"""
        self.running = False
        self.wait()