"""
该模块封装视频处理线程.

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
            "smoothing": 0, # 磨皮强度
            "brighten": 1.0,    # 亮度系数
            "saturation": 1.0,  # 饱和度系数
            "sharpness": 0.0    # 锐度强度
        }

    def update_parameters(self, new_params: dict):
        """实时更新参数字典。
        
        Args:
            new_params: 待更新参数的字典
        """
        self.params.update(new_params)

    def run(self):
        """线程主循环: 执行摄像头采集、图像处理与推理"""
        # 开启摄像头
        cap = open_camera()
        
        # 初始化检测器
        detector = FaceMeshDetector(
            self.cfg['face-mesh']['model_path'], 
            self.params['max_faces']
        )
        
        while self.running:
            ret, frame = cap.read() # 第七一帧的画面
            if not ret: 
                break
            
            # 水平翻转
            frame = cv.flip(frame, 1)   

            # 执行美颜算法
            frame = ImageProcessor.apply_all_filters(frame, self.params)
            
            # 执行推理与绘制
            left, right = detector.find_face_mesh(
                frame, 
                draw_mode=self.params['draw_mode'],
                draw_on_left=self.params['draw_on_left']
            )
            
            # 拼接并发送结果
            combined = np.hstack((left, right))

            self.frame_signal.emit(combined)
        # 释放资源
        cap.release()
        detector.close()

    def stop(self):
        """安全停止。"""
        self.running = False
        self.wait()