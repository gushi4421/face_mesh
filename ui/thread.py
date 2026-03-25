"""
该模块负责后台视频源捕获、静态图片加载与 AI 检测推理.

作为中间桥梁，衔接 core 核心算法层与 ui 前端层.
"""
import cv2 as cv
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal

from core.detector import FaceMeshDetector
from core.processor import ImageProcessor
from utils.camera_utils import open_camera
from utils.logger import logger


class VideoThread(QThread):
    """通用检测线程，支持摄像头、静态图片及本地视频文件模式."""
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, config: dict):
        """初始化线程."""
        super().__init__()
        self.cfg = config
        self.running = True
        # 初始化运行参数，新增数据源相关参数
        self.params = {
            "source_mode": "camera", # "camera", "image", "video"
            "image_src_path": "",
            "video_src_path": "",
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
        # 核心修改1：如果从流模式切换到图片模式，应当停止当前线程循环
        if (self.params["source_mode"] in ["camera", "video"] and 
            new_params.get("source_mode") == "image" and self.isRunning()):
            self.stop()
            
        self.params.update(new_params)

    def run(self):
        """线程主循环：升级为多模式检测流。"""
        self.running = True
        source_mode = self.params["source_mode"]
        logger.info(f"开启检测线程，模式: {source_mode}")
        
        # 使用当前参数初始化检测器
        detector = FaceMeshDetector(
            self.cfg['face-mesh']['model_path'], 
            self.params['max_faces']
        )
        
        cap = None

        if source_mode == "camera":
            # 实时摄像头
            cap = open_camera()
            
        elif source_mode == "video":
            # 本地视频文件
            vid_path = self.params["video_src_path"]
            if not vid_path or not os.path.isfile(vid_path):
                logger.error("视频文件不存在或路径为空。")
                self.frame_signal.emit(np.zeros((480, 640, 4), dtype=np.uint8)) # 发送空帧
                return
            cap = cv.VideoCapture(vid_path)
            
        elif source_mode == "image":
            img_path = self.params["image_src_path"]
            if not img_path or not os.path.isfile(img_path):
                logger.error("图片文件不存在或路径为空。")
                return
            
            # 1. 读取单张图片
            frame = cv.imread(img_path)
            if frame is None:
                logger.error("读取图片失败。")
                return
                
            # 2. 执行算法处理单管线
            beauty_frame = ImageProcessor.apply_all_filters(frame, self.params)
            left, right = detector.find_face_mesh(
                beauty_frame, draw_mode=self.params['draw_mode'],
                draw_on_left=self.params['draw_on_left']
            )
            # 3. 拼接并单次发射
            combined = np.hstack((left, right))
            self.frame_signal.emit(combined)
            
            # 4. 处理完毕，优雅退出
            detector.close()
            logger.info("单张图片检测完成，线程退出。")
            return 

        if cap and cap.isOpened():
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    # 如果是视频流模式，ret为False通常意味着视频播放完毕
                    if source_mode == "video":
                        logger.info("视频播放完毕。")
                    break
                    
                frame = cv.flip(frame, 1) # 视频/摄像头模式通常需要镜像

                # 1. 执行全套美颜算法
                frame = ImageProcessor.apply_all_filters(frame, self.params)
                
                # 2. 执行 AI 推理与 4 通道绘制
                left, right = detector.find_face_mesh(
                    frame, draw_mode=self.params['draw_mode'],
                    draw_on_left=self.params['draw_on_left']
                )
                
                # 3. 拼接结果并展示
                combined = np.hstack((left, right))
                self.frame_signal.emit(combined)
                
            cap.release()
            
        detector.close()
        logger.info("检测流线程停止。")

    def stop(self):
        """安全停止流模式，但不影响静态图片单次执行。"""
        self.running = False
        self.wait()