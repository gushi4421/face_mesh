"""
该模块负责后台视频源捕获、静态图片加载与 AI 检测推理.

作为中间桥梁，衔接 core 核心算法层与 ui 前端层.
"""
import cv2 as cv
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal
import shutil
import tempfile

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
        if (self.params["source_mode"] in ["camera", "video"] and 
            new_params.get("source_mode") == "image" and self.isRunning()):
            self.stop()
            
        self.params.update(new_params)

    def run(self):
        """线程主循环：升级为多模式检测流。"""
        self.running = True
        source_mode = self.params.get("source_mode", "camera")
        
        detector = FaceMeshDetector(
            self.cfg['face-mesh']['model_path'], 
            self.params['max_faces']
        )
        cap = None

        # --- 模式选择与数据源挂载 ---
        if source_mode == "camera":
            cap = open_camera()
            
        elif source_mode == "video":
            vid_path = self.params["video_src_path"]
            if not vid_path or not os.path.isfile(vid_path): return
                
            if not vid_path.isascii():
                logger.info("检测到非 ASCII 路径，正在创建临时视频副本以供 OpenCV 读取...")
                temp_dir = tempfile.gettempdir()
                ext = os.path.splitext(vid_path)[1]
                temp_vid_path = os.path.join(temp_dir, f"temp_face_mesh_video{ext}")
                try:
                    shutil.copy2(vid_path, temp_vid_path)
                    vid_path = temp_vid_path
                except Exception as e:
                    logger.error(f"复制视频缓存失败: {e}")
            cap = cv.VideoCapture(vid_path)
            
        elif source_mode == "image":
            img_path = self.params["image_src_path"]
            if not img_path or not os.path.isfile(img_path): return
            
            frame = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                logger.error(f"解析静态图片失败: {img_path}")
                return
                
            # 执行单次管线推理
            beauty_frame = ImageProcessor.apply_all_filters(frame, self.params)
            left, right = detector.find_face_mesh(
                beauty_frame, draw_mode=self.params['draw_mode'],
                draw_on_left=self.params['draw_on_left']
            )
            combined = np.hstack((left, right))
            self.frame_signal.emit(combined)
            
            # 推理完毕，安全释放并退出线程
            detector.close()
            return 

        # --- 流数据循环 (摄像头/视频) ---
        if cap and cap.isOpened():
            while self.running:
                ret, frame = cap.read()
                if not ret: break # 视频播放结束或摄像头断开
                    
                # 仅在摄像头模式下镜像翻转画面，播放本地视频不应该翻转
                if source_mode == "camera":
                    frame = cv.flip(frame, 1)

                beauty_frame = ImageProcessor.apply_all_filters(frame, self.params)
                left, right = detector.find_face_mesh(
                    beauty_frame, draw_mode=self.params['draw_mode'],
                    draw_on_left=self.params['draw_on_left']
                )
                combined = np.hstack((left, right))
                self.frame_signal.emit(combined)
                
            cap.release()
            
        detector.close()
    def stop(self):
        """安全停止流模式，但不影响静态图片单次执行。"""
        self.running = False
        self.wait()