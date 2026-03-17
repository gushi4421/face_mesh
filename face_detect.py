import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class FaceMeshDetector:
    def __init__(
        self,
        model_path: str,
        maxFaces: int,
    ):
        """
        model_path: 模型文件路径
        maxFaces: 最大人脸数
        """
        self.model_path = model_path
        self.maxFaces = maxFaces
        self.landmarker = self._get_landmarker()

    def _get_landmarker(self):
        """获取人脸检测器"""
        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )  # 创建基础配置
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,  # 基础配置
            num_faces=self.maxFaces,  # 最大识别人脸数
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return landmarker

    def find_face_mesh(self, img, draw: bool):
        """
        在一帧图像中寻找人脸网格，并手动绘制它们
        img: 输入的图片(格式为BGR)
        draw: 是否绘制网络
        """
        img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将 BGR 格式转变为 RGB
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_RGB)
        detection_result = self.landmarker.detect(mp_img)

        processed_img = img.copy()
