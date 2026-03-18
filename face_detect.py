import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from log import logger


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
        logger.info("人脸网格检测器被创建")

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

    def find_face_mesh(self, frame, draw: bool):
        """
        在一帧图像中寻找人脸网格，并手动绘制它们
        img: 输入的图片(格式为BGR)
        draw: 是否绘制网络
        """
        # 将 BGR 格式转变为 RGB
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # 将 NumPy 数组转换为 MediaPipe 图像格式
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_RGB)
        # 使用人脸检测器进行检测
        detection_result = self.landmarker.detect(mp_img)

        processed_frame = frame.copy()
        skeleton_img = np.zeros(frame.shape, np.uint8)

        if detection_result.face_landmarks and draw:
            for face_landmarks in detection_result.face_landmarks:
                for (
                    connection
                ) in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                    start_index = connection.start
                    end_index = connection.end

                    if start_index < len(face_landmarks) and end_index < len(
                        face_landmarks
                    ):
                        start_landmark = face_landmarks[start_index]
                        end_landmark = face_landmarks[end_index]
                        h, w, _ = frame.shape
                        start_point = (
                            int(start_landmark.x * w),
                            int(start_landmark.y * h),
                        )
                        end_point = (
                            int(end_landmark.x * w),
                            int(end_landmark.y * h),
                        )

                        """
                        cv.line参数解析
                            img: 要绘制的图像
                            pt1: 线段的起点坐标 (x, y)
                            pt2: 线段的终点坐标 (x, y)
                            color: 线段的颜色 (B, G, R)
                            thickness: 线段的粗细 (以像素为单位)
                        """

                        cv.line(
                            img=skeleton_img,
                            pt1=start_point,
                            pt2=end_point,
                            color=(0, 255, 0),
                            thickness=1,
                        )
                        cv.line(
                            img=processed_frame,
                            pt1=start_point,
                            pt2=end_point,
                            color=(0, 255, 0),
                            thickness=1,
                        )
                for landmark in face_landmarks:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv.circle(
                        img=processed_frame,
                        center=(x, y),
                        radius=1,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
                    cv.circle(
                        img=skeleton_img,
                        center=(x, y),
                        radius=1,
                        color=(0, 0, 255),
                        thickness=-1,
                    )
        return processed_frame, skeleton_img, detection_result.face_landmarks

    def img_combine(self, img1, img2):
        """水平拼接两个图像"""
        if len(img1.shape) == 3:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            dst = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
            dst[:, :w1] = img1
            dst[:, w1:] = img2
        else:
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            dst = np.zeros((max(h1, h2), w1 + w2), np.uint8)
            dst[:, :w1] = img1
            dst[:, w1:] = img2

        return dst
