import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
import sys

from src.log import logger


class FaceMeshDetector:
    def __init__(
        self,
        model_path: str,
        max_faces: int,
    ):
        """
        model_path: 模型文件路径
        max_faces: 最大人脸数
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file.resolve()}")
        if not model_file.is_file():
            raise ValueError(f"模型路径不是文件: {model_file.resolve()}")

        self.model_path = model_path
        self.max_faces = max_faces
        self.landmarker = self._get_landmarker()
        logger.info("人脸网格检测器被创建")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """释放 MediaPipe 资源"""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
            logger.info("人脸网格检测器资源已释放")

    def _get_landmarker(self):
        """获取人脸检测器"""
        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )  # 创建基础配置
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,  # 基础配置
            num_faces=self.max_faces,  # 最大识别人脸数
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return landmarker

    def find_face_mesh(self, frame, draw: bool = True, left_draw_mode: str = "mesh"):
        """
        在一帧图像中寻找人脸网格, 并支持左侧图像模式的二选一.

        Args:
            frame: 输入的 BGR 格式图像.
            draw: 全局可视化开关.
            left_draw_mode: 左侧原图的绘制模式 ("mesh" 为连线, "points" 为特征点).
        """
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mediapipe_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.landmarker.detect(mediapipe_img)

        if not draw:
            return (
                frame,
                np.zeros(frame.shape, np.uint8),
                detection_result.face_landmarks,
            )

        frame_shape = frame.shape
        processed_frame = frame.copy()
        skeleton_img = np.zeros(frame_shape, np.uint8)

        if detection_result.face_landmarks and draw:
            for face_landmarks in detection_result.face_landmarks:

                # 步骤一: 优先绘制所有的网格连线 (作为底层)
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
                        h, w, _ = frame_shape
                        start_point = (
                            int(start_landmark.x * w),
                            int(start_landmark.y * h),
                        )
                        end_point = (
                            int(end_landmark.x * w),
                            int(end_landmark.y * h),
                        )

                        # 右侧骨架图: 始终无条件绘制绿色连线
                        cv.line(
                            img=skeleton_img,
                            pt1=start_point,
                            pt2=end_point,
                            color=(0, 255, 0),
                            thickness=1,
                        )

                        # 左侧原图: 仅当模式为 mesh 时才绘制连线
                        if left_draw_mode == "mesh":
                            cv.line(
                                img=processed_frame,
                                pt1=start_point,
                                pt2=end_point,
                                color=(0, 255, 0),
                                thickness=1,
                            )

                # 步骤二: 随后绘制所有的特征点 (作为表层, 防止被绿线覆盖)
                for landmark in face_landmarks:
                    h, w, _ = frame_shape
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    # 右侧骨架图: 始终无条件绘制红色特征点, 并且将半径调至 2, 确保在绿线上方清晰可见
                    cv.circle(
                        img=skeleton_img,
                        center=(x, y),
                        radius=2,
                        color=(0, 0, 255),
                        thickness=-1,
                    )

                    # 左侧原图: 仅当模式为 points 时才绘制特征点
                    if left_draw_mode == "points":
                        cv.circle(
                            img=processed_frame,
                            center=(x, y),
                            radius=1,
                            color=(0, 0, 255),
                            thickness=-1,
                        )

        return processed_frame, skeleton_img, detection_result.face_landmarks

    def img_combine(self, img1, img2):
        """水平拼接两个图像"""
        # 当两图高度相同时，直接使用 hstack（本项目中始终成立）
        if img1.shape[0] == img2.shape[0]:
            return np.hstack((img1, img2))

        # 高度不同时，对齐到最大高度
        max_h = max(img1.shape[0], img2.shape[0])
        pad1 = np.zeros(
            (max_h - img1.shape[0], img1.shape[1], *img1.shape[2:]), np.uint8
        )
        pad2 = np.zeros(
            (max_h - img2.shape[0], img2.shape[1], *img2.shape[2:]), np.uint8
        )
        return np.hstack((np.vstack((img1, pad1)), np.vstack((img2, pad2))))
