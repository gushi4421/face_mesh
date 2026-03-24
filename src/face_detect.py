import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path

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

    def find_face_mesh(
        self,
        frame,
        draw: bool = True,
        draw_mode: str = "mesh",
        draw_on_left: bool = True,
    ):
        """
        方法级注释: 在一帧图像中寻找人脸网格.
        核心修改: 引入 4 通道 (RGBA) 图像处理, 恢复右侧实体黑块, 并为左右画面绘制悬浮边框.
        """
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mediapipe_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.landmarker.detect(mediapipe_img)

        frame_shape = frame.shape

        # 1. 左侧画面: 将实拍原图转换为 4 通道 (BGRA)
        processed_frame = cv.cvtColor(frame.copy(), cv.COLOR_BGR2BGRA)

        # 2. 右侧画面: 创建一个 4 通道的全零矩阵
        skeleton_img = np.zeros((frame_shape[0], frame_shape[1], 4), np.uint8)

        # 核心修改 A：将右侧矩阵的 Alpha (透明度) 通道全部设为 255
        # 这会让右侧的画布重新变成“百分百实体”的黑块, 而不再是透明的
        skeleton_img[:, :, 3] = 255

        # 核心修改 B：为左右两个画面绘制高亮边框
        # 颜色格式为 (B, G, R, Alpha), 这里使用纯白色实体边框
        border_color = (255, 255, 255, 255)
        thickness = 2
        # 给左侧摄像头画面画边框
        cv.rectangle(
            processed_frame,
            (0, 0),
            (frame_shape[1] - 1, frame_shape[0] - 1),
            border_color,
            thickness,
        )
        # 给右侧实体黑块画边框
        cv.rectangle(
            skeleton_img,
            (0, 0),
            (frame_shape[1] - 1, frame_shape[0] - 1),
            border_color,
            thickness,
        )

        if not draw:
            return processed_frame, skeleton_img, detection_result.face_landmarks

        if detection_result.face_landmarks and draw:
            for face_landmarks in detection_result.face_landmarks:

                # 绘制网格模式
                if draw_mode == "mesh":
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

                            # 画笔颜色加入第 4 个参数 255, 确保线条本身也是不透明的
                            cv.line(
                                img=skeleton_img,
                                pt1=start_point,
                                pt2=end_point,
                                color=(0, 255, 0, 255),
                                thickness=1,
                            )
                            if draw_on_left:
                                cv.line(
                                    img=processed_frame,
                                    pt1=start_point,
                                    pt2=end_point,
                                    color=(0, 255, 0, 255),
                                    thickness=1,
                                )

                # 绘制特征散点模式
                elif draw_mode == "points":
                    for landmark in face_landmarks:
                        h, w, _ = frame_shape
                        x, y = int(landmark.x * w), int(landmark.y * h)

                        # 同理, 红色散点也加入 Alpha 通道参数 255
                        cv.circle(
                            img=skeleton_img,
                            center=(x, y),
                            radius=2,
                            color=(0, 0, 255, 255),
                            thickness=-1,
                        )
                        if draw_on_left:
                            cv.circle(
                                img=processed_frame,
                                center=(x, y),
                                radius=1,
                                color=(0, 0, 255, 255),
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
