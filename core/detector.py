"""该模块实现 MediaPipe 面部网格检测的渲染逻辑.

支持 4 通道透明渲染、左右画面边框绘制及双视图解耦展示.
"""

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np


class FaceMeshDetector:
    """封装 MediaPipe 检测器，执行坐标计算与绘制任务."""

    def __init__(self, model_path: str, num_faces: int):
        """初始化检测器.

        Args:
            model_path: .task 模型路径.
            num_faces: 最大检测人脸数.
        """
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            num_faces=num_faces,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def find_face_mesh(
        self, frame: np.ndarray, draw_mode: str = "mesh", draw_on_left: bool = True
    ):
        """检测面部网格并生成 4 通道渲染结果.

        Returns:
            tuple: (左侧带检测结果的实拍图, 右侧带网格的实体黑框图).
        """
        h, w = frame.shape[:2]
        # 左侧转 4 通道
        processed_frame = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
        # 右侧初始化为 4 通道实体黑块 (Alpha=255)
        skeleton_img = np.zeros((h, w, 4), dtype=np.uint8)
        skeleton_img[:, :, 3] = 255

        # 绘制白色边框
        cv.rectangle(processed_frame, (0, 0), (w - 1, h - 1), (255, 255, 255, 255), 2)
        cv.rectangle(skeleton_img, (0, 0), (w - 1, h - 1), (255, 255, 255, 255), 2)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        )
        res = self.landmarker.detect(mp_img)

        if res.face_landmarks:
            for face in res.face_landmarks:
                if draw_mode == "mesh":
                    for (
                        conn
                    ) in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                        p1 = (int(face[conn.start].x * w), int(face[conn.start].y * h))
                        p2 = (int(face[conn.end].x * w), int(face[conn.end].y * h))
                        cv.line(skeleton_img, p1, p2, (0, 255, 0, 255), 1)
                        if draw_on_left:
                            cv.line(processed_frame, p1, p2, (0, 255, 0, 255), 1)
                elif draw_mode == "points":
                    for lm in face:
                        p = (int(lm.x * w), int(lm.y * h))
                        cv.circle(skeleton_img, p, 2, (0, 0, 255, 255), -1)
                        if draw_on_left:
                            cv.circle(processed_frame, p, 1, (0, 0, 255, 255), -1)

        return processed_frame, skeleton_img

    def close(self):
        """释放检测器资源."""
        self.landmarker.close()
