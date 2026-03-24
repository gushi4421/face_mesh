"""该模块封装了所有针对单帧图像的美颜与滤镜算法.

包含饱和度、锐化、磨皮及美白处理逻辑.
"""

import cv2 as cv
import numpy as np


class ImageProcessor:
    """提供图像后处理滤镜的类."""

    @staticmethod
    def adjust_saturation(image: np.ndarray, scale: float) -> np.ndarray:
        """调整 BGR 图像的饱和度."""
        if scale == 1.0:
            return image
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

    @staticmethod
    def apply_sharpening(image: np.ndarray, intensity: float) -> np.ndarray:
        """通过拉普拉斯算子对图像进行锐化."""
        if intensity == 0.0:
            return image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        id_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        dyn_kernel = id_kernel + intensity * (kernel - id_kernel)
        return cv.filter2D(image, -1, dyn_kernel)

    @staticmethod
    def apply_skin_smoothing(image: np.ndarray, intensity: int) -> np.ndarray:
        """使用双边滤波实现磨皮."""
        if intensity <= 0:
            return image
        d = int(intensity / 10) + 5
        return cv.bilateralFilter(image, d, intensity + 20, intensity + 20)

    @staticmethod
    def apply_skin_brightening(image: np.ndarray, gamma: float) -> np.ndarray:
        """使用 LUT 查表法实现 Gamma 亮度校正."""
        if gamma == 1.0:
            return image
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        return cv.LUT(image, table)
