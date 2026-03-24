"""该模块封装了所有图像后处理与美颜特效算法.

包含饱和度、锐化、磨皮及美白处理的核心计算逻辑.
"""

import cv2 as cv
import numpy as np


class ImageProcessor:
    """提供图像滤镜处理功能的静态类."""

    @staticmethod
    def adjust_saturation(image: np.ndarray, scale: float) -> np.ndarray:
        """调整图像饱和度."""
        if scale == 1.0:
            return image
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

    @staticmethod
    def apply_sharpening(image: np.ndarray, intensity: float) -> np.ndarray:
        """应用动态锐化滤镜."""
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
        """使用 Gamma 校正实现美白."""
        if gamma == 1.0:
            return image
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        return cv.LUT(image, table)

    @classmethod
    def apply_all_filters(cls, image: np.ndarray, params: dict) -> np.ndarray:
        """一键应用所有美颜参数.

        Args:
            image: 输入的 BGR 图像.
            params: 包含各种强度参数的字典.

        Returns:
            处理后的图像.
        """
        # 按照工业标准顺序进行滤镜叠加
        img = cls.apply_skin_smoothing(image, params.get("smoothing", 0))
        img = cls.apply_skin_brightening(img, params.get("brighten", 1.0))
        img = cls.adjust_saturation(img, params.get("saturation", 1.0))
        img = cls.apply_sharpening(img, params.get("sharpness", 0.0))
        return img
