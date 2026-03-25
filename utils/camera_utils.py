"""
该模块提供摄像头与视频流的开启工具.

基于 OpenCV 实现对本地硬件的访问.
"""
import cv2 as cv
from utils.logger import logger


def open_camera(index: int = 0) -> cv.VideoCapture:
    """开启指定的本地摄像头.

    Args:
        index: 摄像头索引，默认 0.

    Returns:
        cv.VideoCapture 实例.

    Raises:
        RuntimeError: 摄像头无法打开时抛出.
    """
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        logger.error(f"无法开启索引为 {index} 的摄像头.")
        raise RuntimeError("Camera Error")
    return cap