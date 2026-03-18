import cv2 as cv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from log import logger


def open_source(source_path: str) -> cv.VideoCapture:
    """打开视频源"""
    capture = cv.VideoCapture(source_path)
    if not capture.isOpened():
        logger.error(f"无法打开视频源: {source_path}")
    return capture


def open_camera() -> cv.VideoCapture:
    """打开摄像头"""
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        logger.error("无法打开摄像头")
    return capture
