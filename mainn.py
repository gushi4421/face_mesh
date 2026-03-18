import os
from pathlib import Path
import cv2 as cv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

try:
    from face_detect import FaceMeshDetector
except ImportError as e:
    print(
        "无法导入 FaceMeshDetector 类，请检查 face_detect.py 文件是否存在，并且路径正确。"
    )
    print("错误详情:", e)
    exit(1)
