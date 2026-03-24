"""
本文件专门负责图像的后处理与高级美颜特效算法.
包含方法: adjust_saturation, apply_sharpening, apply_skin_smoothing, apply_skin_brightening, apply_beauty_filters.
"""
import cv2 as cv
import numpy as np

def adjust_saturation(image: np.ndarray, scale: float) -> np.ndarray:
    """
    调整图像饱和度.
    核心逻辑: 将 BGR 转换为 HSV, 缩放 S 通道后再转换回 BGR.
    """
    if scale == 1.0:
        return image
        
    # 转换为 HSV 色彩空间进行处理
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * scale
    
    # 防止像素值溢出并转换回 uint8
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

def apply_sharpening(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    对图像应用锐化滤镜.
    核心逻辑: 使用拉普拉斯算子结合原图构建动态卷积核.
    """
    if intensity == 0.0:
        return image
        
    # 定义基础拉普拉斯算子
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # 动态计算插值卷积核
    identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    dynamic_kernel = identity_kernel + intensity * (kernel - identity_kernel)
    
    return cv.filter2D(image, -1, dynamic_kernel)

def apply_skin_smoothing(image: np.ndarray, intensity: int) -> np.ndarray:
    """
    使用双边滤波实现磨皮效果, 在平滑皮肤的同时保留五官边缘细节.
    """
    if intensity <= 0:
        return image
        
    # 动态计算双边滤波参数
    d = int(intensity / 10) + 5
    sigma_color = intensity + 20
    sigma_space = intensity + 20
    
    return cv.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_skin_brightening(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    使用 Gamma 校正实现美白和提亮效果.
    核心逻辑: 构建 LUT 查找表以实现 O(1) 复杂度的像素替换.
    """
    if gamma == 1.0:
        return image
        
    # 预计算查找表以加速推理
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv.LUT(image, table)

def apply_beauty_filters(image: np.ndarray, saturation: float, sharpness: float, smoothing: int, brighten: float) -> np.ndarray:
    """
    统一的美颜处理入口函数.
    """
    # 按顺序应用滤镜
    img = apply_skin_smoothing(image, smoothing)
    img = apply_skin_brightening(img, brighten)
    img = adjust_saturation(img, saturation)
    img = apply_sharpening(img, sharpness)
    
    return img