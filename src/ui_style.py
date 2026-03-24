"""
文件级注释: 本文件负责 GUI 界面的样式与美化控制.
包含 BackgroundWidget 类, 用于实现带有自适应半透明背景图片的主容器部件.
"""
import os
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt

class BackgroundWidget(QWidget):
    """
    类级注释: 自定义的主容器部件, 支持安全的背景图片渲染与自定义透明度.
    核心逻辑: 重写 paintEvent 方法. 若图片有效则按指定透明度绘制, 否则使用默认灰白底色防崩溃.
    """
    def __init__(self, parent=None):
        """
        方法级注释: 初始化容器部件, 默认不加载图片且完全不透明.
        """
        super().__init__(parent)
        self.bg_pixmap = None
        self.opacity = 1.0

    def set_background_image(self, image_path: str):
        """
        方法级注释: 安全加载并设置背景图片.
        包含严格的安全检测机制, 确保图片不存在或损坏时程序能够稳定运行.
        """
        # 1. 安全检测: 验证路径是否存在且确实是一个文件
        if image_path and os.path.isfile(image_path):
            pixmap = QPixmap(image_path)
            
            # 2. 安全检测: 验证图片数据是否有效 (防止用户选择了一个损坏的文件)
            if not pixmap.isNull():
                self.bg_pixmap = pixmap
            else:
                self.bg_pixmap = None
        else:
            self.bg_pixmap = None
            
        # 触发重新绘制事件, 立刻刷新界面
        self.update()

    def set_background_opacity(self, opacity: float):
        """
        方法级注释: 动态更新背景图片的透明度.
        """
        self.opacity = opacity
        self.update()

    def paintEvent(self, event):
        """
        方法级注释: 重写 Qt 的底层绘图事件, 实现半透明背景渲染.
        核心逻辑: 先画安全底色, 再叠加带 Alpha 通道的用户图片.
        """
        painter = QPainter(self)
        
        # 核心逻辑: 首先绘制默认的灰白底色
        # 当背景图片半透明或加载失败时, 这个颜色将作为安全底色展示
        painter.fillRect(self.rect(), Qt.lightGray)
        
        if self.bg_pixmap is not None:
            # 开启平滑抗锯齿
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            # 核心逻辑: 设置画笔的全局透明度, 从而实现图片的半透明效果
            painter.setOpacity(self.opacity)
            
            # 平滑缩放背景图片以铺满当前窗口的实际大小
            scaled_pixmap = self.bg_pixmap.scaled(
                self.size(), 
                Qt.IgnoreAspectRatio, 
                Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_pixmap)