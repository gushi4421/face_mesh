"""
文件级注释: 本文件是 GUI 应用的主入口点.
集成了 BackgroundWidget 容器, 以及解决防漏光黑底问题的 QSS 全局样式表.
"""

import sys
import os
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QGroupBox,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize

from src.tools.load_config import load_config
from src.gui_components import VideoThread, ParameterPanel
from src.ui_style import BackgroundWidget


class MainWindow(QMainWindow):
    """
    类级注释: 主窗口控制器.
    负责界面的宏观布局排版以及跨组件的信号路由转发.
    """

    def __init__(self, config: dict):
        super().__init__()
        gui_config = config["gui"]

        self.setWindowTitle(gui_config["window_title"])
        self.resize(gui_config["window_width"], gui_config["window_height"])

        self.bg_image_path = gui_config.get("bg_image_path", "")
        self.initial_text = gui_config["initial_text"]

        self.thread = VideoThread(config)
        self.init_ui(gui_config, full_config=config)

    def apply_stylesheet(self):
        """
        方法级注释: 应用 Qt Style Sheets (QSS) 美化界面.
        核心逻辑: 精确设置 ID 选择器, 确保视频区域显示为纯黑, 控制台显示为半透明暗色.
        """
        qss = """
        /* 基础 GroupBox 样式 */
        QGroupBox {
            border: 1px solid rgba(255, 255, 255, 60);
            border-radius: 10px;
            margin-top: 20px;
            color: #FFFFFF;
            font-weight: bold;
            font-size: 18px;
        }
        
        /* 仅给右侧的控制台加上半透明暗色底漆 */
        QGroupBox#paramGroup {
            background-color: rgba(20, 20, 20, 180);
        }
        
        QGroupBox#videoGroup {
            background-color: transparent;
        }
        
        /* 核心修改1: 将视频控件的背景恢复为全透明, 去除上下的巨大黑边 */
        QGroupBox#videoGroup QLabel {
            background-color: transparent; 
            border-radius: 5px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            color: #FFFFFF;
        }

        QLabel, QCheckBox {
            color: #FFFFFF;
            background: transparent;
            font-size: 15px;
        }

        QPushButton, QComboBox {
            background-color: rgba(60, 60, 60, 200);
            color: #FFFFFF;
            border: 1px solid #555555;
            border-radius: 5px;
            padding: 8px;
            font-size: 15px;
        }

        QPushButton:hover {
            background-color: rgba(90, 90, 90, 220);
            border: 1px solid #888888;
        }

        QPushButton:disabled {
            background-color: rgba(40, 40, 40, 100);
            color: #888888;
        }
        """
        self.setStyleSheet(qss)

    def init_ui(self, gui_config: dict, full_config: dict):
        self.central_widget = BackgroundWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.set_background_image(self.bg_image_path)
        self.central_widget.set_background_opacity(gui_config.get("bg_opacity", 0.6))

        self.main_h_layout = QHBoxLayout(self.central_widget)

        # ---------------- 左侧视频区 ----------------
        self.left_layout = QVBoxLayout()
        self.main_h_layout.addLayout(self.left_layout, 4)

        self.video_group = QGroupBox("实时监控 (带全套美颜处理)")
        self.video_group.setObjectName("videoGroup")
        self.video_layout = QVBoxLayout()
        self.video_group.setLayout(self.video_layout)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText(self.initial_text)
        self.video_label.setMinimumSize(QSize(640, 480))

        self.video_layout.addWidget(self.video_label)
        self.left_layout.addWidget(self.video_group)

        self.btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)

        self.btn_layout.addWidget(self.start_btn)
        self.btn_layout.addWidget(self.stop_btn)
        self.left_layout.addLayout(self.btn_layout)

        # ---------------- 右侧控制台 ----------------
        self.right_layout = QVBoxLayout()
        self.main_h_layout.addLayout(self.right_layout, 1)

        self.param_group = QGroupBox("控制台")
        self.param_group.setObjectName("paramGroup")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)

        self.param_panel = ParameterPanel(full_config)
        self.param_panel.sig_parameters_changed.connect(self.on_params_updated)
        self.param_layout.addWidget(self.param_panel)

        self.right_layout.addWidget(self.param_group)
        self.right_layout.addStretch()

        # 加载样式表以应用 ID 隔离规则
        self.apply_stylesheet()

    def start_video(self):
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_label.setText("")

    def stop_video(self):
        self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText(self.initial_text)

    def update_image(self, cv_img):
        """
        方法级注释: 接收并渲染 OpenCV 图像.
        核心修改2: 动态判断图像通道数, 完美兼容 4 通道 RGBA 图像渲染.
        """
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w

        if ch == 4:
            # 如果底层传来了带透明通道的 BGRA 图像
            rgba_image = cv.cvtColor(cv_img, cv.COLOR_BGRA2RGBA)
            qt_image = QImage(
                rgba_image.data, w, h, bytes_per_line, QImage.Format_RGBA8888
            )
        else:
            # 如果是普通的 3 通道图像
            rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def on_params_updated(self, params_dict: dict):
        if "bg_image_path" in params_dict:
            new_path = params_dict["bg_image_path"]
            if new_path != self.bg_image_path:
                self.bg_image_path = new_path
                self.central_widget.set_background_image(self.bg_image_path)

        if "bg_opacity" in params_dict:
            self.central_widget.set_background_opacity(params_dict["bg_opacity"])

        self.thread.update_parameters(params_dict)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    config = load_config()
    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())
