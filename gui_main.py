"""
文件级注释: 本文件是 GUI 应用的主入口点.
集成了安全的 BackgroundWidget 容器, 以及 QGraphicsOpacityEffect 视频透视特效.
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
    QGraphicsOpacityEffect,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize

from src.tools.load_config import load_config
from src.gui_components import VideoThread, ParameterPanel
from src.ui_style import BackgroundWidget


class MainWindow(QMainWindow):
    """
    类级注释: MainWindow 负责拦截样式参数并应用到前端组件.
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

    def init_ui(self, gui_config: dict, full_config: dict):
        # 核心逻辑: 注入我们封装的带安全检测的底层容器
        self.central_widget = BackgroundWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.set_background_image(self.bg_image_path)
        self.central_widget.set_background_opacity(gui_config.get("bg_opacity", 0.6))

        self.main_h_layout = QHBoxLayout(self.central_widget)

        self.left_layout = QVBoxLayout()
        self.main_h_layout.addLayout(self.left_layout, 4)

        self.video_group = QGroupBox("实时监控 (带全套美颜处理)")
        self.video_layout = QVBoxLayout()
        self.video_group.setLayout(self.video_layout)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText(self.initial_text)
        self.video_label.setMinimumSize(QSize(640, 480))

        # 核心魔法逻辑: 给视频控件挂载透明度特效组件
        # 这样即使视频正在播放, 也能透出背后的全局图片
        self.video_opacity_effect = QGraphicsOpacityEffect(self.video_label)
        self.video_label.setGraphicsEffect(self.video_opacity_effect)
        self.video_opacity_effect.setOpacity(gui_config.get("video_opacity", 0.85))

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

        self.right_layout = QVBoxLayout()
        self.main_h_layout.addLayout(self.right_layout, 1)

        self.param_group = QGroupBox("控制台")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)

        self.param_panel = ParameterPanel(full_config)
        self.param_panel.sig_parameters_changed.connect(self.on_params_updated)
        self.param_layout.addWidget(self.param_panel)

        self.right_layout.addWidget(self.param_group)
        self.right_layout.addStretch()

    def start_video(self):
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # 清除待机占位文字
        self.video_label.setText("")

    def stop_video(self):
        self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # 恢复占位文字并清空视频画面, 以免残留最后一帧
        self.video_label.clear()
        self.video_label.setText(self.initial_text)

    def update_image(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def on_params_updated(self, params_dict: dict):
        """
        方法级注释: 拦截 UI 样式参数并实时渲染, 将业务参数放行给子线程.
        """
        # 拦截并更新背景图片路径
        if "bg_image_path" in params_dict:
            new_path = params_dict["bg_image_path"]
            if new_path != self.bg_image_path:
                self.bg_image_path = new_path
                self.central_widget.set_background_image(self.bg_image_path)

        # 拦截并更新背景图片的透明度
        if "bg_opacity" in params_dict:
            self.central_widget.set_background_opacity(params_dict["bg_opacity"])

        # 拦截并更新视频画面的透视度
        if "video_opacity" in params_dict:
            self.video_opacity_effect.setOpacity(params_dict["video_opacity"])

        # 将与面部算法及美颜相关的参数传递给视频线程
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
