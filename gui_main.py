"""
本文件是 GUI 应用的全新主入口点, 负责全局界面的排版布局与模块组装.
包含类: MainWindow.
"""

import sys
import cv2 as cv
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

# 导入配置加载工具与核心组件
from src.tools.load_config import load_config
from src.gui_components import VideoThread, ParameterPanel


class MainWindow(QMainWindow):
    """
    MainWindow 类负责整合 UI 模块与逻辑处理线程.
    核心逻辑为读取配置, 实例化组件并完成页面布局.
    """

    def __init__(self, config: dict):
        """
        构建主窗口.
        """
        super().__init__()
        gui_config = config["gui"]

        self.setWindowTitle(gui_config["window_title"])
        self.resize(gui_config["window_width"], gui_config["window_height"])

        self.thread = VideoThread(config)

        self.init_ui(gui_config, config)

    def init_ui(self, gui_config: dict, full_config: dict):
        """
        完成界面的模块化拼装.
        """
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_h_layout = QHBoxLayout(self.central_widget)

        # 布局左侧: 视频区
        self.left_layout = QVBoxLayout()
        self.main_h_layout.addLayout(self.left_layout, 4)

        self.video_group = QGroupBox("实时监控 (带全套美颜处理)")
        self.video_layout = QVBoxLayout()
        self.video_group.setLayout(self.video_layout)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText(gui_config["initial_text"])
        self.video_label.setMinimumSize(QSize(640, 480))
        self.video_layout.addWidget(self.video_label)
        self.left_layout.addWidget(self.video_group)

        # 布局左侧: 按钮区
        self.btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.start_video)
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)

        self.btn_layout.addWidget(self.start_btn)
        self.btn_layout.addWidget(self.stop_btn)
        self.left_layout.addLayout(self.btn_layout)

        # 布局右侧: 参数区
        self.right_layout = QVBoxLayout()
        self.main_h_layout.addLayout(self.right_layout, 1)

        self.param_group = QGroupBox("控制台")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)

        # 挂载参数面板
        self.param_panel = ParameterPanel(full_config)
        self.param_panel.sig_parameters_changed.connect(self.on_params_updated)
        self.param_layout.addWidget(self.param_panel)

        self.right_layout.addWidget(self.param_group)
        self.right_layout.addStretch()

    def start_video(self):
        """
        启动检测线程.
        """
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_video(self):
        """
        停止检测线程.
        """
        self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setText("Camera stopped.")

    def update_image(self, cv_img):
        """
        更新界面图像显示.
        """
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
        接收面板信号并更新底层配置.
        """
        self.thread.update_parameters(params_dict)

    def closeEvent(self, event):
        """
        处理窗口关闭事件.
        """
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    # 调用现有的工具函数安全加载配置
    config = load_config()

    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())
