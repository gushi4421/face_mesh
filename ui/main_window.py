"""该模块定义主窗口 UI.

负责整体布局、QSS 样式注入及信号的分发.
"""

from PyQt5.QtWidgets import (
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QSize
import cv2 as cv


from ui.widgets import BackgroundWidget, ParameterPanel
from ui.thread import VideoThread


class MainWindow(QMainWindow):
    """UI 总控窗口."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.setWindowTitle(config["gui"]["window_title"])
        self.resize(config["gui"]["window_width"], config["gui"]["window_height"])

        self.thread = VideoThread(config)
        self._init_ui()
        self._apply_qss()

    def _init_ui(self):
        # 1. 容器
        self.bg_container = BackgroundWidget(self)
        self.setCentralWidget(self.bg_container)
        self.bg_container.set_background_image(self.config["gui"]["bg_image_path"])

        main_layout = QHBoxLayout(self.bg_container)

        # 2. 左侧显示
        left_box = QGroupBox("实时监控")
        left_box.setObjectName("videoGroup")
        left_v = QVBoxLayout(left_box)
        self.vid_lbl = QLabel(self.config["gui"]["initial_text"])
        self.vid_lbl.setAlignment(Qt.AlignCenter)
        self.vid_lbl.setMinimumSize(QSize(640, 480))
        left_v.addWidget(self.vid_lbl)

        btn_h = QHBoxLayout()
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        btn_h.addWidget(self.start_btn)
        btn_h.addWidget(self.stop_btn)
        left_v.addLayout(btn_h)

        main_layout.addWidget(left_box, 4)

        # 3. 右侧面板
        right_box = QGroupBox("控制台")
        right_box.setObjectName("paramGroup")
        right_v = QVBoxLayout(right_box)
        self.panel = ParameterPanel(self.config)
        self.panel.sig_parameters_changed.connect(self._on_params)
        right_v.addWidget(self.panel)
        right_v.addStretch()

        main_layout.addWidget(right_box, 1)

    def _apply_qss(self):
        """注入高对比度 QSS 样式."""
        qss = """
        QGroupBox { border: 1px solid rgba(255,255,255,60); border-radius: 8px; color: white; font-weight: bold; font-size: 18px; margin-top: 20px; }
        QGroupBox#paramGroup { background-color: rgba(20,20,20,180); }
        QGroupBox#videoGroup QLabel { background-color: transparent; }
        QLabel, QCheckBox { color: white; font-size: 15px; }
        QPushButton { background-color: #444; color: white; padding: 8px; border-radius: 4px; }
        """
        self.setStyleSheet(qss)

    def _on_params(self, data: dict):
        """拦截 UI 样式，透传业务参数."""
        self.bg_container.set_background_image(data["bg_image_path"])
        self.bg_container.set_background_opacity(data["bg_opacity"])
        self.thread.update_params(data)

    def _start(self):
        self.thread.frame_signal.connect(self._update_img)
        self.thread.start()

    def _stop(self):
        self.thread.stop()
        self.vid_lbl.clear()
        self.vid_lbl.setText("检测已停止")

    def _update_img(self, img):
        h, w, ch = img.shape
        # 兼容 4 通道 RGBA 渲染
        fmt = QImage.Format_RGBA8888 if ch == 4 else QImage.Format_RGB888
        if ch == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        qimg = QImage(img.data, w, h, ch * w, fmt)
        self.vid_lbl.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.vid_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def closeEvent(self, event):
        self._stop()
        event.accept()
