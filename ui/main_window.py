"""
该模块定义主窗口 UI.

负责整体布局, QSS 样式注入及信号的分发.
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
    """UI 总控窗口"""

    def __init__(self, config: dict):
        """初始化主窗口.

        Args:
            config: 全局配置字典.
        """
        super().__init__()
        self.config = config
        self.setWindowTitle(config["gui"]["window_title"])
        self.resize(config["gui"]["window_width"], config["gui"]["window_height"])

        # 初始化后台线程
        self.thread = VideoThread(config)
        self._init_ui()
        self._apply_qss()

    def _init_ui(self):
        """构建自适应背景容器与双面板布局."""
        # 1. 注入背景容器
        self.bg_container = BackgroundWidget(self)
        self.setCentralWidget(self.bg_container)
        self.bg_container.set_background_image(
            self.config["gui"].get("bg_image_path", "")
        )
        self.bg_container.set_background_opacity(
            self.config["gui"].get("bg_opacity", 0.6)
        )

        main_layout = QHBoxLayout(self.bg_container)

        # 2. 左侧视频展示区
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
        self.stop_btn.setEnabled(False)
        btn_h.addWidget(self.start_btn)
        btn_h.addWidget(self.stop_btn)
        left_v.addLayout(btn_h)

        main_layout.addWidget(left_box, 4)

        # 3. 右侧参数控制面板
        right_box = QGroupBox("控制台")
        right_box.setObjectName("paramGroup")
        right_v = QVBoxLayout(right_box)

        self.panel = ParameterPanel(self.config)
        # 绑定信号: 当 UI 参数改变时触发 _on_params
        self.panel.sig_parameters_changed.connect(self._on_params)
        right_v.addWidget(self.panel)
        right_v.addStretch()

        main_layout.addWidget(right_box, 1)

    def _apply_qss(self):
        """注入样式表, 确保背景透视与文字对比度."""
        qss = """
        /* 基础 GroupBox 样式 */
        QGroupBox { 
            border: 1px solid rgba(255, 255, 255, 60); 
            border-radius: 8px; 
            color: #FFFFFF; 
            font-weight: bold; 
            font-size: 24px; 
            margin-top: 25px; 
        }
        
        /* 核心修改: 精准控制标题文字的位置, 让它居中卡在边框上 */
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center; /* 绝对顶部居中对齐 */
            padding: 0 16px; /* 左右增加内边距, 打断边框线, 让文字两端有呼吸感 */
            color: #FFFFFF;
        }

        QGroupBox#paramGroup { background-color: rgba(20, 20, 20, 180); }
        /* 关键: 确保视频区域 QLabel 背景透明, 以透出底层的背景图 */
        QGroupBox#videoGroup QLabel { background-color: transparent; }
        
        QLabel, QCheckBox { color: #FFFFFF; font-size: 16px; }
        QPushButton, QComboBox { 
            background-color: rgba(60, 60, 60, 200); 
            color: #FFFFFF; 
            padding: 9px; 
            border-radius: 4px; 
            font-size: 16px;
        }
        QPushButton:hover { background-color: rgba(90, 90, 90, 220); }
        QPushButton:disabled { color: #888888; background-color: rgba(40, 40, 40, 100); }
        """
        self.setStyleSheet(qss)

    def _on_params(self, data: dict):
        """接收 UI 信号并分发参数.

        Args:
            data: 包含所有 UI 控件当前状态的字典.
        """
        # 更新 UI 层的背景显示
        self.bg_container.set_background_image(data["bg_image_path"])
        self.bg_container.set_background_opacity(data["bg_opacity"])

        # 传递业务参数给线程
        self.thread.update_parameters(data)

    def _start(self):
        """开启摄像头线程."""
        self.thread.frame_signal.connect(self._update_img)
        self.thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.vid_lbl.setText("")

    def _stop(self):
        """停止线程并重置 UI 占位."""
        self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.vid_lbl.clear()
        self.vid_lbl.setText("检测已停止")

    def _update_img(self, img):
        """渲染后台传回的 4 通道图像."""
        h, w, ch = img.shape
        # 针对 4 通道 BGRA 图像进行特殊处理
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
        """拦截窗口关闭事件, 确保资源安全释放."""
        self._stop()
        event.accept()
