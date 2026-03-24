"""该模块提供自定义的 PyQt5 基础组件.

包含实现背景渲染的 BackgroundWidget 和交互控制的 ParameterPanel.
"""
import os
from PyQt5.QtWidgets import QWidget, QFormLayout, QComboBox, QSlider, QLabel, QCheckBox, QPushButton, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal


class BackgroundWidget(QWidget):
    """支持半透明背景图片自适应渲染的容器."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_pixmap = None
        self.opacity = 1.0

    def set_background_image(self, path: str):
        """安全加载背景图片."""
        if path and os.path.isfile(path):
            pix = QPixmap(path)
            self.bg_pixmap = pix if not pix.isNull() else None
        else:
            self.bg_pixmap = None
        self.update()

    def set_background_opacity(self, opacity: float):
        """更新透明度."""
        self.opacity = opacity
        self.update()

    def paintEvent(self, event):
        """绘图事件：先画底色，再画半透明图."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.lightGray)
        if self.bg_pixmap:
            painter.setOpacity(self.opacity)
            painter.drawPixmap(self.rect(), self.bg_pixmap)


class ParameterPanel(QWidget):
    """侧边控制面板，封装所有滑块与开关."""
    sig_parameters_changed = pyqtSignal(dict)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._init_ui()

    def _init_ui(self):
        """初始化控件布局."""
        layout = QFormLayout(self)
        
        # 1. 核心控制
        self.faces_combo = QComboBox()
        self.faces_combo.addItems(["1", "2", "3", "4", "5"])
        self.faces_combo.setCurrentText(str(self.config['face-mesh']['initial_max_faces']))
        self.faces_combo.currentTextChanged.connect(self._emit)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["绘制面部网络连接", "在人脸上绘制特征点"])
        self.mode_combo.currentTextChanged.connect(self._emit)

        self.left_cb = QCheckBox("左侧原图显示绘制内容")
        self.left_cb.setChecked(self.config['face-mesh']['draw_on_left'])
        self.left_cb.stateChanged.connect(self._emit)

        # 2. UI控制
        self.bg_btn = QPushButton("浏览选图...")
        self.bg_btn.clicked.connect(self._pick_file)
        self.bg_path = self.config['gui']['bg_image_path']
        
        self.opacity_sld = QSlider(Qt.Horizontal)
        self.opacity_sld.setRange(0, 100)
        self.opacity_sld.setValue(int(self.config['gui']['bg_opacity'] * 100))
        self.opacity_sld.valueChanged.connect(self._emit)

        # 3. 美颜控制 (示例一个)
        self.smooth_sld = QSlider(Qt.Horizontal)
        self.smooth_sld.setRange(0, 100)
        self.smooth_sld.valueChanged.connect(self._emit)

        layout.addRow("识别脸数:", self.faces_combo)
        layout.addRow("绘制模式:", self.mode_combo)
        layout.addRow(self.left_cb)
        layout.addRow("界面背景:", self.bg_btn)
        layout.addRow("背景透明度:", self.opacity_sld)
        layout.addRow("磨皮强度:", self.smooth_sld)

    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.bg_path = path
            self._emit()

    def _emit(self):
        """打包字典并发送."""
        data = {
            "max_faces": int(self.faces_combo.currentText()),
            "draw_mode": "points" if self.mode_combo.currentText() == "在人脸上绘制特征点" else "mesh",
            "draw_on_left": self.left_cb.isChecked(),
            "bg_image_path": self.bg_path,
            "bg_opacity": self.opacity_sld.value() / 100.0,
            "smoothing": self.smooth_sld.value()
        }
        self.sig_parameters_changed.emit(data)