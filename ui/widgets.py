"""该模块提供 GUI 的自定义交互组件.

包含实现背景渲染的 BackgroundWidget 和交互控制的 ParameterPanel.
"""

import os
from PyQt5.QtWidgets import (
    QWidget,
    QFormLayout,
    QComboBox,
    QSlider,
    QLabel,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal


class BackgroundWidget(QWidget):
    """支持半透明背景图片自适应渲染的容器.

    Attributes:
        bg_pixmap: 缓存的背景图片像素图.
        opacity: 背景图片的透明度.
    """

    def __init__(self, parent=None):
        """初始化容器部件."""
        super().__init__(parent)
        self.bg_pixmap = None
        self.opacity = 1.0

    def set_background_image(self, path: str):
        """安全加载背景图片并刷新界面.

        Args:
            path: 图像文件的磁盘路径.
        """
        if path and os.path.isfile(path):
            pix = QPixmap(path)
            self.bg_pixmap = pix if not pix.isNull() else None
        else:
            self.bg_pixmap = None
        self.update()

    def set_background_opacity(self, opacity: float):
        """更新背景图片的透明度数值.

        Args:
            opacity: 透明度比例 (0.0 到 1.0).
        """
        self.opacity = opacity
        self.update()

    def paintEvent(self, event):
        """重写绘图事件：先绘制防穿帮底色，再绘制半透明用户背景图."""
        painter = QPainter(self)
        # 绘制默认灰白底色
        painter.fillRect(self.rect(), Qt.lightGray)

        if self.bg_pixmap:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.setOpacity(self.opacity)
            # 缩放图片以铺满容器
            painter.drawPixmap(self.rect(), self.bg_pixmap)


class ParameterPanel(QWidget):
    """侧边控制面板，封装所有滑块与模式选择控件."""

    sig_parameters_changed = pyqtSignal(dict)

    def __init__(self, config: dict):
        """初始化面板并设置初始 UI 状态.

        Args:
            config: 全局配置字典.
        """
        super().__init__()
        self.config = config
        self.bg_path = config["gui"].get("bg_image_path", "")
        self._init_ui()

    def _init_ui(self):
        """构建控件布局."""
        layout = QFormLayout(self)

        # 1. 检测模式与开关
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["绘制面部网络连接", "在人脸上绘制特征点"])
        self.mode_combo.currentTextChanged.connect(self._emit)

        self.left_cb = QCheckBox("左侧原图显示绘制内容")
        self.left_cb.setChecked(True)
        self.left_cb.stateChanged.connect(self._emit)

        # 2. 全局背景控制
        self.bg_btn = QPushButton("浏览选图...")
        self.bg_btn.clicked.connect(self._pick_file)

        self.bg_op_sld = QSlider(Qt.Horizontal)
        self.bg_op_sld.setRange(0, 100)
        self.bg_op_sld.setValue(60)
        self.bg_op_sld.valueChanged.connect(self._emit)

        # 3. 美颜滑块组 (统一使用 _sld 后缀)
        self.smooth_sld, self.smooth_lbl = self._create_slider(0, 100, 0, "磨皮强度: 0")
        self.bright_sld, self.bright_lbl = self._create_slider(
            5, 15, 10, "美白程度: 1.0"
        )
        self.sat_sld, self.sat_lbl = self._create_slider(0, 20, 10, "饱和度: 1.0")
        self.sharp_sld, self.sharp_lbl = self._create_slider(0, 10, 0, "锐化强度: 0.0")

        # 将控件添加到表单
        layout.addRow("绘制模式:", self.mode_combo)
        layout.addRow(self.left_cb)
        layout.addRow("界面背景:", self.bg_btn)
        layout.addRow("背景透明度:", self.bg_op_sld)

        # 修复处：确保引用的变量名与上方定义的一致
        layout.addRow(self.smooth_lbl, self.smooth_sld)
        layout.addRow(self.bright_lbl, self.bright_sld)
        layout.addRow(self.sat_lbl, self.sat_sld)
        layout.addRow(self.sharp_lbl, self.sharp_sld)

    def _create_slider(self, min_v, max_v, init_v, label_text):
        """创建滑块与标签的辅助方法."""
        sld = QSlider(Qt.Horizontal)
        sld.setRange(min_v, max_v)
        sld.setValue(init_v)
        sld.valueChanged.connect(self._emit)
        lbl = QLabel(label_text)
        return sld, lbl

    def _pick_file(self):
        """打开文件对话框选择图片."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "", "Images (*.png *.jpg *.bmp)"
        )
        if path:
            self.bg_path = path
            self._emit()

    def _emit(self):
        """收集所有参数并发送信号，同时实时更新标签文字."""
        # 实时更新标签数值显示
        self.smooth_lbl.setText(f"磨皮强度: {self.smooth_sld.value()}")
        self.bright_lbl.setText(f"美白程度: {self.bright_sld.value() / 10.0}")
        self.sat_lbl.setText(f"饱和度: {self.sat_sld.value() / 10.0}")
        self.sharp_lbl.setText(f"锐化强度: {self.sharp_sld.value() / 10.0}")

        # 组织发送到线程的参数字典
        data = {
            "draw_mode": (
                "points"
                if self.mode_combo.currentText() == "在人脸上绘制特征点"
                else "mesh"
            ),
            "draw_on_left": self.left_cb.isChecked(),
            "bg_image_path": self.bg_path,
            "bg_opacity": self.bg_op_sld.value() / 100.0,
            "smoothing": self.smooth_sld.value(),
            "brighten": self.bright_sld.value() / 10.0,
            "saturation": self.sat_sld.value() / 10.0,
            "sharpness": self.sharp_sld.value() / 10.0,
        }
        self.sig_parameters_changed.emit(data)
