"""
该模块提供 GUI 的自定义交互组件.

包含实现背景渲染的 BackgroundWidget 和交互控制的 ParameterPanel.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QComboBox, QSlider, QLabel, 
    QCheckBox, QPushButton, QFileDialog, QHBoxLayout, QMessageBox
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
        """重写绘图事件: 先绘制防穿帮底色, 再绘制半透明用户背景图."""
        painter = QPainter(self)
        # 绘制默认灰白底色
        painter.fillRect(self.rect(), Qt.lightGray)

        if self.bg_pixmap:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.setOpacity(self.opacity)
            # 缩放图片以铺满容器
            painter.drawPixmap(self.rect(), self.bg_pixmap)


class ParameterPanel(QWidget):
    """侧边控制面板, 封装所有检测与美颜控制项."""

    sig_parameters_changed = pyqtSignal(dict)
    sig_source_file_changed = pyqtSignal(str, str)

    def __init__(self, config: dict):
        """初始化面板.

        Args:
            config: 全局配置字典.
        """
        super().__init__()
        self.config = config
        self.bg_path = config["gui"].get("bg_image_path", "")
        self.image_src_path = ""
        self.video_src_path = ""
        self._init_ui()

    def _init_ui(self):
        """构建控件布局."""
        layout = QFormLayout(self)

        # --- 1. 检测配置 ---

        # 数据源模式选择
        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItems(["实时摄像头", "静态图片文件", "本地视频文件"])
        self.source_mode_combo.currentIndexChanged.connect(self._emit)
        
        # 图片导入按钮
        self.import_img_btn = QPushButton("导入图片...")
        self.import_img_btn.clicked.connect(self._pick_image_source)
        
        # 视频导入按钮
        self.import_vid_btn = QPushButton("导入视频...")
        self.import_vid_btn.clicked.connect(self._pick_video_source)

        self.faces_combo = QComboBox()
        self.faces_combo.addItems(["1", "2", "3", "4", "5"])
        init_faces = self.config["face-mesh"].get("initial_max_faces", 2)
        self.faces_combo.setCurrentText(str(init_faces))
        self.faces_combo.currentTextChanged.connect(self._emit)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["绘制面部网络连接", "在人脸上绘制特征点"])
        self.mode_combo.currentTextChanged.connect(self._emit)

        self.left_cb = QCheckBox("左侧原图显示绘制内容")
        self.left_cb.setChecked(self.config["face-mesh"].get("draw_on_left", True))
        self.left_cb.stateChanged.connect(self._emit)

        # --- 2. 全局背景控制 ---
        self.bg_btn = QPushButton("浏览选图...")
        self.bg_btn.clicked.connect(self._pick_bg_file)
        init_bg_op = int(self.config["gui"].get("bg_opacity", 0.6) * 100)
        self.bg_op_sld, self.bg_op_lbl = self._create_slider(
            0, 100, init_bg_op, f"背景透明度: {init_bg_op}%"
        )

        # --- 3. 美颜控制组 ---
        self.smooth_sld, self.smooth_lbl = self._create_slider(0, 100, 0, "磨皮强度: 0")
        self.bright_sld, self.bright_lbl = self._create_slider(
            5, 15, 10, "美白程度: 1.0"
        )
        self.sat_sld, self.sat_lbl = self._create_slider(0, 20, 10, "饱和度: 1.0")
        self.sharp_sld, self.sharp_lbl = self._create_slider(0, 10, 0, "锐化强度: 0.0")

        # 将所有控件加入表单
        layout.addRow("数据源模式:", self.source_mode_combo)
        layout.addRow("图片检测源:", self.import_img_btn)
        layout.addRow("视频检测源:", self.import_vid_btn)
        layout.addRow("最大识别脸数:", self.faces_combo)
        layout.addRow("绘制模式选择:", self.mode_combo)
        layout.addRow(self.left_cb)
        layout.addRow("全局 UI 背景:", self.bg_btn)
        layout.addRow(self.bg_op_lbl, self.bg_op_sld)
        layout.addRow(self.smooth_lbl, self.smooth_sld)
        layout.addRow(self.bright_lbl, self.bright_sld)
        layout.addRow(self.sat_lbl, self.sat_sld)
        layout.addRow(self.sharp_lbl, self.sharp_sld)

    def _create_slider(self, min_v, max_v, init_v, label_text):
        """辅助方法: 创建滑块与文字标签.

        Args:
            min_v: 滑块最小值.
            max_v: 滑块最大值.
            init_v: 初始值.
            label_text: 标签的初始文本.

        Returns:
            创建好的 QSlider 和 QLabel 实例.
        """
        sld = QSlider(Qt.Horizontal)
        sld.setRange(min_v, max_v)
        sld.setValue(init_v)
        sld.valueChanged.connect(self._emit)
        lbl = QLabel(label_text)
        return sld, lbl

    def _pick_bg_file(self):
        """导入 UI 背景图片."""
        path, _ = QFileDialog.getOpenFileName(self, "选择背景图片", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.bg_path = path
            self._emit()

    def _pick_image_source(self):
        """导入静态图片检测源."""
        path, _ = QFileDialog.getOpenFileName(self, "选择待检测图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.image_src_path = path
            QMessageBox.information(self, "导入成功", f"已选择图片检测源:\n{os.path.basename(path)}")
            # 切换模式并通知
            self.source_mode_combo.setCurrentText("静态图片文件")
            self.sig_source_file_changed.emit("image", path)
            self._emit()

    def _pick_video_source(self):
        """导入本地视频检测源."""
        path, _ = QFileDialog.getOpenFileName(self, "选择待检测视频", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.video_src_path = path
            QMessageBox.information(self, "导入成功", f"已选择视频检测源:\n{os.path.basename(path)}")
            # 切换模式并通知
            self.source_mode_combo.setCurrentText("本地视频文件")
            self.sig_source_file_changed.emit("video", path)
            self._emit()

    def _emit(self):
        """收集参数并发送信号."""
        # 实时更新所有标签文字显示
        self.bg_op_lbl.setText(f"背景透明度: {self.bg_op_sld.value()}%")
        self.smooth_lbl.setText(f"磨皮强度: {self.smooth_sld.value()}")
        self.bright_lbl.setText(f"美白程度: {self.bright_sld.value() / 10.0}")
        self.sat_lbl.setText(f"饱和度: {self.sat_sld.value() / 10.0}")
        self.sharp_lbl.setText(f"锐化强度: {self.sharp_sld.value() / 10.0}")

        data = {
            "max_faces": int(self.faces_combo.currentText()),
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
