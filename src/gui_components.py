"""
本文件包含 GUI 应用的核心交互组件与后台处理线程.
包含类: VideoThread, ParameterPanel.
"""

import cv2 as cv
from networkx import draw
import numpy as np
from PyQt5.QtWidgets import QWidget, QFormLayout, QComboBox, QSlider, QLabel, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt


from src.face_detect import FaceMeshDetector
from src.tools.open import open_camera
from src.filters import apply_beauty_filters


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, config: dict):
        """
        初始化线程并加载初始算法参数.
        """
        super().__init__()
        self._run_flag = True
        self.model_path = config["face-mesh"]["model_path"]
        self.max_faces = config["face-mesh"]["initial_max_faces"]
        self.draw_mode = config["face-mesh"].get("draw_mode", "mesh")
        self.draw_on_left = config["face-mesh"].get("draw_on_left", True)

        self.saturation = config["beauty"]["initial_saturation"]
        self.sharpness = config["beauty"]["initial_sharpness"]
        self.smoothing = config["beauty"]["initial_smoothing"]
        self.brighten = config["beauty"]["initial_brighten"]

    def update_parameters(self, params_dict: dict):
        """
        统一接收并更新来自前端面板的参数字典.
        """
        if "max_faces" in params_dict:
            self.max_faces = params_dict["max_faces"]
        if "draw_mode" in params_dict:
            self.draw_mode = params_dict["draw_mode"]
        if "draw_on_left" in params_dict:
            self.draw_on_left = params_dict["draw_on_left"]
        if "saturation" in params_dict:
            self.saturation = params_dict["saturation"]
        if "sharpness" in params_dict:
            self.sharpness = params_dict["sharpness"]
        if "smoothing" in params_dict:
            self.smoothing = params_dict["smoothing"]
        if "brighten" in params_dict:
            self.brighten = params_dict["brighten"]

    def run(self):
        """
        线程的主循环, 执行图像捕获, 美颜处理与模型推理.
        """
        capture = open_camera()
        if not capture.isOpened():
            return

        with FaceMeshDetector(
            model_path=self.model_path, max_faces=self.max_faces
        ) as detector:
            while self._run_flag:
                status, frame = capture.read()
                if status:
                    frame = cv.flip(frame, 1)

                    beauty_frame = apply_beauty_filters(
                        image=frame,
                        saturation=self.saturation,
                        sharpness=self.sharpness,
                        smoothing=self.smoothing,
                        brighten=self.brighten,
                    )

                    # 核心逻辑: 将动态的连线状态参数传入底层检测算法
                    processed_frame, skeleton_img, _ = detector.find_face_mesh(
                        frame=beauty_frame,
                        draw=True,
                        draw_mode=self.draw_mode,
                        draw_on_left=self.draw_on_left,
                    )

                    dst = detector.img_combine(processed_frame, skeleton_img)
                    self.change_pixmap_signal.emit(dst)

        capture.release()

    def stop(self):
        """
        安全停止线程.
        """
        self._run_flag = False
        self.wait()


class ParameterPanel(QWidget):
    """
    ParameterPanel 类封装了右侧所有的参数控制项.
    向外暴露包含所有参数状态的字典信号.
    """

    sig_parameters_changed = pyqtSignal(dict)

    def __init__(self, config: dict):
        """构建包含多项控制的表单布局."""
        super().__init__()
        self.layout = QFormLayout(self)

        self.max_faces_combo = QComboBox()
        self.max_faces_combo.addItems(["1", "2", "3", "4", "5"])
        self.max_faces_combo.setCurrentText(
            str(config["face-mesh"]["initial_max_faces"])
        )
        self.max_faces_combo.currentTextChanged.connect(self.on_parameter_changed)

        self.draw_mode_combo = QComboBox()
        self.draw_mode_combo.addItems(["绘制面部网络连接", "在人脸上绘制特征点"])
        init_mode = config["face-mesh"].get("draw_mode", "mesh")
        if init_mode == "points":
            self.draw_mode_combo.setCurrentText("在人脸上绘制特征点")
        else:
            self.draw_mode_combo.setCurrentText("绘制面部网络连接")
        self.draw_mode_combo.currentTextChanged.connect(self.on_parameter_changed)

        self.draw_on_left_checkbox = QCheckBox("在左侧(原图)上显示绘制内容")
        self.draw_on_left_checkbox.setChecked(
            config["face-mesh"].get("draw_on_left", True)
        )
        self.draw_on_left_checkbox.stateChanged.connect(self.on_parameter_changed)
        # 2. 饱和度调节滑块
        self.sat_slider = QSlider(Qt.Horizontal)
        self.sat_slider.setRange(0, 20)
        self.sat_slider.setValue(int(config["beauty"]["initial_saturation"] * 10))
        self.sat_label = QLabel(f"饱和度: {config['beauty']['initial_saturation']}")
        self.sat_slider.valueChanged.connect(self.update_sat_label_and_emit)

        # 3. 锐化调节滑块
        self.sharp_slider = QSlider(Qt.Horizontal)
        self.sharp_slider.setRange(0, 10)
        self.sharp_slider.setValue(int(config["beauty"]["initial_sharpness"] * 10))
        self.sharp_label = QLabel(f"锐化强度: {config['beauty']['initial_sharpness']}")
        self.sharp_slider.valueChanged.connect(self.update_sharp_label_and_emit)

        # 4. 磨皮调节滑块
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(config["beauty"]["initial_smoothing"])
        self.smooth_label = QLabel(f"磨皮强度: {config['beauty']['initial_smoothing']}")
        self.smooth_slider.valueChanged.connect(self.update_smooth_label_and_emit)

        # 5. 美白调节滑块
        self.bright_slider = QSlider(Qt.Horizontal)
        self.bright_slider.setRange(5, 10)
        self.bright_slider.setValue(int(config["beauty"]["initial_brighten"] * 10))
        self.bright_label = QLabel(
            f"美白程度: {11 - int(config['beauty']['initial_brighten'] * 10)}"
        )
        self.bright_slider.valueChanged.connect(self.update_bright_label_and_emit)

        self.layout.addRow("最大识别脸数:", self.max_faces_combo)
        self.layout.addRow("绘制模式选择:", self.draw_mode_combo)
        self.layout.addRow(self.draw_on_left_checkbox)
        self.layout.addRow(self.sat_label, self.sat_slider)
        self.layout.addRow(self.sharp_label, self.sharp_slider)
        self.layout.addRow(self.smooth_label, self.smooth_slider)
        self.layout.addRow(self.bright_label, self.bright_slider)

    def update_sat_label_and_emit(self, value: int):
        """
        更新饱和度标签并触发数据收集.
        """
        self.sat_label.setText(f"饱和度: {value / 10.0}")
        self.on_parameter_changed()

    def update_sharp_label_and_emit(self, value: int):
        """
        更新锐化标签并触发数据收集.
        """
        self.sharp_label.setText(f"锐化强度: {value / 10.0}")
        self.on_parameter_changed()

    def update_smooth_label_and_emit(self, value: int):
        """
        更新磨皮标签并触发数据收集.
        """
        self.smooth_label.setText(f"磨皮强度: {value}")
        self.on_parameter_changed()

    def update_bright_label_and_emit(self, value: int):
        """
        更新美白标签并触发数据收集.
        """
        display_level = 11 - value
        self.bright_label.setText(f"美白程度: {display_level}")
        self.on_parameter_changed()

    def on_parameter_changed(self):
        """
        方法级注释: 集中收集所有组件的数值, 打包为字典发送.
        """
        # 将中文选项映射为内部状态标识符
        mode_text = self.draw_mode_combo.currentText()
        draw_mode = "points" if mode_text == "在人脸上绘制特征点" else "mesh"

        params = {
            "max_faces": int(self.max_faces_combo.currentText()),
            "draw_mode": draw_mode,
            "draw_on_left": self.draw_on_left_checkbox.isChecked(),
            "saturation": self.sat_slider.value() / 10.0,
            "sharpness": self.sharp_slider.value() / 10.0,
            "smoothing": self.smooth_slider.value(),
            "brighten": self.bright_slider.value() / 10.0,
        }
        self.sig_parameters_changed.emit(params)
