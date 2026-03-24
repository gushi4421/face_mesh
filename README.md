# Face Mesh: 基于 MediaPipe 的面部检测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-red.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision_AI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-success.svg)

> 🎓 **计算机视觉与机器学习课程作业**
> 本项目不仅实现了前沿的 AI 算法调用，更严格遵循现代软件工程的设计模式。通过纯正的分层解耦架构（Core / UI / Utils），实现了极佳的代码可维护性与扩展性。

---

## 📋 项目简介

**Face Mesh Pro** 是一套集成了实时人脸特征点追踪与工业级图像后处理的桌面级应用程序。本项目摒弃了传统的“意大利面条式”脚本编写方式，将底层 AI 算法逻辑与前端 GUI 交互进行了彻底分离。

系统在前端提供了丝滑的参数控制面板与深色模式控制台，在后端则依托 MediaPipe 与 OpenCV 构建了低延迟、高精度的“全息渲染+动态美颜”管线，完美兼顾了 AI 算法的底层硬核实现与极佳的用户交互体验。

---

## ✨ 核心亮点与特性 (Highlights)

本项目在常规的 CV 算法基础上，在图形渲染与架构设计上进行了深度创新：

- **🏗️ 纯正的架构解耦 (MVC 思想)**
  全面抛弃脚本化编程，将系统严格划分为 `core` (算法模型层)、`ui` (表现交互层) 与 `utils` (通用工具层)。业务逻辑与界面渲染完全剥离，通过安全的线程通信机制（Signal-Slot）进行数据交换。
- **🎮 极客风 GUI 与动态视觉适配**
  基于 `PyQt5` 构建，原生实现了深色模式控制台。通过精细的 QSS 样式表深度定制组件，完美解决了高透 UI 下文字与复杂壁纸背景的对比度问题，提供沉浸式的操作体验。
- **🚀 低延迟 AI 视觉引擎**
  无缝接入 `MediaPipe Vision API`，利用其轻量级模型实现全天候、低延迟的 468 个人脸 3D 网格（Face Mesh）与高精度特征点实时追踪。
- **💎 创新级 4 通道 (RGBA) 全息渲染流**
  **独创的透明渲染管线：** 在 OpenCV 矩阵处理环节重写 Alpha 通道，实现左侧“实拍源视频”与右侧“骨架拓扑图”的对比呈现。未渲染面部网格的背景区域被设为完全透明，使 3D 面部拓扑仿佛“全息悬浮”于你的自定义桌面壁纸之上，极具赛博朋克视觉冲击力。
- **🎛️ 工业级手工美颜管线**
  不依赖黑盒封装，纯手工调用 OpenCV 基础算子重构了完整的美颜链路：
  - **磨皮**：基于双边滤波（Bilateral Filter）保留边缘的平滑处理。
  - **美白**：基于 Gamma 颜色空间的高光映射。
  - **锐化**：动态拉普拉斯算子（Laplacian）增强五官立体感。
  - **调色**：HSV 色彩空间动态饱和度与明度调节。

---

## 📁 目录结构与架构说明

本项目严格遵循 **MVC (Model-View-Controller)** 架构思想：
- **Model (`core/`)**：封装底层数据处理与 AI 算法。
- **View (`ui/`)**：负责界面排版与视觉呈现。
- **Controller (`app.py` & `ui/thread.py`)**：负责调度数据流与 UI 状态更新。

```text
face_mesh_pro/
├── app.py                 # [Controller] 程序唯一启动入口
├── config.yaml            # 全局配置文件 (包含 UI 样式和算法初始参数)
├── requirements.txt       # 项目依赖清单
├── models/                # 模型文件夹
│   └── face_landmarker.task # MediaPipe 官方预训练模型文件
├── core/                  # [Model] 核心算法与数据处理层
│   ├── detector.py        # 封装 MediaPipe 面部检测与 4 通道透明渲染逻辑
│   └── processor.py       # 封装 OpenCV 图像后处理与高阶美颜滤镜算法
├── ui/                    # [View] 界面呈现与交互层
│   ├── main_window.py     # 主窗口排版, QSS 样式注入及跨组件信号路由
│   ├── widgets.py         # 封装参数控制面板与自适应半透明背景容器
│   └── thread.py          # 后台视频处理线程, 衔接核心算法与前端 UI 的桥梁
└── utils/                 # 通用工具层
    ├── config_loader.py   # YAML 配置文件安全加载器
    ├── camera_utils.py    # 本地摄像头与硬件视频流控制工具
    └── logger.py          # 全局日志记录
```
## 🚀 快速开始
- **1.环境配置**
建议使用 Python 3.8 或更高版本的虚拟环境进行安装与测试。
```bash
# 克隆项目仓库
git clone https://github.com/gushi4421/face_mesh.git
cd face_mesh

# 安装核心依赖包
pip install -r requirements.txt
```
- **2.运行系统**
确保 models/face_landmarker.task 预训练模型文件已就位，随后在根目录下直接启动主入口程序：

``` bash
python app.py
```

- **3.使用说明**
  - 启动后点击 `开始检测`，系统将自动打开摄像头并显示实时视频流。
  - 可动态调整检测人脸数量，实时观察效果变化。
  - 自定义调整绘制模式（点、线）以获得不同的视觉效果。
  - 右侧控制面板提供了磨皮、锐化、美白、调色等美颜模式，调整后效果会实时反映在视频中。
  - 支持加载自定义的桌面壁纸作为背景，支持调整透明度以获得最佳视觉效果。

- **4.自定义参数设置**
系统支持通过 `config.yaml` 文件进行全局参数配置，包括但不限于：
- **5. 项目打包**

如果为了追求极致的用户体验，可以将项目打包成独立的可执行文件
``` bash
# 安装打包工具
pip install pyinstaller

# 执行打包命令 (-D 生成目录模式，-w 隐藏控制台，指定应用名称)
pyinstaller -D -w app.py --name "FaceMeshPro"
```
## 👤 作者

**gushi4421**

- GitHub: [@gushi4421](https://github.com/gushi4421)
- 身份：2024 届在读本科生(一位编程小登)

## 🙏 致谢
- 感谢所有开源贡献者的无私分享

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [Issue](https://github.com/gushi4421/face_mesh/issues)
- 发送邮件至：gushi4421@qq.com
  
⭐ 如果这个项目对你有帮助，欢迎 Star 支持！