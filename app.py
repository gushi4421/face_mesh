"""
该模块是系统的启动入口.

负责加载配置、初始化 UI 组件并启动事件循环.
"""

import sys
from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow
from utils.config_loader import load_config
from utils.logger import logger


def main():
    """主启动函数."""
    try:
        config = load_config()
        app = QApplication(sys.argv)

        # 实例化重构后的主窗口
        main_win = MainWindow(config)
        main_win.show()

        logger.info("系统初始化完成, 进入主循环.")
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"程序启动失败: {e}")


if __name__ == "__main__":
    main()
