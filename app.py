"""该模块是整个应用程序的启动入口.

负责加载全局配置，初始化 GUI 事件循环，并启动主窗口.
"""
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from utils.config_loader import load_config
from utils.logger import logger


def run_app():
    """初始化并运行应用."""
    try:
        # 1. 加载配置
        config = load_config("config.yaml")
        logger.info("配置文件加载成功.")

        # 2. 启动 Qt
        app = QApplication(sys.argv)
        
        # 3. 实例化并展示主窗口
        window = MainWindow(config)
        window.show()
        
        logger.info("主窗口已启动，进入事件循环.")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"程序运行发生致命错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_app()