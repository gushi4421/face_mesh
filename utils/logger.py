"""该模块初始化全局日志记录器.

根据配置文件的设定，将日志同步输出至文件和控制台.
"""

import logging
from utils.config_loader import load_config


def setup_logging() -> logging.Logger:
    """配置并返回日志记录器实例.

    Returns:
        配置好的 logging.Logger 对象.
    """
    config = load_config()
    log_config = config.get("logging", {})

    logging.basicConfig(
        level=log_config.get("level", "INFO"),
        format=log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_config.get("path", "app.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("FaceMeshApp")


logger = setup_logging()
