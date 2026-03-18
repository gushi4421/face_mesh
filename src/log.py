import logging
import sys
from pathlib import Path

from tools.load_config import load_config


def setup_logging():
    """创建日志记录器函数"""
    # 加载配置
    config = load_config()
    # 获取日志配置
    log_config = config.get("logging")

    log_level = log_config.get("level")
    log_path = log_config.get("path")
    log_format = log_config.get("format")
    log_encoding = log_config.get("encoding")

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, encoding=log_encoding),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()
