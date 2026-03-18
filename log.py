import logging
from turtle import setup
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def load_config(config_path="config.yaml"):
    """加载 yaml 配置文件"""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件{config_path}不存在")
        raise
    except yaml.YAMLError as e:
        print(f"解析配置文件{config_path}出错: {e}")
        raise


def setup_logging(config_path="config.yaml"):
    """创建日志记录器函数"""
    # 加载配置
    config = load_config(config_path)
    # 获取日志配置
    log_config = config.get("logging")

    log_level = log_config.get("level")
    log_path = log_config.get("path")
    log_format = log_config.get("format")
    log_encoding = log_config.get("encoding")
    logging.basicConfig(
        level=log_level,
        format=log_format,
        filename=log_path,
        handlers=[
            logging.FileHandler(log_path, encoding=log_encoding),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()

