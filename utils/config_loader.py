"""该模块负责加载 YAML 格式的配置文件.

支持 UTF-8 编码读取，防止 Windows 环境下的编码冲突.
"""

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """加载并解析 YAML 配置文件.

    Args:
        config_path: 配置文件路径，默认为 "config.yaml".

    Returns:
        解析后的配置字典.

    Raises:
        FileNotFoundError: 如果路径不存在.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: 配置文件 {config_path} 未找到.")
        raise
