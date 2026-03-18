import yaml


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
