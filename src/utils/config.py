from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    with open(file=config_path) as f:
        config = yaml.safe_load(f)
    return config