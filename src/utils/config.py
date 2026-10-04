from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML experiment configuration from disk."""
    with open(file=config_path) as f:
        config = yaml.safe_load(f)
    return config