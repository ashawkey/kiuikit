import os
from pathlib import Path

import yaml

CONFIG_PATH = Path.home() / ".kiui.yaml"

def _load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# load at init
conf = _load_config()