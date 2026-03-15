import os
from pathlib import Path

import yaml

LOCAL_CONFIG_PATH = Path.cwd() / ".kiui.yaml"
HOME_CONFIG_PATH = Path.home() / ".kiui.yaml"

def _load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        if LOCAL_CONFIG_PATH.exists():
            config_path = LOCAL_CONFIG_PATH 
        elif HOME_CONFIG_PATH.exists():
            config_path = HOME_CONFIG_PATH
        else:
            return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
conf = _load_config()