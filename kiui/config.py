import os
import yaml

def _load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        # default to ~/.kiui.yaml
        config_path = os.path.join(os.path.expanduser('~'), '.kiui.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# load at init
conf = _load_config()