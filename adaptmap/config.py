"""Config loading. The YAML is kept as plain nested dicts."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "adaptmap.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "model_real"):
        for key in ("config", "checkpoint"):
            if section in cfg and key in cfg[section]:
                p = Path(cfg[section][key])
                if not p.is_absolute():
                    cfg[section][key] = str(ROOT / p)
    return cfg
