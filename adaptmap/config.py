"""Config loading. The YAML is kept as plain nested dicts."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "adaptmap.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for key in ("config", "checkpoint"):
        p = Path(cfg["model"][key])
        if not p.is_absolute():
            cfg["model"][key] = str(ROOT / p)
    return cfg
