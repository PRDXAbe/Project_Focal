"""Config loading. The YAML is kept as plain nested dicts.

Everything about the platform (sensor mounting, body box, footprint, height, topics, Nav2 footprint) lives in a
robot profile, configs/robots/<name>.yaml, merged over the main config. The profile is chosen by the `robot`
argument, else $ADAPTMAP_ROBOT, else the main config's `robot:` key.
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "adaptmap.yaml"
ROBOTS = ROOT / "configs" / "robots"


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_robot(name):
    with open(ROBOTS / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_config(path=None, robot=None):
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    name = robot or os.environ.get("ADAPTMAP_ROBOT") or cfg.get("robot", "rover")
    prof = load_robot(name)
    for section in ("preprocess", "terrain"):   # the profile's values for these sections win
        _merge(cfg.setdefault(section, {}), prof.pop(section, {}))
    cfg["robot"] = dict(prof, name=name)       # lidar_offset, topics, nav2 footprint, ...
    for section in ("model", "model_real"):
        for key in ("config", "checkpoint"):
            if section in cfg and key in cfg[section]:
                p = Path(cfg[section][key])
                if not p.is_absolute():
                    cfg[section][key] = str(ROOT / p)
    return cfg
