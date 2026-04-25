from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
