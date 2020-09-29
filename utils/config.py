import yaml

from pathlib import Path
from typing import Union


def load_config(path: Union[Path, str]):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config
