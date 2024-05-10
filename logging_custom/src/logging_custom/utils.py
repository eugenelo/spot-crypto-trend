import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: Path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    logging.config.dictConfig(config)
