import os
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'log')
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
