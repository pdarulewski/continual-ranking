import os
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'log')
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

WANDB_DIR = os.path.join(ROOT_DIR, 'wandb')
