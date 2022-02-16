from dataclasses import dataclass
from typing import List, Any, Union

from omegaconf import DictConfig


@dataclass
class NewClassesDatamodule:
    batch_size: int
    num_workers: int
    split_num: 1


@dataclass
class NewInstancesDatamodule:
    batch_size: int
    num_workers: int
    split_num: 1


@dataclass
class Datamodule:
    name: str
    params: Union[
        DictConfig,
        NewClassesDatamodule,
        NewInstancesDatamodule
    ]


@dataclass
class EWC:
    ewc_lambda: float
    mode: str
    decay_factor: float
    keep_importance_data: bool


@dataclass
class Strategy:
    name: str
    params: Union[
        DictConfig,
        EWC,
    ]


@dataclass
class BaseConfig:
    project_name: str
    baseline: str
    model: str
    strategies: List[Strategy]
    datamodule: Datamodule
    max_epochs: int
