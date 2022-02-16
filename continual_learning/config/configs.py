from dataclasses import dataclass
from typing import List, Union

from omegaconf import DictConfig


@dataclass
class NewClassesDataModule:
    batch_size: int
    num_workers: int
    splits: 1


@dataclass
class NewInstancesDataModule:
    batch_size: int
    num_workers: int
    splits: 1


@dataclass
class DataModule:
    name: str
    params: Union[
        DictConfig,
        NewClassesDataModule,
        NewInstancesDataModule
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
    datamodule: DataModule
    max_epochs: int
