from typing import Dict

from avalanche.benchmarks import SplitMNIST, PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive, EWC, Replay, GEM
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from continual_learning.experiments.avalanche import Avalanche

CASES = {
    'ti': {
        'scenario': SplitMNIST(n_experiences=10, seed=42),
        'name':     'SplitMNIST'
    },
    'ci': {
        'scenario': PermutedMNIST(n_experiences=10, seed=42),
        'name':     'PermutedMNIST'
    }
}


def experiment_setup(case: Dict):
    experiments = [
        Avalanche(
            model=SimpleMLP,
            optimizer=SGD,
            optimizer_kwargs={'lr': 0.001, 'momentum': 0.9},
            criterion=CrossEntropyLoss(),
            strategy=Naive,
            scenario=case['scenario'],
            log_config={
                'scenario':   case['name'],
                'optimizer':  'SGD',
                'criterion':  'CrossEntropyLoss',
                'model_name': 'SimpleMLP',
                'strategy':   'Naive',
            }
        ),
        Avalanche(
            model=SimpleMLP,
            optimizer=SGD,
            optimizer_kwargs={'lr': 0.001, 'momentum': 0.9},
            criterion=CrossEntropyLoss(),
            strategy=EWC,
            scenario=case['scenario'],
            strategy_kwargs={'ewc_lambda': 0.1},
            log_config={
                'scenario':   case['name'],
                'optimizer':  'SGD',
                'criterion':  'CrossEntropyLoss',
                'model_name': 'SimpleMLP',
                'strategy':   'EWC',
            }
        ),
        Avalanche(
            model=SimpleMLP,
            optimizer=SGD,
            optimizer_kwargs={'lr': 0.001, 'momentum': 0.9},
            criterion=CrossEntropyLoss(),
            strategy=Replay,
            scenario=case['scenario'],
            log_config={
                'scenario':   case['name'],
                'optimizer':  'SGD',
                'criterion':  'CrossEntropyLoss',
                'model_name': 'SimpleMLP',
                'strategy':   'Replay',
            }
        ),
        Avalanche(
            model=SimpleMLP,
            optimizer=SGD,
            optimizer_kwargs={'lr': 0.001, 'momentum': 0.9},
            criterion=CrossEntropyLoss(),
            strategy=GEM,
            scenario=case['scenario'],
            strategy_kwargs={'patterns_per_exp': 1},
            log_config={
                'scenario':   case['name'],
                'optimizer':  'SGD',
                'criterion':  'CrossEntropyLoss',
                'model_name': 'SimpleMLP',
                'strategy':   'GEM',
            }
        ),
    ]

    return experiments


def run_avalanche_experiment_pipeline():
    ti_experiments = experiment_setup(CASES['ti'])
    ci_experiments = experiment_setup(CASES['ti'])

    for experiment in ti_experiments:
        experiment.run_experiment('task_incremental_poc')

    for experiment in ci_experiments:
        experiment.run_experiment('class_incremental_poc')
