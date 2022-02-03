import logging
import os
from datetime import datetime
from typing import List, Type, Union

from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCScenario
from avalanche.benchmarks.scenarios.new_instances.ni_scenario import NIScenario
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, StreamConfusionMatrix
from avalanche.logging import WandBLogger, TextLogger, InteractiveLogger, StrategyLogger
from avalanche.models import BaseModel
from avalanche.training import strategies
from avalanche.training.plugins import EvaluationPlugin
from torch import nn
from torch import optim
from tqdm import tqdm

from continual_learning.config.paths import LOG_DIR


class Avalanche:

    def __init__(
            self,
            model: Type[Union[Type[nn.Module], Type[BaseModel]]],
            optimizer: Type[optim.Optimizer],
            optimizer_kwargs: dict,
            criterion: nn.Module,
            strategy: Type[strategies.BaseStrategy],
            scenario: Union[NCScenario, NIScenario],
            strategy_kwargs=None,
            log_config: dict = None
    ):
        self.scenario = scenario
        self.num_classes = self.scenario.n_classes

        self.model = model(num_classes=scenario.n_classes)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)
        self.criterion = criterion
        self.strategy = strategy

        if strategy_kwargs is None:
            strategy_kwargs = {}

        self.strategy_kwargs = strategy_kwargs

        self.now = datetime.now()
        self.wandb_logger: WandBLogger
        self.log = logging.getLogger()
        self.log_config: dict = log_config
        self.log_config['datetime'] = self.now

    def _set_loggers(self, project_name: str) -> List[StrategyLogger]:
        self.wandb_logger = WandBLogger(
            project_name=project_name,
            run_name=f"{self.log_config['strategy']} - {self.log_config['scenario']}",
            path=LOG_DIR,
            params={
                'config': self.log_config
            }
        )

        text_logger = TextLogger(open(os.path.join(LOG_DIR, 'log.txt'), 'a'))
        interactive_logger = InteractiveLogger()
        return [self.wandb_logger, text_logger, interactive_logger]

    @staticmethod
    def _set_plugin(num_classes: int, loggers: List[StrategyLogger]) -> EvaluationPlugin:
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            StreamConfusionMatrix(num_classes=num_classes, save_image=False),
            # timing_metrics(epoch=True),
            # cpu_usage_metrics(experience=True),
            # disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loggers=loggers
        )

        return eval_plugin

    def _set_strategy(self, eval_plugin: EvaluationPlugin) -> strategies.BaseStrategy:
        strategy = self.strategy(
            self.model,
            self.optimizer,
            self.criterion,
            train_mb_size=500,
            train_epochs=25,
            eval_mb_size=100,
            evaluator=eval_plugin,
            **self.strategy_kwargs
        )

        return strategy

    def run_experiment(self, project_name: str):
        self.log.info('Preparing experiment based on requirements.')

        loggers = self._set_loggers(project_name)
        eval_plugin = self._set_plugin(self.num_classes, loggers)
        strategy = self._set_strategy(eval_plugin)

        self.log.info('Starting experiment...')
        results = []

        for experience in tqdm(self.scenario.train_stream):
            self.log.info("Start of experience: ", experience.current_experience)
            self.log.info("Current Classes: ", experience.classes_in_this_experience)

            strategy.train(experience, num_workers=0)

            self.log.info('Training completed')

            self.log.info('Computing accuracy on the whole test set')
            results.append(strategy.eval(self.scenario.test_stream, num_workers=0))

        self.wandb_logger.wandb.finish()
