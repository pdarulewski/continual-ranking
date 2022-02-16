from typing import List, Optional, Union

from avalanche import benchmarks
from avalanche.benchmarks import NIScenario, NCScenario
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training import BaseStrategy
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import functional
from tqdm import tqdm

from continual_learning.config.configs import DataModule, Strategy
from continual_learning.config.dicts import DATA_MODULES, MODELS, AVALANCHE_STRATEGIES
from continual_learning.config.paths import LOG_DIR
from continual_learning.experiments.experiment import Experiment


class AvalancheBaseline(Experiment):

    def __init__(
            self,
            model: str,
            datamodule: DataModule,
            strategies: List[Strategy],
            project_name: str = None,
            max_epochs: int = 1,
    ):
        super().__init__(
            model=model,
            datamodule=datamodule,
            strategies=strategies,
            project_name=project_name,
            max_epochs=max_epochs
        )

        self.optimizer = None
        self.train_scenario: Optional[Union[NCScenario, NIScenario]] = None
        self.test_scenario: Optional[Union[NCScenario, NIScenario]] = None
        self.plugins = []
        self.strategy: Optional[BaseStrategy] = None

    def prepare_dataloaders(self) -> None:
        class_ = DATA_MODULES[self.datamodule_conf.name]
        datamodule = class_(**self.datamodule_conf.params)
        datamodule.prepare_data()

        scenario = benchmarks.nc_benchmark(
            datamodule.train_dataset,
            datamodule.test_dataset,
            n_experiences=self.datamodule_conf.params.splits,
            shuffle=True,
            task_labels=False
        )

        self.train_scenario = scenario
        self.test_scenario = benchmarks.nc_benchmark(
            datamodule.train_dataset,
            datamodule.test_dataset,
            n_experiences=1,
            shuffle=True,
            task_labels=False
        )

    def setup_loggers(self):
        wandb_logger = WandBLogger(
            project_name=self.project_name,
            path=LOG_DIR,
        )

        interactive_logger = InteractiveLogger()
        self.loggers = [wandb_logger, interactive_logger]

    def setup_strategies(self) -> None:
        for d in self.strategies_conf:
            try:
                if d.name == 'ewc' and d.params.decay_factor is not None and d.params.mode == 'separate':
                    d.params.decay_factor = None
            except KeyError:
                pass

            strategy = AVALANCHE_STRATEGIES[d.name](**d.params)
            self.plugins.append(strategy)

    def setup_model(self) -> None:
        self.model = MODELS[self.model_name]()
        self.optimizer = self.model.configure_optimizers()

    def setup_trainer(self) -> None:
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=False, stream=False),
            loss_metrics(minibatch=True, epoch=True, experience=False, stream=False),
            forgetting_metrics(experience=True, stream=True),
            loggers=self.loggers
        )

        self.strategy = BaseStrategy(
            self.model,
            self.optimizer,
            functional.cross_entropy,
            plugins=self.plugins,
            evaluator=eval_plugin,
            train_mb_size=self.datamodule_conf.params.batch_size,
            eval_mb_size=self.datamodule_conf.params.batch_size
        )

    def run_training(self):
        for experience in tqdm(self.train_scenario.train_stream):
            self.strategy.train(experience, num_workers=0)
            self.strategy.eval(self.train_scenario.test_stream, num_workers=0)

        self.strategy.eval(self.test_scenario.test_stream, num_workers=0)
