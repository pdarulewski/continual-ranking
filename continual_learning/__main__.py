import hydra
from pytorch_lightning import seed_everything

from continual_learning.config.configs import BaseConfig
from continual_learning.config.paths import CONFIG_DIR
from continual_learning.experiments.experiment_runner import ExperimentRunner


@hydra.main(config_path=CONFIG_DIR, config_name='base_config')
def main(cfg: BaseConfig):
    seed_everything(42)

    if cfg.baseline:
        pass

    else:
        experiment = ExperimentRunner(
            model=cfg.model,
            datamodule=cfg.datamodule,
            strategies=cfg.strategies,
            project_name=cfg.project_name,
            max_epochs=cfg.max_epochs,
        )
        experiment.setup()
        experiment.run_training()


if __name__ == '__main__':
    main()
