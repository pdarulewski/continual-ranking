from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loops import FitLoop


class ContinualTrainer(pl.Trainer):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.task_id: int = 0
        self.fit_loop = ContinualFitLoop()


class ContinualFitLoop(FitLoop):
    def run(self, *args: Any, **kwargs: Any):
        self.reset()

        self.on_run_start()

        self.trainer.should_stop = False

        while not self.done:
            try:
                self.on_advance_start()
                self.advance()
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False

        self.on_run_end()
