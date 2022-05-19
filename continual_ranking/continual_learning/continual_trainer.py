import pytorch_lightning as pl


class ContinualTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id: int = 0
