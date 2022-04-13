from torch.utils import data


class CustomDataLoader(data.DataLoader):
    def __init__(self, classes: list, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
