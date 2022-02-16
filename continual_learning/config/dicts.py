from continual_learning.datamodules.new_classes.mnist import MNIST as NC_MNIST
from continual_learning.datamodules.new_instances.mnist import MNIST as NI_MNIST
from continual_learning.models.cnn import CNN
from continual_learning.strategies.ewc import EWC


STRATEGIES = {
    'ewc': EWC
}

MODELS = {
    'cnn': CNN
}

DATA_MODULES = {
    'nc_mnist': NC_MNIST,
    'ni_mnist': NI_MNIST,
}
