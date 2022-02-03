import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn


class CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        acc = torchmetrics.Accuracy()
        self.train_acc = acc.clone()
        self.val_acc = acc.clone()
        self.test_acc = acc.clone()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        prob = F.log_softmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)

        return loss, prob, y

    def training_step(self, batch, batch_idx):
        loss, prob, targets = self.shared_step(batch, batch_idx)
        self.train_acc(prob, targets)
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/accuracy', self.train_acc, on_step=True, on_epoch=True)

        return {'loss': loss, 'accuracy': self.train_acc}

    def validation_step(self, batch, batch_idx):
        loss, prob, targets = self.shared_step(batch, batch_idx)
        self.val_acc(prob, targets)
        self.log('val/loss', loss, on_step=True, on_epoch=True)
        self.log('val/accuracy', self.val_acc, on_step=True, on_epoch=True)

        return {'loss': loss, 'accuracy': self.val_acc}

    def test_step(self, batch, batch_idx):
        loss, prob, targets = self.shared_step(batch, batch_idx)
        self.test_acc(prob, targets)
        self.log('test/loss', loss, on_step=True, on_epoch=True)
        self.log('test/accuracy', self.test_acc, on_step=True, on_epoch=True)

        return {'loss': loss, 'accuracy': self.test_acc}
