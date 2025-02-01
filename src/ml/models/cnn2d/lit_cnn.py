from typing import Literal, TypedDict

import lightning.pytorch as pl
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torchmetrics.classification import MulticlassAccuracy

from src.ml.models.cnn2d.networks.resnet18 import ResNet18


class ConfigCNN(BaseModel):
    """A Pydantic Model to validate the LitCNN config given by the user.

    Attributes
    ----------
    input_shape:
        number of channels of the input
    n_classes:
        number of classes
    lr: float
        the learning rate
    _logging: bool
        whether to log or not
    """

    input_shape: int
    n_classes: int
    lr: float
    _logging: bool = False

    model_config = ConfigDict(extra="forbid")


class ConfigModel_ChestXray(BaseModel):
    """Pydantic BaseModel to validate Configuration for "cifar" Model.

    Attributes
    ----------
    name:
        designation "cifar" to choose
    config:
        configuration for the model LitCNN
    """

    name: Literal["chest_xray"]
    config: ConfigCNN

    model_config = ConfigDict(extra="forbid")


class CNNSignature(TypedDict):
    loss: torch.Tensor
    accuracy: torch.Tensor


class LitCNN2d(pl.LightningModule):
    def __init__(
        self,
        input_shape: int,
        n_classes: int,
        lr: float,
        _logging: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self._logging = _logging
        
        self.model = ResNet18(in_channels=self.input_shape, n_classes=self.n_classes)

        self.loss = nn.CrossEntropyLoss()
        # self.loss = nn.BCELoss()

        self.accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        
        self._signature = CNNSignature

        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    @property
    def signature(self):
        return self._signature

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch: torch.Tensor, batch_idx) -> CNNSignature:
        input, labels = batch
        
        outputs = self.forward(input)
        loss = self.loss(outputs, labels)
        
        # outputs = self.sigmoid(outputs)
        outputs = self.softmax(outputs)
        
        acc = self.accuracy(torch.argmax(self.softmax(outputs), dim=1), torch.argmax(labels, dim=1))
        
        if self._logging:
            self.log("train_loss", loss, sync_dist=True)
            self.log("train_acc", acc, sync_dist=True)
            
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch: torch.Tensor, batch_idx) -> CNNSignature:
        input, labels = batch
        
        outputs = self.forward(input)
        loss = self.loss(outputs, labels)

        acc = self.accuracy(torch.argmax(self.softmax(outputs), dim=1), torch.argmax(labels, dim=1))
        
        if self._logging:
            self.log("val_loss", loss, sync_dist=True)
            self.log("val_acc", acc, sync_dist=True)
            
        return {"loss": loss, "accuracy": acc}

    def test_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        input, labels = batch
        outputs, _ = torch.max(self.forward(input), 1)
        loss = self.loss(outputs, labels)
        return loss

    def configure_optimizers(self) -> None:
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        # return [optimizer], [scheduler]
