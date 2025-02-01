from typing import Literal, Optional

import lightning.pytorch as pl
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from src.ml.datamodules.chest_xray.chest_dataset import ChestXrayDataset


class ConfigChestXrayData(BaseModel):
    """A Pydantic Model to validate the MedicalLitDataModule config givent by the user.

    Attributes
    ----------
    dir_train: str
        path to the directory holding the training data
    dir_val: str
        path to the directory holding the validating data
    dir_test: str, optional
        path to the directory holding the testing data
    batch_size: int, optional
        the batch size (default to 1)
    shape_img: tuple[float, float, float, float], optional
        the shape of the image (default to (128, 128, 128, 1))
    shape_label: tuple[float, float, float, float], optional
        the shape of the label (default to (128, 128, 128, 6))
    augment: bool, optional
        whether to use augmentation of data (default to False)
    preprocessed: bool, optional
        whether the data have already been preprocessed or not (default to True)
    num_workers: int, optional
        the number of workers for the DataLoaders (default to 0)
    """

    dir_train: str
    dir_val: str
    dir_test: str = None
    batch_size: int = 1
    # shape_img: tuple[float, float, float, float] = (128, 128, 128, 1)
    # shape_label: tuple[float, float, float, float] = (128, 128, 128, 6)
    augment: bool = False
    # preprocessed: bool = True
    num_workers: int = 0

    model_config = ConfigDict(extra="forbid")


class ConfigData_ChestXray(BaseModel):
    name: Literal["chest_xray"]
    config: ConfigChestXrayData

    model_config = ConfigDict(extra="forbid")


class ChestXrayLitDataModule(pl.LightningDataModule):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    pl : _type_
        _description_
    """

    def __init__(
        self,
        dir_train: str,
        dir_val: str,
        dir_test: str,
        batch_size: int = 1,
        augment: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir_train = dir_train
        self.data_dir_val = dir_val
        self.data_dir_test = dir_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.augment = augment

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.data_train = ChestXrayDataset(
                data_dir=self.data_dir_train, augment=self.augment
            )
            self.data_val = ChestXrayDataset(data_dir=self.data_dir_val)

        if stage == "test" or stage is None:
            self.data_test = ChestXrayDataset(data_dir=self.data_dir_test)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            # pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
        )
