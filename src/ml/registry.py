from src.ml.datamodules.chest_xray.chest_datamodule import ChestXrayLitDataModule
from src.ml.models.cnn2d.lit_cnn import LitCNN2d

model_registry = {
    "resnet18": LitCNN2d,
}
datamodule_registry = {
    "chest_xray": ChestXrayLitDataModule,
}
