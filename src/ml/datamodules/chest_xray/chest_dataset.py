import os

from os.path import join
from pathlib import Path
from PIL import Image

# import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.console import console

torch.backends.cudnn.enabled = True


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize an image.

    Args:
    -----
        image (torch.Tensor): The image to normalize.

    Returns:
    --------
        torch.Tensor: The normalized image.
    """
    return (image - image.min()) / (image.max() - image.min())


def transform_image(shape: tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Transform an image.

    Args:
    -----
        shape (tuple[int, int]): The shape of the image.

    Returns:
    --------
        v2.Compose: The transformation to apply to the image.
    """
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(shape, antialias=True),
            # v2.Normalize(mean=[122.358], std=[121.465]),
            # v2.Normalize(mean=0.5, std=0.5),
            # v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


def augment_image() -> torch.Tensor:
    """Augment an image.

    Args:
    -----
        shape (tuple[int, int]): The shape of the image.

    Returns:
    --------
        v2.Compose: The augmentation to apply to the image.
    """
    augment = v2.Compose(
        [
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomVerticalFlip(p=0.5),
            # v2.RandomRotation(90),
            v2.RandomAdjustSharpness(0.5),
            # v2.RandomAutocontrast(0.5),
            v2.GaussianBlur(3, sigma=(0.1, 2.0)),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    )
    return augment


# class ChestXrayDataset(Dataset):
#     def __init__(
#         self,
#         data_dir: str,
#         shape_img=(224, 224),
#         # shape_img=(256, 256),
#         augment: bool = False,
#         preprocessed: bool = True,
#     ) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.augment = augment
#         self.preprocessed = preprocessed
#         if self.preprocessed is True:
#             self.path = Path(self.data_dir)
#             console.log(f"Path: {self.path}")

#         self.transforms = transform_image(shape_img)
#         self.augment_image = augment_image()

#         labels = {
#             "COVID": 0,
#             "Lung_Opacity": 1,
#             "Normal": 2,
#             "Viral Pneumonia": 3,
#         }

#         # One-hot encoding
#         # lbl = torch.tensor(list(labels.values()))
#         # one_hot = torch.nn.functional.one_hot(lbl)
#         # print(f"One-hot encoding: {one_hot}")
#         # labels = {
#         #     "COVID": [1, 0, 0, 0],
#         #     "Lung_Opacity": [0, 1, 0, 0],
#         #     "Normal": [0, 0, 1, 0],
#         #     "Viral Pneumonia": [0, 0, 0, 1],
#         # }

#         dirs = os.listdir(self.data_dir)
#         self.images = []
#         for dir in dirs:
#             for file in os.listdir(os.path.join(self.data_dir, dir)):
#                 if (
#                     file.endswith(".png")
#                     or file.endswith(".jpeg")
#                     or file.endswith(".jpg")
#                 ):
#                     # self.images.append(
#                     #     [join(self.data_dir, dir, file), 0 if dir == "NORMAL" else 1]
#                     # )
#                     self.images.append(
#                         [
#                             join(self.data_dir, dir, file),
#                             labels[dir],
#                         ]
#                     )

#         # Make images serializable/pickleable
#         self.images = np.array(self.images)

#         console.log(f"Length images {len(self.images)}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
#         data = self.images[idx]
#         # img = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)
#         img = PIL.Image.open(data[0]).convert("L")
#         label = torch.tensor(int(data[1]), dtype=torch.float32)

#         img = self.transforms(img)
#         # print(f"Image shape: {img.shape}")

#         if self.augment:
#             img = self.augment_image(img)

#         return img, label.long()


# class ChestXrayDataset(Dataset):
#     def __init__(
#         self,
#         data_dir: str,
#         shape_img=(224, 224),
#         # shape_img=(128, 128),
#         # shape_img=(256, 256),
#         augment: bool = False,
#         preprocessed: bool = True,
#     ) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.augment = augment
#         self.preprocessed = preprocessed
#         if self.preprocessed is True:
#             self.path = Path(self.data_dir)
#             console.log(f"Path: {self.path}")

#         self.transforms = transform_image(shape_img)
#         self.augment_image = augment_image()

#         # dirs = os.listdir(self.data_dir)
#         # data_csv = [
#         #     f for f in elements if f.startswith("data_entry") and f.endswith(".csv")
#         # ][0]

#         # self.data_df = pd.read_csv(join(self.data_dir, data_csv))
#         self.data_df = pd.read_csv(self.data_dir)

#         # Select subset
#         self.data_df = self.data_df[self.data_df["Finding Labels"] != "No Finding"]
#         self.data_df = self.data_df[self.data_df["Finding Labels"] != "Hernia"]
#         # self.data_df = self.data_df.iloc[:500]

#         self.data_dir = "/".join(self.data_dir.split("/")[:-2])

#         # Shuffle the dataset
#         self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
#         # print(self.data_df.head())

#         self.img_names = self.data_df["Image Index"].values
#         self.paths = self.data_df["Path"].values
#         self.paths = [path.replace("extracted", "resized") for path in self.paths]
#         # # --- COVID-19 ---
#         # self.data_df.drop(
#         #     columns=["Image Index", "Finding Labels", "Path"], inplace=True
#         # )
#         # --- NIH Chest X-ray ---
#         # print(self.data_df.head())
#         self.data_df.drop(
#             columns=[
#                 # "Unnamed: 0.1",
#                 "Unnamed: 0",
#                 "Image Index",
#                 "Finding Labels",
#                 "Follow-up #",
#                 "Patient ID",
#                 "Patient Age",
#                 "Patient Gender",
#                 "View Position",
#                 "OriginalImagePixelSpacing[x",
#                 "y]",
#                 "Path",
#                 "No Finding",
#                 # "Hernia"
#             ],
#             inplace=True,
#             # errors="ignore",
#         )
#         # print(self.data_df.head())
#         self.data_df = torch.tensor(self.data_df.values, dtype=torch.float32)
#         # print(self.data_df[:5])
#         # self.data_df.drop(columns=["aaa"], inplace=True)
#         # print("HAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYA")
#         # print(self.data_df[:5])

#         # # Make images serializable/pickleable
#         # self.images = np.array(self.images)

#         # console.log(f"Length images {len(self.images)}")
#         console.log(f"Length images {len(self.data_df)}")

#     def __len__(self):
#         # return len(self.images)
#         return len(self.data_df)

#     def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
#         # data = self.data_df.iloc[idx]
#         # data = self.img_names[idx]

#         path = join(
#             # self.data_dir,
#             # self.data_dir.removesuffix("datasets/"),
#             self.data_dir,
#             # data["Path"].removeprefix("../dataset/"),
#             # data["Path"].replace("datasets/", "chest_xray_covid/"),
#             # self.paths[idx],
#             # # --- COVID-19 ---
#             # join(self.paths[idx].replace("datasets/", "chest_xray_covid/"), "images/"),
#             # --- NIH Chest X-ray ---
#             # self.paths[idx].replace("../dataset/resized/", "extracted/"),
#             self.paths[idx].removeprefix("../dataset/"),
#             self.img_names[idx],
#         )

#         img = Image.open(path).convert("L")
#         # img = Image.open(path).convert("RGB")
#         # img = normalize_image(
#         #     torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)
#         # )
#         img = self.transforms(img)

#         # if self.augment:
#         #     img = self.augment_image(img)

#         # One-hot encoded labels for multi-class classification (15 classes - 15 last columns)
#         # label = torch.tensor(
#         #     data.iloc[-15:].values.astype(np.float32), dtype=torch.float32
#         # )

#         # label = torch.tensor(
#         #     # data.iloc[-10:].values.astype(np.float32), dtype=torch.float32
#         #     data.iloc[-4:].values.astype(np.float32),
#         #     dtype=torch.float32,
#         # )
#         label = self.data_df[idx]
#         # print(f"Label: {label} - Shape: {label.shape}")

#         # return img, label.long()
#         return img, label


# class ChestXrayDataset(Dataset):
#     def __init__(
#         self,
#         data_dir: str,
#         shape_img=(224, 224),
#         # shape_img=(128, 128),
#         # shape_img=(256, 256),
#         augment: bool = False,
#         preprocessed: bool = True,
#     ) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.augment = augment
#         self.preprocessed = preprocessed
#         if self.preprocessed is True:
#             self.path = Path(self.data_dir)
#             console.log(f"Path: {self.path}")

#         self.transforms = transform_image(shape_img)
#         self.augment_image = augment_image()

#         self.data_df = pd.read_csv(self.data_dir)
#         self.data_dir = "/".join(self.data_dir.split("/")[:-2])

#         # Shuffle the dataset
#         self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
#         # self.data_df = self.data_df.iloc[:1500]

#         self.img_names = self.data_df["Image Index"].values
#         self.paths = self.data_df["Path"].values
#         self.paths = [path.replace("extracted", "resized") for path in self.paths]

#         self.data_df.drop(
#             columns=[
#                 # "Unnamed: 0.1",
#                 "Unnamed: 0",
#                 "Image Index",
#                 "Finding Labels",
#                 "Follow-up #",
#                 "Patient ID",
#                 "Patient Age",
#                 "Patient Gender",
#                 "View Position",
#                 "OriginalImagePixelSpacing[x",
#                 "y]",
#                 "Path",
#             ],
#             inplace=True,
#             errors="ignore",
#         )
#         self.data_df = torch.tensor(self.data_df.values, dtype=torch.float32)

#         console.log(f"Length images {len(self.data_df)}")

#     def __len__(self):
#         return len(self.data_df)

#     def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
#         path = join(
#             self.data_dir,
#             self.paths[idx].removeprefix("../dataset/"),
#             self.img_names[idx],
#         )

#         img = Image.open(path).convert("L")
#         img = self.transforms(img)

#         if self.augment:
#             img = self.augment_image(img)

#         label = self.data_df[idx]

#         return img, label


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        shape_img=(224, 224),
        # shape_img=(128, 128),
        # shape_img=(256, 256),
        augment: bool = False,
        preprocessed: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.augment = augment
        self.preprocessed = preprocessed
        if self.preprocessed is True:
            self.path = Path(self.data_dir)
            console.log(f"Path: {self.path}")

        self.transforms = transform_image(shape_img)
        self.augment_image = augment_image()

        self.data_df = pd.read_csv(self.data_dir)
        # # FL
        # self.data_dir = "/".join(self.data_dir.split("/")[:-2])
        # CL
        self.data_dir = "/".join(self.data_dir.split("/")[:-1])

        self.data_dir = self.data_dir.replace("csv_splits", "")
        self.data_dir = self.data_dir.replace("2_clients", "")
        self.data_dir = self.data_dir.replace("4_clients", "")
        self.data_dir = self.data_dir.replace("8_clients", "")

        # Shuffle the dataset
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)
        # print(self.data_df.head())

        self.img_names = self.data_df["Image Index"].values
        self.paths = self.data_df["Path"].values

        # ------ COVID-19 -------
        self.paths = [path.removeprefix("datasets/") for path in self.paths]

        self.data_df.drop(
            columns=["Image Index", "Finding Labels", "Path"], inplace=True
        )
        # -----------------------

        # # --- NIH Chest X-ray ---
        # # print(self.data_df.head())
        # self.paths = [path.replace("extracted", "resized") for path in self.paths]
        #
        # self.data_df.drop(
        #     columns=[
        #         # "Unnamed: 0.1",
        #         "Unnamed: 0",
        #         "Image Index",
        #         "Finding Labels",
        #         "Follow-up #",
        #         "Patient ID",
        #         "Patient Age",
        #         "Patient Gender",
        #         "View Position",
        #         "OriginalImagePixelSpacing[x",
        #         "y]",
        #         "Path",
        #     ],
        #     inplace=True,
        #     errors="ignore",
        # )
        # # -----------------------

        self.data_df = torch.tensor(self.data_df.values, dtype=torch.float32)

        console.log(f"Length images {len(self.data_df)}")

    def __len__(self):
        # return len(self.images)
        return len(self.data_df)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        path = join(
            # self.data_dir,
            self.data_dir,
            # self.paths[idx].removeprefix("../dataset/"),
            self.paths[idx],
            self.img_names[idx],
        )

        img = Image.open(path).convert("L")
        img = self.transforms(img)

        if self.augment:
            img = self.augment_image(img)

        label = self.data_df[idx]

        return img, label
