import json

from os import path, listdir
from pathlib import Path

from monai.data import CacheDataset
from monai.transforms import Compose

from dtt.utils.logging import get_console

console = get_console()



class JSONCacheDataset:
    """A custom MONAI CacheDataset that loads data from a JSON file.
    The JSON file should contain a list of dictionaries, each with keys "image" and "label", pointing to file paths.
    An additional "name" key can be included for case identification.
    """
    def __init__(
        self,
        data_dir: str,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        transforms: Compose = None
    ):
        self.data_dir = Path(data_dir)
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.transforms = transforms

        # Prepare data list
        self.data = self.create_data_list()

        # Create CacheDataset
        # Note: num_workers=0 for CacheDataset to avoid memory issues
        # The DataLoader will handle parallelism during training
        self.dataset = CacheDataset(
            data=self.data,
            transform=self.transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,  # Set to 0 to prevent double worker spawning
        )

    def __len__(self):
        return len(self.dataset)

    def create_data_list(self):
        try:
            with open(self.data_dir, "r") as f:
                console.log(f"Loading data from {self.data_dir}")
                return json.load(f)
        except Exception:
            console.log(f"Failed to load data from {self.data_dir}")

    def get_dataset(self):
        return self.dataset
