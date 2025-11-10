from __future__ import annotations


def seed_everything(
    seed: int = 42, workers: bool = True, deterministic: bool | None = None
) -> None:
    """Seed common libraries and set deterministic flags if Lightning is available.

    If lightning is installed, delegate to lightning.pytorch.seed_everything for consistent behavior.
    Otherwise, fallback to numpy/python.
    """
    try:
        from lightning.pytorch import seed_everything as _seed

        _seed(seed, workers=workers)
        if deterministic is not None:
            # Lightning sets deterministic flags via Trainer; leave here as a hint
            pass
    except Exception:
        import os
        import random

        import numpy as np

        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
