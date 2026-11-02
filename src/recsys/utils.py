import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed all random number generators for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
