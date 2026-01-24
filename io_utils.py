from pathlib import Path

import pandas as pd
import numpy as np


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_npy(csv_path: str, x: np.ndarray, y: np.ndarray) -> None:
    base_dir = Path(csv_path).parent
    np.save(base_dir / "x_data.npy", x)
    np.save(base_dir / "y_data.npy", y)
