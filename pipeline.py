"""
Pipeline orchestration module.
"""

import numpy as np
import pandas as pd

from features import build_xy


def run_pipeline(csv_path: str) -> None:
    """
    Run full data processing pipeline.

    Parameters
    ----------
    csv_path : str
        Path to input CSV file.
    """
    dataframe = pd.read_csv(csv_path)

    features, target = build_xy(dataframe)

    np.save("x_data.npy", features)
    np.save("y_data.npy", target)
