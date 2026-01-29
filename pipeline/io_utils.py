from pathlib import Path

import pandas as pd
import numpy as np


def read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Успешно загружено: {len(df)} строк")
        return df
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        raise


def save_npy(csv_path: str, x: np.ndarray, y: np.ndarray) -> None:
    try:
        base_dir = Path(csv_path).parent
        x_path = base_dir / "x_data.npy"
        y_path = base_dir / "y_data.npy"
        
        np.save(str(x_path), x)
        np.save(str(y_path), y)
        
        print(f"Сохранено: {x_path}")
        print(f"Сохранено: {y_path}")
    except Exception as e:
        print(f"Ошибка сохранения .npy файлов: {e}")
        raise