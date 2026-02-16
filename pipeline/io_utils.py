from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    """Загрузка CSV файла.

    Parameters
    ----------
    path : str
        Путь к CSV файлу

    Returns
    -------
    pd.DataFrame
        Загруженный DataFrame

    Raises
    ------
    FileNotFoundError
        Если файл не найден
    ValueError
        Если файл некорректен
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Успешно загружено: {len(df)} строк")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("Файл пуст")
    except pd.errors.ParserError as e:
        raise ValueError(f"Ошибка парсинга CSV: {e}")
    except Exception as e:
        raise ValueError(f"Ошибка загрузки CSV: {e}")


def save_npy(
    csv_path: str, x: np.ndarray, y: np.ndarray, x_suffix: str = "x_data.npy"
) -> None:
    """Сохранение данных в формате .npy.

    Parameters
    ----------
    csv_path : str
        Исходный путь к CSV файлу (для определения директории)
    x : np.ndarray
        Матрица признаков
    y : np.ndarray
        Вектор целевой переменной
    x_suffix : str, optional
        Суффикс для файла X, по умолчанию "x_data.npy"

    Raises
    ------
    ValueError
        Если данные пусты
    """
    if x.size == 0 or y.size == 0:
        raise ValueError("Пустые данные для сохранения")

    try:
        base_dir = Path(csv_path).parent
        x_path = base_dir / x_suffix
        y_path = base_dir / "y_data.npy"

        np.save(str(x_path), x)
        np.save(str(y_path), y)

        print(f"Сохранено: {x_path}")
        print(f"Сохранено: {y_path}")
    except Exception as e:
        raise ValueError(f"Ошибка сохранения .npy файлов: {e}")

def save_classification_data(
    csv_path: str, 
    X: np.ndarray, 
    y: np.ndarray,
    x_suffix: str = "x_class.npy",
    y_suffix: str = "y_class.npy"
) -> None:
    """
    Сохранение данных для классификации в формате .npy.
    
    Parameters
    ----------
    csv_path : str
        Исходный путь к CSV файлу (для определения директории)
    X : np.ndarray
        Матрица признаков
    y : np.ndarray
        Вектор меток классов
    x_suffix : str, optional
        Суффикс для файла X, по умолчанию "x_class.npy"
    y_suffix : str, optional
        Суффикс для файла Y, по умолчанию "y_class.npy"
        
    Raises
    ------
    ValueError
        Если данные пусты
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("Пустые данные для сохранения")
    
    try:
        base_dir = Path(csv_path).parent
        x_path = base_dir / x_suffix
        y_path = base_dir / y_suffix
        
        np.save(str(x_path), X)
        np.save(str(y_path), y)
        
        print(f"Сохранено: {x_path}")
        print(f"Сохранено: {y_path}")
        print(f"Размерность X: {X.shape}, y: {y.shape}")
    except Exception as e:
        raise ValueError(f"Ошибка сохранения .npy файлов: {e}")