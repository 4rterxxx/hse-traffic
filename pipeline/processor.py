"""
Главный процессор для пайплайна обработки данных.
"""

import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .features import build_xy
from .handlers import (
    CleanTextHandler,
    DropDuplicatesHandler,
    DropEmptySalaryHandler,
)
from .io_utils import read_csv, save_npy


def create_handler_chain() -> CleanTextHandler:
    """Создание цепочки обработчиков данных.

    Returns
    -------
    CleanTextHandler
        Первый обработчик в цепочке
    """
    handler1 = CleanTextHandler()
    handler2 = DropDuplicatesHandler()
    handler3 = DropEmptySalaryHandler()

    handler1.set_next(handler2).set_next(handler3)
    return handler1


def run_processing_pipeline(
    csv_path: str, x_suffix: str = "x_data.npy"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Запуск полного пайплайна обработки данных.

    Parameters
    ----------
    csv_path : str
        Путь к исходному CSV файлу
    x_suffix : str, optional
        Суффикс для сохраненного файла X, по умолчанию "x_data.npy"

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        Матрица признаков X и вектор целевой переменной y,
        или (None, None) в случае ошибки
    """
    try:
        df = read_csv(csv_path)
        print(f"Загружено: {len(df)} строк")

        handler_chain = create_handler_chain()
        df_processed = handler_chain.handle(df)

        print(f"После обработки: {len(df_processed)} строк")

        X, y = build_xy(df_processed)

        if len(X) == 0 or len(y) == 0:
            print("Ошибка: не удалось извлечь признаки")
            return None, None

        print(f"Признаки: {X.shape}, Целевая: {y.shape}")

        save_npy(csv_path, X, y, x_suffix)

        return X, y

    except Exception as e:
        print(f"Ошибка в пайплайне обработки: {e}")
        return None, None


def main() -> None:
    """Основная функция для запуска из командной строки."""
    if len(sys.argv) != 2:
        print("Использование: python processor.py <csv_file>")
        sys.exit(1)

    X, y = run_processing_pipeline(sys.argv[1])

    if X is None or y is None:
        print("Ошибка при обработке данных")
        sys.exit(1)


if __name__ == "__main__":
    main()