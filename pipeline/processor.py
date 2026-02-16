"""
Главный процессор для пайплайна обработки данных.
"""

import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .features import build_xy, build_classification_data  # Добавлен импорт
from .handlers import (
    CleanTextHandler,
    DropDuplicatesHandler,
    DropEmptySalaryHandler,
)
from .io_utils import read_csv, save_npy, save_classification_data  # Добавлен импорт
from .labeling import add_level_labels  # Добавлен импорт


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


def prepare_classification_data(
    csv_path: str,
    x_suffix: str = "x_class.npy",
    y_suffix: str = "y_class.npy"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Подготовка данных для задачи классификации уровней.
    
    Parameters
    ----------
    csv_path : str
        Путь к исходному CSV файлу
    x_suffix : str, optional
        Суффикс для сохраненного файла X
    y_suffix : str, optional
        Суффикс для сохраненного файла Y
        
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        Матрица признаков X и вектор меток классов y
    """
    try:
        df = read_csv(csv_path)
        print(f"Загружено: {len(df)} строк")
        
        # Обработка через цепочку обработчиков
        handler_chain = create_handler_chain()
        df_processed = handler_chain.handle(df)
        print(f"После обработки: {len(df_processed)} строк")
        
        # Добавляем метки уровней
        print("Разметка уровней специалистов...")
        df_labeled = add_level_labels(df_processed)
        
        # Строим данные для классификации
        print("Построение матрицы признаков...")
        X, y = build_classification_data(df_labeled)
        
        if len(X) == 0 or len(y) == 0:
            print("Ошибка: не удалось подготовить данные для классификации")
            return None, None
        
        print(f"\nИтоговые данные:")
        print(f"  Признаки: {X.shape}")
        print(f"  Метки: {y.shape}")
        print(f"  Уникальные классы: {np.unique(y)}")
        
        # Сохраняем данные
        save_classification_data(csv_path, X, y, x_suffix, y_suffix)
        
        return X, y
        
    except Exception as e:
        print(f"Ошибка в пайплайне подготовки данных: {e}")
        return None, None


def main() -> None:
    """Основная функция для запуска из командной строки."""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python processor.py <csv_file>                    # для регрессии")
        print("  python processor.py --classify <csv_file>         # для классификации")
        sys.exit(1)

    if len(sys.argv) == 3 and sys.argv[1] == "--classify":
        # Режим классификации
        X, y = prepare_classification_data(sys.argv[2])
    else:
        # Режим регрессии
        X, y = run_processing_pipeline(sys.argv[1])

    if X is None or y is None:
        print("Ошибка при обработке данных")
        sys.exit(1)


if __name__ == "__main__":
    main()