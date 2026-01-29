"""
Главный процессор для пайплайна обработки данных
"""

import pandas as pd
import numpy as np

from .handlers import CleanTextHandler, DropDuplicatesHandler, DropEmptySalaryHandler
from .features import build_xy
from .io_utils import read_csv, save_npy


def create_handler_chain():
    handler1 = CleanTextHandler()
    handler2 = DropDuplicatesHandler()
    handler3 = DropEmptySalaryHandler()
    
    handler1.set_next(handler2).set_next(handler3)
    return handler1


def run_processing_pipeline(csv_path):
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
    
    save_npy(csv_path, X, y)
    
    return X, y


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Использование: python processor.py <csv_file>")
        sys.exit(1)
    
    run_processing_pipeline(sys.argv[1])