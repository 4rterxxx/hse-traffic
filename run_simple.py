# run_simple.py
"""Простой запуск классификации"""

import sys
import os

# Добавляем путь к папке hh_classification
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hh_classification.main import run_pipeline

if __name__ == "__main__":
    # Укажите путь к вашему CSV файлу
    data_path = "hh.csv"
    
    if not os.path.exists(data_path):
        print(f"Файл {data_path} не найден!")
        print("Пожалуйста, укажите правильный путь к CSV файлу")
    else:
        run_pipeline(data_path=data_path, output_dir="hh_results")