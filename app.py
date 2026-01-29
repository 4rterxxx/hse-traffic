#!/usr/bin/env python3
"""
Пайплайн обработки данных резюме hh.ru
"""

import sys
import os


def print_help():
    print("Использование: python app.py path/to/hh.csv")
    print()
    print("Создаст рядом с CSV файлом:")
    print("  - x_data.npy - матрицу признаков")
    print("  - y_data.npy - вектор зарплат")
    print()
    print("Пример:")
    print("  python app.py hh.csv")


def main():
    if len(sys.argv) != 2:
        print_help()
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Файл не найден: {csv_path}")
        sys.exit(1)
    
    print(f"Обработка файла: {csv_path}")
    
    try:
        from pipeline.processor import run_processing_pipeline
        run_processing_pipeline(csv_path)
        
        print("Обработка завершена")
        print("Созданы x_data.npy и y_data.npy")
        
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()