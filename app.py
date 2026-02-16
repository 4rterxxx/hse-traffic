#!/usr/bin/env python3
"""
Главный интерфейс приложения
"""

import sys
import os


def print_help():
    print("Использование:")
    print("  Обработка данных:   python app.py process <hh.csv>")
    print("  Обучение модели:    python app.py train <x.npy> <y.npy>")
    print("  Предсказание:       python app.py predict <x.npy>")
    print("  Классификация IT:   python app.py classify <hh.csv> [output_dir]")
    print()
    print("Примеры:")
    print("  python app.py process hh.csv")
    print("  python app.py train x_data.npy y_data.npy")
    print("  python app.py predict x_data.npy")
    print("  python app.py classify hh.csv")
    print("  python app.py classify hh.csv my_results")


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "process":
        if len(sys.argv) != 3:
            print("Ошибка: укажите путь к CSV файлу")
            sys.exit(1)
        
        csv_path = sys.argv[2]
        if not os.path.exists(csv_path):
            print(f"Файл не найден: {csv_path}")
            sys.exit(1)
        
        print(f"Обработка файла: {csv_path}")
        
        try:
            from pipeline.processor import run_processing_pipeline
            run_processing_pipeline(csv_path)
            print("Обработка завершена. Созданы x_data.npy и y_data.npy")
        except Exception as e:
            print(f"Ошибка обработки: {e}")
            sys.exit(1)
    
    elif command == "train":
        if len(sys.argv) != 4:
            print("Ошибка: укажите оба файла данных")
            sys.exit(1)
        
        x_path, y_path = sys.argv[2], sys.argv[3]
        
        for path in [x_path, y_path]:
            if not os.path.exists(path):
                print(f"Файл не найден: {path}")
                sys.exit(1)
        
        print(f"Обучение модели...")
        
        try:
            from hh_regression.trainer import train_and_save_model
            train_and_save_model(x_path, y_path)
            print("Модель обучена и сохранена")
        except Exception as e:
            print(f"Ошибка обучения: {e}")
            sys.exit(1)
    
    elif command == "predict":
        if len(sys.argv) != 3:
            print("Ошибка: укажите файл с признаками")
            sys.exit(1)
        
        x_path = sys.argv[2]
        if not os.path.exists(x_path):
            print(f"Файл не найден: {x_path}")
            sys.exit(1)
        
        if not os.path.exists("resources/model_weights.npy"):
            print("Обученная модель не найдена.")
            print("   Сначала выполните: python app.py train x_data.npy y_data.npy")
            sys.exit(1)
        
        print(f"Предсказание зарплат...")
        
        try:
            from hh_regression.predictor import predict_salaries
            salaries = predict_salaries(x_path)
            
            for salary in salaries:
                print(f"{salary:.2f}")
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            sys.exit(1)
    
    elif command == "classify":
        """Классификация IT-разработчиков"""
        if len(sys.argv) < 3:
            print("Ошибка: укажите путь к CSV файлу")
            sys.exit(1)
        
        csv_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "hh_results"
        
        if not os.path.exists(csv_path):
            print(f"Файл не найден: {csv_path}")
            sys.exit(1)
        
        print(f"Классификация IT-разработчиков из файла: {csv_path}")
        print(f"Результаты будут сохранены в папке: {output_dir}")
        print()
        
        try:
            from hh_classification.main import run_pipeline
            run_pipeline(data_path=csv_path, output_dir=output_dir)
        except ImportError:
            print("Модуль классификации не найден. Проверьте наличие папки hh_classification/")
            sys.exit(1)
        except Exception as e:
            print(f"Ошибка классификации: {e}")
            sys.exit(1)
    
    elif command in ["help", "--help", "-h"]:
        print_help()
    
    else:
        print(f"Неизвестная команда: {command}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()