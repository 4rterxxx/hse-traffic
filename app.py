#!/usr/bin/env python3
"""
Главный интерфейс приложения
"""

import sys
import os


def print_help():
    print("Использование:")
    print("  Обработка данных (регрессия): python app.py process <hh.csv>")
    print("  Обучение модели (регрессия): python app.py train <x.npy> <y.npy>")
    print("  Предсказание (регрессия):    python app.py predict <x.npy>")
    print()
    print("  Подготовка данных (классификация): python app.py prepare-class <hh.csv>")
    print("  Обучение классификатора:          python app.py train-class <x_class.npy> <y_class.npy>")
    print("  Предсказание уровня:               python app.py predict-level <x_class.npy>")
    print("  Сравнение классификаторов:        python app.py compare-class <x_class.npy> <y_class.npy>")
    print()
    print("Примеры:")
    print("  python app.py process hh.csv")
    print("  python app.py train x_data.npy y_data.npy")
    print("  python app.py prepare-class hh.csv")
    print("  python app.py train-class x_class.npy y_class.npy")
    print("  python app.py predict-level x_class.npy")


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
    
    elif command == "prepare-class":
        if len(sys.argv) != 3:
            print("Ошибка: укажите путь к CSV файлу")
            sys.exit(1)
        
        csv_path = sys.argv[2]
        if not os.path.exists(csv_path):
            print(f"Файл не найден: {csv_path}")
            sys.exit(1)
        
        print(f"Подготовка данных для классификации: {csv_path}")
        
        try:
            from pipeline.processor import prepare_classification_data
            prepare_classification_data(csv_path)
            print("Подготовка завершена. Созданы x_class.npy и y_class.npy")
        except Exception as e:
            print(f"Ошибка подготовки: {e}")
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
        
        print(f"Обучение модели регрессии...")
        
        try:
            from regression.trainer import train_and_save_model
            train_and_save_model(x_path, y_path)
            print("Модель регрессии обучена и сохранена")
        except Exception as e:
            print(f"Ошибка обучения: {e}")
            sys.exit(1)
    
    elif command == "train-class":
        if len(sys.argv) != 4:
            print("Ошибка: укажите оба файла данных")
            sys.exit(1)
        
        x_path, y_path = sys.argv[2], sys.argv[3]
        
        for path in [x_path, y_path]:
            if not os.path.exists(path):
                print(f"Файл не найден: {path}")
                sys.exit(1)
        
        print(f"Обучение классификатора уровней...")
        
        try:
            from classification.trainer import train_classifier
            classifier, metrics = train_classifier(
                x_path, y_path,
                model_type="random_forest"
            )
            
            # Сохранение модели
            os.makedirs("resources", exist_ok=True)
            classifier.save("resources/class_model.pkl")
            print(f"\nКлассификатор сохранён: resources/class_model.pkl")
            
        except Exception as e:
            print(f"Ошибка обучения классификатора: {e}")
            sys.exit(1)
    
    elif command == "compare-class":
        if len(sys.argv) != 4:
            print("Ошибка: укажите оба файла данных")
            sys.exit(1)
        
        x_path, y_path = sys.argv[2], sys.argv[3]
        
        for path in [x_path, y_path]:
            if not os.path.exists(path):
                print(f"Файл не найден: {path}")
                sys.exit(1)
        
        print(f"Сравнение классификаторов...")
        
        try:
            from classification.trainer import compare_classifiers
            results = compare_classifiers(x_path, y_path)
            
            # Сохранение лучшей модели
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            print(f"\nЛучшая модель: {best_model[0]} с accuracy {best_model[1]['accuracy']:.4f}")
            
            os.makedirs("resources", exist_ok=True)
            best_model[1]['classifier'].save("resources/class_model.pkl")
            print(f"Лучшая модель сохранена: resources/class_model.pkl")
            
        except Exception as e:
            print(f"Ошибка сравнения: {e}")
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
            print("Обученная модель регрессии не найдена.")
            print("   Сначала выполните: python app.py train x_data.npy y_data.npy")
            sys.exit(1)
        
        print(f"Предсказание зарплат...")
        
        try:
            from regression.predictor import predict_salaries
            salaries = predict_salaries(x_path)
            
            for salary in salaries:
                print(f"{salary:.2f}")
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            sys.exit(1)
    
    elif command == "predict-level":
        if len(sys.argv) != 3:
            print("Ошибка: укажите файл с признаками")
            sys.exit(1)
        
        x_path = sys.argv[2]
        if not os.path.exists(x_path):
            print(f"Файл не найден: {x_path}")
            sys.exit(1)
        
        if not os.path.exists("resources/class_model.pkl"):
            print("Обученный классификатор не найден.")
            print("   Сначала выполните: python app.py train-class x_class.npy y_class.npy")
            sys.exit(1)
        
        print(f"Предсказание уровней...")
        
        try:
            from classification.predictor import predict_levels, predict_with_probabilities
            
            # Простое предсказание
            levels = predict_levels(x_path)
            print("\nПредсказанные уровни:")
            for i, level in enumerate(levels[:10]):  # Показываем первые 10
                print(f"  {i}: {level}")
            
            # Предсказание с вероятностями (опционально)
            print("\nХотите увидеть предсказания с вероятностями? (y/n)")
            response = input().lower()
            if response == 'y':
                results = predict_with_probabilities(x_path)
                print("\nДетальные предсказания (первые 5):")
                for r in results[:5]:
                    probs = r['probabilities']
                    print(f"  {r['index']}: {r['predicted_level']}")
                    print(f"    Junior: {probs['Junior']:.3f}, "
                          f"Middle: {probs['Middle']:.3f}, "
                          f"Senior: {probs['Senior']:.3f}")
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            sys.exit(1)
    
    elif command in ["help", "--help", "-h"]:
        print_help()
    
    else:
        print(f"Неизвестная команда: {command}")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()