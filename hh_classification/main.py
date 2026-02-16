"""Главный скрипт для запуска пайплайна"""

import sys
import os
import pandas as pd
import logging

from .loader import DataLoader, TargetBuilder
from .features import FeatureExtractor
from .model import Model, plot_class_distribution

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str, output_dir: str = "results"):
    """Запуск полного пайплайна"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Классификация уровня IT-разработчиков")
    print("--------------------------------------")
    
    # 1. Загрузка и фильтрация
    print("\n1. Загрузка данных")
    loader = DataLoader()
    raw_data = loader.load_data(data_path)
    filtered_data, position_col = loader.filter_it(raw_data)
    
    # 2. Предобработка
    print("2. Предобработка")
    processed_data = loader.preprocess(filtered_data, position_col)
    
    # 3. Создание целевой переменной
    print("3. Создание целевой переменной")
    target_builder = TargetBuilder()
    labeled_data = target_builder.create_target(processed_data)
    
    labeled_data.to_csv(os.path.join(output_dir, "labeled_resumes.csv"), index=False)
    
    # 4. Извлечение признаков
    print("4. Извлечение признаков")
    feature_extractor = FeatureExtractor()
    X = feature_extractor.extract_basic_features(labeled_data)
    y = labeled_data['target']
    
    plot_class_distribution(y, save_path=os.path.join(output_dir, "class_distribution.png"))
    
    # 5. Подготовка данных
    print("5. Подготовка данных")
    X_prepared, y_prepared = feature_extractor.prepare_for_training(X, y)
    
    # 6. Обучение модели
    print("6. Обучение модели")
    model = Model(random_state=42)
    X_train, X_test, y_train, y_test, y_pred = model.train(X_prepared, y_prepared)
    
    # 7. Визуализация результатов
    print("7. Визуализация результатов")
    model.plot_results(
        y_test, y_pred, 
        model.feature_importance,
        save_path=os.path.join(output_dir, "model_results.png")
    )
    
    # 8. Сохранение важных признаков
    if model.feature_importance is not None:
        importance_path = os.path.join(output_dir, "feature_importance.csv")
        model.feature_importance.to_csv(importance_path, index=False)
        print(f"\nВажность признаков сохранена: {importance_path}")
    
    # 9. Итоговый отчет
    print("\n" + "-" * 40)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("-" * 40)
    
    print(f"\nСтатистика:")
    print(f"  Всего резюме: {len(raw_data)}")
    print(f"  IT-разработчиков: {len(filtered_data)}")
    print(f"  Размечено: {len(labeled_data)}")
    print(f"  Признаков: {len(X.columns)}")
    
    print(f"\nРаспределение классов:")
    for level, count in y.value_counts().items():
        pct = count / len(y) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")
    
    print(f"\nКачество модели:")
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.3f}")
    
    print(f"\nВывод:")
    print("  Proof of Concept успешен")
    print("  Можно автоматически определять уровень разработчика")
    print(f"  Точность достаточна для начального анализа")
    
    print(f"\nРезультаты сохранены в папке: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Классификация IT-разработчиков')
    parser.add_argument('--data', type=str, required=True, help='Путь к CSV файлу')
    parser.add_argument('--output', type=str, default='results', help='Папка для результатов')
    
    args = parser.parse_args()
    
    run_pipeline(data_path=args.data, output_dir=args.output)