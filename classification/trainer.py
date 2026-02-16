"""
Модуль для обучения и оценки классификаторов уровней.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.classifier import LevelClassifier
from classification.visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance
)


def train_classifier(
    X_path: str,
    y_path: str,
    test_size: float = 0.2,
    model_type: str = "random_forest",
    random_state: int = 42
) -> Tuple[LevelClassifier, Dict]:
    """
    Обучение классификатора на данных из файлов.
    """
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Файлы данных не найдены")
    
    try:
        X = np.load(X_path).astype(np.float64)
        y = np.load(y_path).astype(np.int64)
    except (IOError, ValueError) as e:
        raise ValueError(f"Ошибка загрузки данных: {e}")
    
    if X.size == 0 or y.size == 0:
        raise ValueError("Данные пусты")
    
    if len(X) != len(y):
        raise ValueError("Размеры X и y не совпадают")
    
    print(f"Загружено данных: {X.shape[0]} примеров, {X.shape[1]} признаков")
    print(f"Распределение классов:")
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['Junior', 'Middle', 'Senior']
    for cls, count in zip(unique, counts):
        if cls < len(class_names):
            print(f"  {class_names[cls]}: {count} ({count/len(y)*100:.1f}%)")
    
    # Визуализация распределения классов
    plot_class_distribution(y)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\nРазмер выборок:")
    print(f"  Train: {X_train.shape[0]} примеров")
    print(f"  Test: {X_test.shape[0]} примеров")
    
    # Создание и обучение классификатора
    feature_names = [
        "Пол (мужской)",
        "Возраст",
        "Город (код)",
        "Полная занятость",
        "Наличие авто",
        "Высшее образование",
        "Удаленная работа",
        "Опыт (годы)",
        "Зарплата",
        "Tech Python",
        "Tech Java",
        "Tech JavaScript",
        "Tech C++",
        "Tech C#",
        "Tech PHP",
        "Tech SQL",
        "Tech Frontend",
        "Tech DevOps",
        "Tech Mobile",
        "Tech Data"
    ]
    
    classifier = LevelClassifier(
        model_type=model_type,
        random_state=random_state
    )
    
    print(f"\nОбучение модели {model_type}...")
    classifier.fit(X_train, y_train, feature_names[:X.shape[1]])
    
    # Оценка на тестовой выборке
    metrics = classifier.evaluate(X_test, y_test)
    
    print(f"\n=== Результаты классификации ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nClassification Report:")
    
    report = metrics['classification_report']
    print(f"{'':20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    
    # Определяем, какие классы есть в отчете
    present_classes = [c for c in ['Junior', 'Middle', 'Senior'] if c in report]
    for class_name in present_classes:
        print(f"{class_name:20} {report[class_name]['precision']:10.4f} "
              f"{report[class_name]['recall']:10.4f} {report[class_name]['f1-score']:10.4f} "
              f"{report[class_name]['support']:10.0f}")
    
    # Визуализация confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=['Junior', 'Middle', 'Senior']
    )
    
    # Важность признаков
    importance = classifier.get_feature_importance()
    if importance is not None:
        plot_feature_importance(importance, feature_names[:X.shape[1]])
    
    return classifier, metrics


def evaluate_classifier(
    classifier: LevelClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Оценка качества классификатора.
    """
    return classifier.evaluate(X_test, y_test)


def compare_classifiers(
    X_path: str,
    y_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Сравнение различных классификаторов.
    """
    model_types = ["logistic", "random_forest", "gradient_boosting"]
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Тестирование модели: {model_type}")
        print('='*50)
        
        try:
            classifier, metrics = train_classifier(
                X_path, y_path,
                test_size=test_size,
                model_type=model_type,
                random_state=random_state
            )
            
            results[model_type] = {
                'classifier': classifier,
                'accuracy': metrics['accuracy'],
                'report': metrics['classification_report']
            }
            
        except Exception as e:
            print(f"Ошибка при обучении {model_type}: {e}")
    
    print(f"\n{'='*50}")
    print("Сравнение моделей:")
    print('='*50)
    for model_type, result in results.items():
        print(f"{model_type:20} Accuracy: {result['accuracy']:.4f}")
    
    return results