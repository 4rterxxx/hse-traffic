"""
Модуль для предсказания уровней разработчиков.
"""

import os
import sys
from typing import List, Union

import numpy as np

from .classifier import LevelClassifier


def load_classifier() -> LevelClassifier:
    """
    Загрузка обученного классификатора из файла.
    
    Returns
    -------
    LevelClassifier
        Загруженный классификатор
        
    Raises
    ------
    FileNotFoundError
        Если файл модели не найден
    """
    model_path = "resources/class_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Классификатор не найден. Сначала обучите модель."
        )
    
    return LevelClassifier.load(model_path)


def predict_levels(x_path: str) -> List[Union[str, int]]:
    """
    Предсказание уровней для данных из файла.
    
    Parameters
    ----------
    x_path : str
        Путь к файлу с признаками
        
    Returns
    -------
    List[Union[str, int]]
        Список предсказанных уровней (как строки или числа)
        
    Raises
    ------
    FileNotFoundError
        Если файл не найден
    ValueError
        Если данные некорректны
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Файл не найден: {x_path}")
    
    try:
        X = np.load(x_path).astype(np.float64)
    except (IOError, ValueError) as e:
        raise ValueError(f"Ошибка загрузки данных: {e}")
    
    if X.size == 0:
        raise ValueError("Файл содержит пустые данные")
    
    classifier = load_classifier()
    predictions = classifier.predict(X)
    
    # Конвертируем в названия уровней
    level_names = ['Junior', 'Middle', 'Senior']
    return [level_names[p] for p in predictions]


def predict_with_probabilities(x_path: str) -> List[dict]:
    """
    Предсказание уровней с вероятностями.
    
    Parameters
    ----------
    x_path : str
        Путь к файлу с признаками
        
    Returns
    -------
    List[dict]
        Список словарей с предсказаниями и вероятностями
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Файл не найден: {x_path}")
    
    try:
        X = np.load(x_path).astype(np.float64)
    except (IOError, ValueError) as e:
        raise ValueError(f"Ошибка загрузки данных: {e}")
    
    classifier = load_classifier()
    predictions = classifier.predict(X)
    probabilities = classifier.predict_proba(X)
    
    level_names = ['Junior', 'Middle', 'Senior']
    results = []
    
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        results.append({
            'index': i,
            'predicted_level': level_names[pred],
            'predicted_class': int(pred),
            'probabilities': {
                level_names[j]: float(probs[j])
                for j in range(len(level_names))
            }
        })
    
    return results