"""
Пакет для классификации уровней разработчиков (junior/middle/senior)
"""
from .classifier import LevelClassifier
from .trainer import train_classifier, evaluate_classifier, compare_classifiers
from .predictor import predict_levels, predict_with_probabilities
from .visualization import (
    plot_class_distribution, 
    plot_confusion_matrix, 
    plot_feature_importance,
    plot_training_history
)

__all__ = [
    'LevelClassifier',
    'train_classifier',
    'evaluate_classifier',
    'compare_classifiers',
    'predict_levels',
    'predict_with_probabilities',
    'plot_class_distribution',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'plot_training_history',
]