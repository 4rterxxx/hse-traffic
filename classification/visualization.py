"""
Модуль для визуализации результатов классификации.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Optional


def plot_class_distribution(
    y: np.ndarray,
    class_names: List[str] = None,
    title: str = "Распределение классов",
    save_path: Optional[str] = None
) -> None:
    """
    Построение графика распределения классов.
    """
    if class_names is None:
        class_names = ['Junior', 'Middle', 'Senior']
    
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    
    # Берем только те классы, которые есть в данных
    present_classes = [class_names[i] for i in unique if i < len(class_names)]
    present_counts = counts
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    bars = plt.bar(present_classes, present_counts, color=colors[:len(present_classes)])
    
    for bar, count in zip(bars, present_counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{count}\n({count/len(y)*100:.1f}%)',
            ha='center',
            va='bottom'
        )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Уровень', fontsize=12)
    plt.ylabel('Количество резюме', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    title: str = "Матрица ошибок",
    save_path: Optional[str] = None
) -> None:
    """
    Построение матрицы ошибок.
    """
    if class_names is None:
        class_names = ['Junior', 'Middle', 'Senior']
    
    # Определяем, какие классы реально есть в матрице
    n_classes = cm.shape[0]
    present_class_names = class_names[:n_classes]
    
    plt.figure(figsize=(8, 6))
    
    # Нормализация
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    
    # Аннотации
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        xticklabels=present_class_names,
        yticklabels=present_class_names,
        cmap='Blues',
        cbar=True,
        square=True
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    title: str = "Важность признаков",
    save_path: Optional[str] = None
) -> None:
    """
    Построение графика важности признаков.
    """
    indices = np.argsort(importance)[::-1]
    
    # Берем топ-15 или меньше если признаков меньше
    n_features = min(15, len(importance))
    sorted_features = [feature_names[i] for i in indices[:n_features]]
    sorted_importance = importance[indices[:n_features]]
    
    plt.figure(figsize=(12, 8))
    
    bars = plt.barh(range(len(sorted_importance)), sorted_importance)
    plt.yticks(range(len(sorted_features)), sorted_features)
    
    for bar, imp in zip(bars, sorted_importance):
        bar.set_color(plt.cm.viridis(imp / max(sorted_importance)))
    
    plt.xlabel('Важность', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, (bar, imp) in enumerate(zip(bars, sorted_importance)):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{imp:.4f}',
            va='center'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = None) -> None:
    """
    Построение графиков обучения.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    if history.get('accuracy'):
        plt.plot(history.get('accuracy', []), label='Train')
    if history.get('val_accuracy'):
        plt.plot(history.get('val_accuracy', []), label='Validation')
    plt.title('Точность', fontsize=12, fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if history.get('loss'):
        plt.plot(history.get('loss', []), label='Train')
    if history.get('val_loss'):
        plt.plot(history.get('val_loss', []), label='Validation')
    plt.title('Функция потерь', fontsize=12, fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()