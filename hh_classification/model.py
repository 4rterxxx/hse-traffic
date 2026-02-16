"""Модель и обучение"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class Model:
    """Модель классификации"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=random_state
        )
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        """Обучение модели"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"  Размер train: {X_train.shape}, test: {X_test.shape}")
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  Точность (accuracy): {accuracy:.3f}")
        
        print("\n  Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=False)
        lines = report.split('\n')
        for line in lines[2:-3]:
            if line.strip():
                print(f"  {line}")
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def plot_results(self, y_test, y_pred, feature_importance, save_path=None):
        """Визуализация результатов"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        classes = np.unique(y_test)
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[1].bar(x - width, precision, width, label='Precision')
        axes[1].bar(x, recall, width, label='Recall')
        axes[1].bar(x + width, f1, width, label='F1')
        axes[1].set_title('Metrics by Class')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes)
        axes[1].legend()
        
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            axes[2].barh(range(len(top_features)), top_features['importance'][::-1])
            axes[2].set_yticks(range(len(top_features)))
            axes[2].set_yticklabels(top_features['feature'][::-1])
            axes[2].set_title('Top 10 Feature Importance')
            axes[2].set_xlabel('Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  График сохранен: {save_path}")
        
        plt.show()


def plot_class_distribution(y: pd.Series, save_path=None):
    """График распределения классов"""
    counts = y.value_counts().sort_index()
    colors = ['#FF9999', '#66B2FF', '#99FF99'][:len(counts)]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(counts)), counts.values, color=colors)
    plt.xticks(range(len(counts)), counts.index)
    plt.title('Распределение классов')
    plt.ylabel('Количество')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()