"""
Модуль с моделями классификации уровней разработчиков.
"""

import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler


class LevelClassifier:
    """
    Классификатор уровней разработчиков (junior/middle/senior).
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
        **kwargs
    ) -> None:
        """
        Инициализация классификатора.
        
        Parameters
        ----------
        model_type : str
            Тип модели: 'logistic', 'random_forest', 'gradient_boosting'
        random_state : int
            Seed для воспроизводимости
        **kwargs
            Дополнительные параметры для модели
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.class_names = ['Junior', 'Middle', 'Senior']
        
        # Создание модели
        if model_type == "logistic":
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                multi_class='ovr',
                **kwargs
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                random_state=random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> "LevelClassifier":
        """
        Обучение классификатора.
        
        Parameters
        ----------
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор меток классов (0, 1, 2)
        feature_names : List[str], optional
            Названия признаков
            
        Returns
        -------
        LevelClassifier
            Обученный классификатор
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Пустые данные для обучения")
        
        self.feature_names = feature_names
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение модели
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание классов для новых данных.
        
        Parameters
        ----------
        X : np.ndarray
            Матрица признаков
            
        Returns
        -------
        np.ndarray
            Предсказанные классы (0, 1, 2)
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов.
        
        Parameters
        ----------
        X : np.ndarray
            Матрица признаков
            
        Returns
        -------
        np.ndarray
            Матрица вероятностей для каждого класса
        """
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Union[float, str, np.ndarray]]:
        """
        Оценка качества классификатора.
        
        Parameters
        ----------
        X_test : np.ndarray
            Тестовые признаки
        y_test : np.ndarray
            Истинные метки
            
        Returns
        -------
        Dict
            Словарь с метриками качества
        """
        y_pred = self.predict(X_test)
        
        # Classification report
        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Accuracy
        accuracy = np.mean(y_pred == y_test)
        
        return {
            'accuracy': accuracy,
            'classification_report': report_dict,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Кросс-валидация модели.
        
        Parameters
        ----------
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор меток
        cv : int
            Количество фолдов
            
        Returns
        -------
        Dict[str, float]
            Метрики кросс-валидации
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Для совместимости создаем временную модель того же типа
        if self.model_type == "logistic":
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == "random_forest":
            model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        else:
            model = GradientBoostingClassifier(random_state=self.random_state)
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'scores': scores
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Получение важности признаков (для моделей, которые это поддерживают).
        
        Returns
        -------
        Optional[np.ndarray]
            Важность признаков или None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0)
        else:
            return None
    
    def save(self, path: str) -> None:
        """
        Сохранение модели в файл.
        
        Parameters
        ----------
        path : str
            Путь для сохранения
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> "LevelClassifier":
        """
        Загрузка модели из файла.
        
        Parameters
        ----------
        path : str
            Путь к файлу модели
            
        Returns
        -------
        LevelClassifier
            Загруженная модель
        """
        with open(path, 'rb') as f:
            return pickle.load(f)