"""Извлечение и подготовка признаков"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Извлечение признаков"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовые признаки"""
        features = pd.DataFrame(index=df.index)
        
        if 'salary' in df.columns:
            features['salary'] = df['salary'].fillna(df['salary'].median())
            features['salary_log'] = np.log1p(features['salary'])
        
        if 'age' in df.columns:
            features['age'] = df['age'].fillna(df['age'].median())
        
        if 'experience' in df.columns:
            features['experience'] = df['experience'].fillna(0)
        
        if 'city' in df.columns:
            top_cities = df['city'].value_counts().head(10).index
            features['city_encoded'] = df['city'].apply(
                lambda x: x if x in top_cities.values else 'other'
            )
            try:
                features['city_encoded'] = self.city_encoder.fit_transform(features['city_encoded'].fillna('unknown'))
            except:
                features['city_encoded'] = 0
        
        if 'position' in df.columns:
            pos = df['position'].fillna('').astype(str).str.lower()
            
            features['pos_length'] = pos.str.len()
            
            techs = {
                'python': r'python|питон',
                'java': r'java|джава',
                'javascript': r'javascript|js|react|vue|angular',
                'c++': r'c\+\+|с\+\+',
                'c#': r'c#|с#',
                'php': r'php',
                'sql': r'sql|баз данных'
            }
            
            for tech, pattern in techs.items():
                features[f'tech_{tech}'] = pos.str.contains(pattern).astype(int)
        
        if 'Образование и ВУЗ' in df.columns:
            edu = df['Образование и ВУЗ'].fillna('').astype(str).str.lower()
            features['has_higher_edu'] = edu.str.contains('высшее|вуз|институт|универ').astype(int)
        
        print(f"  Создано признаков: {len(features.columns)}")
        return features
    
    def prepare_for_training(self, X: pd.DataFrame, y: pd.Series):
        """Подготовка к обучению"""
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
        
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        
        return X, y