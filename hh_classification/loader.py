"""Загрузка и предобработка данных"""

import pandas as pd
import numpy as np
import re
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Загрузка и обработка резюме"""
    
    def __init__(self):
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка CSV"""
        logger.info(f"Загрузка данных из {filepath}")
        self.raw_data = pd.read_csv(filepath, encoding='utf-8')
        logger.info(f"Загружено {len(self.raw_data)} строк")
        return self.raw_data
    
    def filter_it(self, df: pd.DataFrame):
        """Фильтрация IT-разработчиков"""
        from .config import IT_KEYWORDS
        pattern = re.compile('|'.join(IT_KEYWORDS), re.IGNORECASE)
        
        position_col = None
        for col in df.columns:
            if 'должн' in str(col).lower():
                position_col = col
                break
        
        if position_col is None:
            position_col = 'Ищет работу на должность:'
        
        it_mask = df[position_col].astype(str).str.contains(pattern, na=False)
        filtered = df[it_mask].copy()
        print(f"  Найдено IT-разработчиков: {len(filtered)}")
        return filtered, position_col
    
    def extract_age(self, text: str) -> Optional[int]:
        """Извлечение возраста"""
        if pd.isna(text):
            return None
        match = re.search(r'(\d+)\s*год', str(text))
        return int(match.group(1)) if match else None
    
    def extract_experience(self, text: str) -> Optional[float]:
        """Извлечение опыта"""
        if pd.isna(text):
            return None
        
        text = str(text).lower()
        years = months = 0
        
        year_match = re.search(r'(\d+)\s*год', text)
        if year_match:
            years = float(year_match.group(1))
        
        month_match = re.search(r'(\d+)\s*мес', text)
        if month_match:
            months = float(month_match.group(1))
        
        total = years + (months / 12)
        return total if total > 0 else None
    
    def clean_salary(self, salary: str) -> Optional[float]:
        """Очистка зарплаты"""
        if pd.isna(salary):
            return None
        try:
            clean = re.sub(r'[^\d.]', '', str(salary))
            return float(clean) if clean else None
        except:
            return None
    
    def preprocess(self, df: pd.DataFrame, position_col: str) -> pd.DataFrame:
        """Основная предобработка"""
        processed = df.copy()
        
        if 'ЗП' in processed.columns:
            processed['salary'] = processed['ЗП'].apply(self.clean_salary)
        
        if 'Пол, возраст' in processed.columns:
            processed['age'] = processed['Пол, возраст'].apply(self.extract_age)
        
        if 'Опыт (двойное нажатие для полной версии)' in processed.columns:
            processed['experience'] = processed['Опыт (двойное нажатие для полной версии)'].apply(self.extract_experience)
        
        if 'Город' in processed.columns:
            processed['city'] = processed['Город'].str.split(',').str[0].str.strip()
        
        processed['position'] = processed[position_col]
        
        self.processed_data = processed
        return processed


class TargetBuilder:
    """Создание целевой переменной"""
    
    def __init__(self):
        from .config import JUNIOR_KEYWORDS, MIDDLE_KEYWORDS, SENIOR_KEYWORDS, EXPERIENCE_THRESHOLDS
        self.junior_re = re.compile('|'.join(JUNIOR_KEYWORDS), re.IGNORECASE)
        self.middle_re = re.compile('|'.join(MIDDLE_KEYWORDS), re.IGNORECASE)
        self.senior_re = re.compile('|'.join(SENIOR_KEYWORDS), re.IGNORECASE)
        self.exp_thresholds = EXPERIENCE_THRESHOLDS
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание целевой переменной"""
        result = df.copy()
        
        def get_level_from_position(pos):
            if pd.isna(pos):
                return None
            pos = str(pos).lower()
            if self.senior_re.search(pos):
                return 'senior'
            elif self.junior_re.search(pos):
                return 'junior'
            elif self.middle_re.search(pos):
                return 'middle'
            return None
        
        result['level_pos'] = result['position'].apply(get_level_from_position)
        
        def get_level_from_experience(exp):
            if pd.isna(exp):
                return None
            for level, thresh in self.exp_thresholds.items():
                if thresh['min'] <= exp < thresh['max']:
                    return level
            return None
        
        if 'experience' in result.columns:
            result['level_exp'] = result['experience'].apply(get_level_from_experience)
        else:
            result['level_exp'] = None
        
        def combine_levels(row):
            if pd.notna(row['level_pos']):
                return row['level_pos']
            elif pd.notna(row['level_exp']):
                return row['level_exp']
            return None
        
        result['target'] = result.apply(combine_levels, axis=1)
        
        labeled = result[result['target'].notna()].copy()
        
        print(f"  Размечено резюме: {len(labeled)}")
        print(f"  Распределение:")
        for level, count in labeled['target'].value_counts().items():
            pct = count / len(labeled) * 100
            print(f"    {level}: {count} ({pct:.1f}%)")
        
        return labeled