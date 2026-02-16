"""
Модуль для разметки уровней специалистов (junior/middle/senior) на основе данных резюме.
"""

import re
from typing import Optional
import pandas as pd
import numpy as np


JUNIOR_KEYWORDS = ['junior', 'младший', 'начинающий', 'стажер', 'trainee', 'intern', 'помощник', 'ассистент']
MIDDLE_KEYWORDS = ['middle', 'миддл', 'мидл']
SENIOR_KEYWORDS = [
    'senior', 'сеньор', 'старший', 'ведущий', 'lead', 'team lead', 
    'тимлид', 'главный', 'principal', 'architect', 'архитектор',
    'руководитель', 'head', 'chief'
]

EXPERIENCE_THRESHOLDS = {
    'junior': (0, 2),
    'middle': (2, 5),
    'senior': (5, 100)
}

LEVEL_MAPPING = {'junior': 0, 'middle': 1, 'senior': 2}


def extract_job_title(df: pd.DataFrame) -> pd.Series:
    """
    Извлечение названия должности.
    """
    position_col = None
    for col in df.columns:
        if 'должн' in str(col).lower():
            position_col = col
            break
    
    if position_col is None:
        position_col = 'Ищет работу на должность:'
    
    return df[position_col].fillna("").astype(str)


def extract_experience_years(experience_text: str) -> Optional[float]:
    """
    Извлечение количества лет опыта.
    """
    if pd.isna(experience_text) or not isinstance(experience_text, str):
        return None
    
    text = str(experience_text).lower()
    years = months = 0
    
    year_match = re.search(r'(\d+)\s*(?:год|года|лет)', text)
    if year_match:
        years = float(year_match.group(1))
    
    month_match = re.search(r'(\d+)\s*мес', text)
    if month_match:
        months = float(month_match.group(1))
    
    total = years + (months / 12)
    return total if total > 0 else None


def get_experience_column(df: pd.DataFrame) -> pd.Series:
    """
    Поиск колонки с опытом.
    """
    for col in df.columns:
        if 'опыт' in str(col).lower():
            return df[col]
    return pd.Series([None] * len(df))


def get_level(df: pd.DataFrame) -> pd.Series:
    """
    Определение уровня: сначала по должности, потом по опыту.
    """
    job_titles = extract_job_title(df)
    experience_col = get_experience_column(df)
    experience_years = experience_col.apply(extract_experience_years)
    
    levels = []
    
    for title, exp in zip(job_titles, experience_years):
        title_lower = str(title).lower()
        level = None
        
        # 1. Сначала по ключевым словам в должности
        for kw in SENIOR_KEYWORDS:
            if kw in title_lower:
                level = 'senior'
                break
        
        if level is None:
            for kw in JUNIOR_KEYWORDS:
                if kw in title_lower:
                    level = 'junior'
                    break
        
        if level is None:
            for kw in MIDDLE_KEYWORDS:
                if kw in title_lower:
                    level = 'middle'
                    break
        
        # 2. Если не определили по должности - по опыту
        if level is None and exp is not None:
            if exp < 2:
                level = 'junior'
            elif exp < 5:
                level = 'middle'
            else:
                level = 'senior'
        
        # 3. Если всё ещё None - назначаем по опыту (даже если None, ставим middle как компромисс)
        if level is None:
            if exp is not None:
                if exp < 2:
                    level = 'junior'
                elif exp < 5:
                    level = 'middle'
                else:
                    level = 'senior'
            else:
                level = 'middle'  # fallback для完全没有 данных
        
        levels.append(LEVEL_MAPPING[level])
    
    return pd.Series(levels)


def add_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление колонки с метками уровней.
    """
    df = df.copy()
    
    df['level'] = get_level(df)
    df['job_title'] = extract_job_title(df)
    df['experience_years_parsed'] = get_experience_column(df).apply(extract_experience_years)
    
    print("\nРаспределение уровней:")
    for level_name, level_num in LEVEL_MAPPING.items():
        count = (df['level'] == level_num).sum()
        percentage = (count / len(df) * 100)
        print(f"  {level_name.title()}: {count} ({percentage:.1f}%)")
    
    return df