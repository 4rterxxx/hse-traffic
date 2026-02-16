"""
Модуль для извлечения признаков из сырых данных резюме HH.

Содержит функции для парсинга полей резюме HH
и построения матрицы признаков X и вектора целевой переменной y.
"""

import re
from typing import Tuple

import numpy as np
import pandas as pd


def _parse_salary(series: pd.Series) -> pd.Series:
    """Парсинг зарплаты из текста в целые числа.

    Parameters
    ----------
    series : pd.Series
        Колонка с зарплатой

    Returns
    -------
    pd.Series
        Распарсенные значения зарплаты как целые числа
    """
    return (
        series.astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", pd.NA)
        .replace("nan", pd.NA)
        .astype("Int64")
    )


def _parse_gender(series: pd.Series) -> pd.Series:
    """Извлечение индикатора пола.

    Returns
    -------
    pd.Series
        1 для мужчин, 0 иначе
    """
    return (
        series.fillna("")
        .astype(str)
        .str.contains("Мужчина", na=False)
        .astype(int)
    )


def _parse_age(series: pd.Series) -> pd.Series:
    """Извлечение возраста в годах из текста.

    Parameters
    ----------
    series : pd.Series
        Колонка с информацией о возрасте

    Returns
    -------
    pd.Series
        Возраст в годах
    """
    series_filled = series.fillna("")
    extracted = series_filled.astype(str).str.extract(r"(\d+)\s*год")[0]
    return extracted.replace("", pd.NA).astype("Int64")


def _parse_city(series: pd.Series) -> pd.Series:
    """Кодирование названий городов как категориальных целых чисел.

    Parameters
    ----------
    series : pd.Series
        Колонка с городом

    Returns
    -------
    pd.Series
        Коды городов
    """
    cities = (
        series.fillna("")
        .astype(str)
        .str.split(",")
        .str[0]
        .str.strip()
    )
    
    # Топ-10 городов для кодирования
    top_cities = cities.value_counts().head(10).index.tolist()
    
    def encode_city(city):
        if city in top_cities:
            return top_cities.index(city) + 1
        return 0  # other
    
    return cities.apply(encode_city)


def _parse_full_time(series: pd.Series) -> pd.Series:
    """Определение полной занятости.

    Parameters
    ----------
    series : pd.Series
        Колонка с типом занятости

    Returns
    -------
    pd.Series
        1 для полной занятости, 0 иначе
    """
    return (
        series.fillna("")
        .astype(str)
        .str.contains("полная", case=False, na=False)
        .astype(int)
    )


def _parse_has_car(series: pd.Series) -> pd.Series:
    """Определение наличия автомобиля.

    Parameters
    ----------
    series : pd.Series
        Колонка с информацией об автомобиле

    Returns
    -------
    pd.Series
        1 если есть автомобиль, 0 иначе
    """
    return (
        series.fillna("")
        .astype(str)
        .str.contains("автомоб", case=False, na=False)
        .astype(int)
    )


def _parse_higher_education(series: pd.Series) -> pd.Series:
    """Определение наличия высшего образования.

    Parameters
    ----------
    series : pd.Series
        Колонка с образованием

    Returns
    -------
    pd.Series
        1 если есть высшее образование, 0 иначе
    """
    return (
        series.fillna("")
        .astype(str)
        .str.contains("Высшее", case=False, na=False)
        .astype(int)
    )


def _parse_remote(series: pd.Series) -> pd.Series:
    """Определение возможности удаленной работы.

    Parameters
    ----------
    series : pd.Series
        Колонка с графиком работы

    Returns
    -------
    pd.Series
        1 если возможна удаленная работа, 0 иначе
    """
    return (
        series.fillna("")
        .astype(str)
        .str.contains("удален", case=False, na=False)
        .astype(int)
    )


def _parse_experience_years(series: pd.Series) -> pd.Series:
    """Извлечение общего опыта работы в годах.

    Parameters
    ----------
    series : pd.Series
        Колонка с опытом работы

    Returns
    -------
    pd.Series
        Опыт работы в годах
    """
    def extract_years(text):
        if pd.isna(text):
            return pd.NA
        
        text = str(text).lower()
        years = 0
        months = 0
        
        year_match = re.search(r'(\d+)\s*(?:год|года|лет)', text)
        if year_match:
            years = int(year_match.group(1))
        
        month_match = re.search(r'(\d+)\s*мес', text)
        if month_match:
            months = int(month_match.group(1))
        
        total = years + (months / 12)
        return total if total > 0 else pd.NA
    
    return series.apply(extract_years)


def _get_experience_column(df: pd.DataFrame) -> pd.Series:
    """Поиск колонки с опытом работы по префиксу.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными

    Returns
    -------
    pd.Series
        Колонка с опытом работы или пустая серия если не найдена
    """
    for col in df.columns:
        if col.startswith("Опыт") or 'опыт' in str(col).lower():
            return df[col]
    return pd.Series([pd.NA] * len(df))


def _extract_tech_stack(series: pd.Series) -> pd.DataFrame:
    """Извлечение технологий из описания.

    Parameters
    ----------
    series : pd.Series
        Колонка с описанием опыта или должности

    Returns
    -------
    pd.DataFrame
        Признаки по технологиям
    """
    tech_patterns = {
        'python': r'python|питон|django|flask',
        'java': r'java|spring|jvm',
        'javascript': r'javascript|js|react|vue|angular|node',
        'cplus': r'c\+\+|с\+\+',
        'csharp': r'c#|с#|\.net',
        'php': r'php|laravel|symfony',
        'sql': r'sql|mysql|postgresql|баз данных',
        'frontend': r'html|css|frontend|фронтенд',
        'devops': r'devops|docker|kubernetes|aws|cloud',
        'mobile': r'mobile|android|ios|swift|kotlin',
        'data': r'data|аналитик|ml|machine learning|ai',
    }
    
    text = series.fillna("").astype(str).str.lower()
    tech_features = pd.DataFrame(index=series.index)
    
    for tech, pattern in tech_patterns.items():
        tech_features[f'tech_{tech}'] = text.str.contains(pattern, na=False).astype(int)
    
    return tech_features


def build_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Построение матрицы признаков X и вектора целевой переменной y.

    Parameters
    ----------
    df : pd.DataFrame
        Очищенный входной DataFrame

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X - матрица признаков
        y - вектор целевой переменной (зарплата)
    """
    required_cols = [
        "ЗП",
        "Пол, возраст",
        "Город",
        "Занятость",
        "Авто",
        "Образование и ВУЗ",
        "График",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"Предупреждение: отсутствуют колонки: {missing_cols}")
        return np.array([]), np.array([])

    y = _parse_salary(df["ЗП"])
    experience_series = _get_experience_column(df)
    
    tech_features = _extract_tech_stack(df.get('Ищет работу на должность:', pd.Series([""] * len(df))))

    X = pd.DataFrame(
        {
            "gender_male": _parse_gender(df["Пол, возраст"]),
            "age": _parse_age(df["Пол, возраст"]),
            "city": _parse_city(df["Город"]),
            "full_time": _parse_full_time(df["Занятость"]),
            "has_car": _parse_has_car(df["Авто"]),
            "higher_education": _parse_higher_education(
                df["Образование и ВУЗ"]
            ),
            "remote_work": _parse_remote(df["График"]),
            "experience_years": _parse_experience_years(experience_series),
        }
    )
    
    X = pd.concat([X, tech_features], axis=1)

    mask = y.notna()

    if mask.sum() == 0:
        print("Предупреждение: нет данных с указанной зарплатой")
        return np.array([]), np.array([])

    X = X[mask].fillna(0)
    y = y[mask]

    return (
        X.astype("int64").to_numpy(),
        y.astype("int64").to_numpy(),
    )


def build_classification_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Построение матрицы признаков X и вектора целевой переменной y для классификации.
    
    Parameters
    ----------
    df : pd.DataFrame
        Очищенный входной DataFrame (должен содержать колонку 'level')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X - матрица признаков
        y - вектор меток классов (0 - junior, 1 - middle, 2 - senior)
    """
    if 'level' not in df.columns:
        print("Ошибка: колонка 'level' отсутствует. Сначала выполните разметку.")
        return np.array([]), np.array([])
    
    required_cols = [
        "Пол, возраст",
        "Город",
        "Занятость",
        "Авто",
        "Образование и ВУЗ",
        "График",
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Предупреждение: отсутствуют колонки: {missing_cols}")
        return np.array([]), np.array([])
    
    experience_series = _get_experience_column(df)
    
    salary_col = None
    for col in df.columns:
        if col.startswith("ЗП"):
            salary_col = col
            break
    
    if salary_col is None:
        salary_data = pd.Series([0] * len(df))
    else:
        salary_data = _parse_salary(df[salary_col])
    
    tech_features = _extract_tech_stack(df.get('Ищет работу на должность:', pd.Series([""] * len(df))))
    
    X = pd.DataFrame({
        "gender_male": _parse_gender(df["Пол, возраст"]),
        "age": _parse_age(df["Пол, возраст"]),
        "city": _parse_city(df["Город"]),
        "full_time": _parse_full_time(df["Занятость"]),
        "has_car": _parse_has_car(df["Авто"]),
        "higher_education": _parse_higher_education(df["Образование и ВУЗ"]),
        "remote_work": _parse_remote(df["График"]),
        "experience_years": _parse_experience_years(experience_series),
        "salary": salary_data,
    })
    
    X = pd.concat([X, tech_features], axis=1)
    
    y = df['level']
    
    mask = X.notna().all(axis=1) & (y != -1)
    
    if mask.sum() == 0:
        print("Предупреждение: нет валидных данных")
        return np.array([]), np.array([])
    
    X_clean = X[mask].fillna(0)
    y_clean = y[mask]
    
    print(f"Подготовлено данных для классификации: {len(X_clean)} примеров")
    print(f"Распределение классов:")
    for level in [0, 1, 2]:
        count = (y_clean == level).sum()
        if count > 0:
            level_name = ['Junior', 'Middle', 'Senior'][level]
            print(f"  {level_name}: {count} ({count/len(y_clean)*100:.1f}%)")
    
    return (
        X_clean.astype("float64").to_numpy(),
        y_clean.astype("int64").to_numpy(),
    )