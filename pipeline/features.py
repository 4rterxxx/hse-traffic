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

    Пример:
        "60 000 руб." -> 60000

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
    cities = cities.replace("", "Unknown")
    return pd.Categorical(cities).codes


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
    years = series.fillna("").astype(str).str.extract(r"(\d+)\s*лет")[0]
    return years.replace("", pd.NA).astype("Int64")


def _get_experience_column(df: pd.DataFrame) -> pd.Series:
    """Поиск колонки с опытом работы по префиксу.

    CSV файлы HH содержат длинные названия колонок,
    поэтому мы ищем колонку, начинающуюся с 'Опыт'.

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
        if col.startswith("Опыт"):
            return df[col]
    return pd.Series([pd.NA] * len(df))


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