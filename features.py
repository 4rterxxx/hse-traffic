"""
Feature extraction module.

Contains functions for parsing raw HH resume fields
and building feature matrix X and target vector y.
"""

from typing import Tuple
import re

import numpy as np
import pandas as pd


def _parse_salary(series: pd.Series) -> pd.Series:
    """
    Parse salary values from text to integers.

    Example:
        "60 000 руб." -> 60000

    Parameters
    ----------
    series : pd.Series
        Salary column.

    Returns
    -------
    pd.Series
        Parsed salary values as integers.
    """
    return (
        series.astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", pd.NA)
        .astype("Int64")
    )


def _parse_gender(series: pd.Series) -> pd.Series:
    """
    Extract gender indicator.

    Returns 1 for male, 0 otherwise.
    """
    return series.str.contains("Мужчина", na=False).astype(int)


def _parse_age(series: pd.Series) -> pd.Series:
    """
    Extract age in years from text.
    """
    return series.str.extract(r"(\d+)\s*год")[0].astype("Int64")


def _parse_city(series: pd.Series) -> pd.Series:
    """
    Encode city names as categorical integer codes.
    """
    return series.str.split(",").str[0].astype("category").cat.codes


def _parse_full_time(series: pd.Series) -> pd.Series:
    """
    Detect full-time employment.
    """
    return series.str.contains("полная", case=False, na=False).astype(int)


def _parse_has_car(series: pd.Series) -> pd.Series:
    """
    Detect whether the candidate has a car.
    """
    return series.str.contains("автомоб", case=False, na=False).astype(int)


def _parse_higher_education(series: pd.Series) -> pd.Series:
    """
    Detect presence of higher education.
    """
    return series.str.contains("Высшее", case=False, na=False).astype(int)


def _parse_remote(series: pd.Series) -> pd.Series:
    """
    Detect remote work option.
    """
    return series.str.contains("удален", case=False, na=False).astype(int)


def _parse_experience_years(series: pd.Series) -> pd.Series:
    """
    Extract total work experience in years.
    """
    years = series.str.extract(r"(\d+)\s*лет")[0]
    return years.astype("Int64")


def _get_experience_column(df: pd.DataFrame) -> pd.Series:
    """
    Find experience column by prefix.

    HH CSV contains long column names, so we search
    for a column starting with 'Опыт'.

    Returns
    -------
    pd.Series
        Experience column or empty series if not found.
    """
    for col in df.columns:
        if col.startswith("Опыт"):
            return df[col]
    return pd.Series([pd.NA] * len(df))


def build_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input dataframe.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X - feature matrix
        y - target vector (salary)
    """
    y = _parse_salary(df["ЗП"])

    experience_series = _get_experience_column(df)

    X = pd.DataFrame({
        "gender_male": _parse_gender(df["Пол, возраст"]),
        "age": _parse_age(df["Пол, возраст"]),
        "city": _parse_city(df["Город"]),
        "full_time": _parse_full_time(df["Занятость"]),
        "has_car": _parse_has_car(df["Авто"]),
        "higher_education": _parse_higher_education(df["Образование и ВУЗ"]),
        "remote_work": _parse_remote(df["График"]),
        "experience_years": _parse_experience_years(experience_series),
    })

    mask = y.notna()

    X = X[mask].fillna(0)
    y = y[mask]

    return (
        X.astype("int64").to_numpy(),
        y.astype("int64").to_numpy(),
    )
