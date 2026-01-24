import pandas as pd
import numpy as np


def _parse_salary(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", pd.NA)
        .astype("Int64")
    )


def _parse_gender(series: pd.Series) -> pd.Series:
    return series.str.contains("Мужчина").astype(int)


def _parse_age(series: pd.Series) -> pd.Series:
    return series.str.extract(r"(\d+)\s*год")[0].astype("Int64")


def _parse_city(series: pd.Series) -> pd.Series:
    return series.str.split(",").str[0].astype("category").cat.codes


def _parse_full_time(series: pd.Series) -> pd.Series:
    return series.str.contains("полная", case=False).astype(int)


def build_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = _parse_salary(df["ЗП"])

    X = pd.DataFrame({
        "gender_male": _parse_gender(df["Пол, возраст"]),
        "age": _parse_age(df["Пол, возраст"]),
        "city": _parse_city(df["Город"]),
        "full_time": _parse_full_time(df["Занятость"]),
    })

    mask = y.notna()

    X = X[mask].fillna(0)
    y = y[mask]

    return (
        X.astype("int64").to_numpy(),
        y.astype("int64").to_numpy(),
    )
