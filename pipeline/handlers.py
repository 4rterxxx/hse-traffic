"""
Обработчики данных в паттерне Chain of Responsibility.
"""

from abc import ABC, abstractmethod
from typing import Optional
import re

import pandas as pd


class Handler(ABC):
    """Базовый класс обработчика."""

    def __init__(self) -> None:
        """Инициализация обработчика."""
        self._next: Optional["Handler"] = None

    def set_next(self, handler: "Handler") -> "Handler":
        """Установка следующего обработчика в цепочке.

        Parameters
        ----------
        handler : Handler
            Следующий обработчик

        Returns
        -------
        Handler
            Установленный обработчик для fluent-интерфейса
        """
        self._next = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Входной DataFrame

        Returns
        -------
        pd.DataFrame
            Обработанный DataFrame
        """
        df = self._process(df)
        if self._next:
            return self._next.handle(df)
        return df

    @abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Основная логика обработки.

        Parameters
        ----------
        df : pd.DataFrame
            Входной DataFrame

        Returns
        -------
        pd.DataFrame
            Обработанный DataFrame
        """
        pass


class CleanTextHandler(Handler):
    """Удаление неразрывных пробелов и тримминг строк."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка текстовых данных.

        Parameters
        ----------
        df : pd.DataFrame
            Входной DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame с очищенными текстовыми полями
        """
        result = df.copy()
        for col in result.columns:
            if result[col].dtype == "object":  # только строковые колонки
                result[col] = result[col].apply(
                    lambda x: re.sub(
                        r"[\u00a0\u200b\ufeff]", " ", str(x)
                    ).strip()
                    if pd.notna(x)
                    else x
                )
        return result


class DropDuplicatesHandler(Handler):
    """Удаление дублирующихся строк."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление дубликатов.

        Parameters
        ----------
        df : pd.DataFrame
            Входной DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame без дубликатов
        """
        initial_len = len(df)
        result = df.drop_duplicates().reset_index(drop=True)
        removed = initial_len - len(result)
        if removed > 0:
            print(f"Удалено дубликатов: {removed}")
        return result


class DropEmptySalaryHandler(Handler):
    """Удаление строк без указания зарплаты."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация строк с пустой зарплатой.

        Parameters
        ----------
        df : pd.DataFrame
            Входной DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame с заполненной зарплатой

        Raises
        ------
        KeyError
            Если колонка 'ЗП' отсутствует
        """
        if "ЗП" not in df.columns:
            raise KeyError("Колонка 'ЗП' отсутствует в DataFrame")

        initial_len = len(df)
        result = df[df["ЗП"].notna()]
        removed = initial_len - len(result)
        if removed > 0:
            print(f"Удалено строк без зарплаты: {removed}")
        return result