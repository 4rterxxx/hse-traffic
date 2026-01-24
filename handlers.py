from abc import ABC, abstractmethod
from typing import Optional
import re

import pandas as pd


class Handler(ABC):
    def __init__(self) -> None:
        self._next: Optional["Handler"] = None

    def set_next(self, handler: "Handler") -> "Handler":
        self._next = handler
        return handler

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._process(df)
        if self._next:
            return self._next.handle(df)
        return df

    @abstractmethod
    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class CleanTextHandler(Handler):
    """Remove non-breaking spaces and trim strings."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.applymap(
            lambda x: re.sub(r"[\u00a0\u200b\ufeff]", " ", x).strip()
            if isinstance(x, str)
            else x
        )


class DropDuplicatesHandler(Handler):
    """Remove duplicated rows."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)


class DropEmptySalaryHandler(Handler):
    """Remove rows without salary."""

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["ЗП"].notna()]
