from handlers import (
    CleanTextHandler,
    DropDuplicatesHandler,
    DropEmptySalaryHandler,
)
from io_utils import read_csv, save_npy
from features import build_xy


def run_pipeline(csv_path: str) -> None:
    df = read_csv(csv_path)

    pipeline = (
        CleanTextHandler()
        .set_next(DropDuplicatesHandler())
        .set_next(DropEmptySalaryHandler())
    )

    df = pipeline.handle(df)

    x, y = build_xy(df)
    save_npy(csv_path, x, y)
