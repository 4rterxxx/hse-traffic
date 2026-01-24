import sys

from pipeline import run_pipeline


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python app.py path/to/hh.csv")

    csv_path = sys.argv[1]
    run_pipeline(csv_path)


if __name__ == "__main__":
    main()
