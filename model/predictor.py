import os
import pickle
import sys
from typing import List, Union

import numpy as np

from .linear_reg import LinearRegression, SalaryModel


def load_trained_model() -> Union[LinearRegression, SalaryModel]:
    """Загрузка обученной модели из файлов.

    Returns
    -------
    Union[LinearRegression, SalaryModel]
        Загруженная модель

    Raises
    ------
    FileNotFoundError
        Если файлы модели не найдены
    """
    model_type_path = "resources/model_type.npy"

    if not os.path.exists(model_type_path):
        raise FileNotFoundError(
            "Модель не найдена. Сначала обучите модель."
        )

    model_type = np.load(model_type_path, allow_pickle=True)[0]

    if model_type == "linear":
        model = LinearRegression()

        weights = np.load("resources/model_weights.npy")
        scaler_params = np.load(
            "resources/scaler_params.npy", allow_pickle=True
        ).item()

        model.weights = weights
        model.X_mean = scaler_params["X_mean"]
        model.X_std = scaler_params["X_std"]
        model.y_mean = scaler_params["y_mean"]
        model.y_std = scaler_params["y_std"]
        model.alpha = scaler_params.get("alpha", 1.0)
        model.degree = scaler_params.get("degree", 1)
        model.normalize = scaler_params.get("normalize", True)

        return model
    else:
        with open("resources/model.pkl", "rb") as f:
            model = pickle.load(f)
        return model


def predict_salaries(x_path: str) -> List[float]:
    """Предсказание зарплат для данных из файла.

    Parameters
    ----------
    x_path : str
        Путь к файлу с признаками

    Returns
    -------
    List[float]
        Список предсказанных зарплат

    Raises
    ------
    FileNotFoundError
        Если файл не найден
    ValueError
        Если данные некорректны
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Файл не найден: {x_path}")

    try:
        X = np.load(x_path).astype(np.float64)
    except (IOError, ValueError) as e:
        raise ValueError(f"Ошибка загрузки данных: {e}")

    if X.size == 0:
        raise ValueError("Файл содержит пустые данные")

    model = load_trained_model()
    salaries = model.predict(X)

    return salaries.tolist()


def main() -> None:
    """Основная функция для запуска из командной строки."""
    if len(sys.argv) != 2:
        print("Использование: python predictor.py <x_data.npy>")
        sys.exit(1)

    try:
        salaries = predict_salaries(sys.argv[1])
        for salary in salaries:
            print(f"{salary:.2f}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()