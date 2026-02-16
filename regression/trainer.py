import os
import pickle
import sys
from typing import Tuple

import numpy as np

from .linear_reg import LinearRegression, SalaryModel


def train_and_save_model(
    x_path: str, y_path: str, test_size: float = 0.2
) -> None:
    """Обучение и сохранение лучшей модели.

    Parameters
    ----------
    x_path : str
        Путь к файлу с признаками
    y_path : str
        Путь к файлу с целевой переменной
    test_size : float, optional
        Доля тестовой выборки, по умолчанию 0.2

    Raises
    ------
    FileNotFoundError
        Если файлы не найдены
    ValueError
        Если данные некорректны
    """
    # Загрузка данных
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Файлы данных не найдены")

    try:
        X = np.load(x_path).astype(np.float64)
        y = np.load(y_path).astype(np.float64)
    except (IOError, ValueError) as e:
        raise ValueError(f"Ошибка загрузки данных: {e}")

    if X.size == 0 or y.size == 0:
        raise ValueError("Данные пусты")

    if len(X) != len(y):
        raise ValueError("Размеры X и y не совпадают")

    # Фильтрация данных
    mask = (y >= 10000) & (y <= 500000)
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        raise ValueError("Нет данных после фильтрации")

    print(f"Данные: {X.shape[0]} примеров")

    # Разделение на train/test
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))

    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Тестирование моделей
    models_to_test = [
        ("linear", LinearRegression(alpha=1.0, normalize=True, degree=2)),
        ("ridge", SalaryModel(model_type="ridge", use_log=True)),
        (
            "random_forest",
            SalaryModel(model_type="random_forest", use_log=True),
        ),
        (
            "gradient_boosting",
            SalaryModel(model_type="gradient_boosting", use_log=True),
        ),
    ]

    best_model = None
    best_r2 = -float("inf")
    best_name = ""

    for name, model in models_to_test:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            print(f"{name.title()}: R² = {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")

    if best_model is None:
        raise RuntimeError("Не удалось обучить ни одну модель")

    print(f"\nЛучшая модель: {best_name}, R² = {best_r2:.4f}")

    # Дообучение на всех данных
    print(f"\nДообучение {best_name} на всех данных...")

    if best_name == "linear":
        final_model = LinearRegression(alpha=1.0, normalize=True, degree=2)
    elif best_name == "ridge":
        final_model = SalaryModel(model_type="ridge", use_log=True)
    elif best_name == "random_forest":
        final_model = SalaryModel(model_type="random_forest", use_log=True)
    else:  # gradient_boosting
        final_model = SalaryModel(model_type="gradient_boosting", use_log=True)

    final_model.fit(X, y)

    # Сохранение модели
    os.makedirs("resources", exist_ok=True)

    if best_name == "linear":
        np.save("resources/model_weights.npy", final_model.weights)
        np.save("resources/model_type.npy", np.array(["linear"]))

        scaler_params = {
            "X_mean": final_model.X_mean,
            "X_std": final_model.X_std,
            "y_mean": final_model.y_mean,
            "y_std": final_model.y_std,
            "alpha": final_model.alpha,
            "degree": final_model.degree,
            "normalize": final_model.normalize,
        }
    else:
        np.save("resources/model_type.npy", np.array([best_name]))

        with open("resources/model.pkl", "wb") as f:
            pickle.dump(final_model, f)

        scaler_params = {"use_log": final_model.use_log}

    np.save("resources/scaler_params.npy", scaler_params)

    print("Модель сохранена в resources/")


def main() -> None:
    """Основная функция для запуска из командной строки."""
    if len(sys.argv) != 3:
        print("Использование: python trainer.py <x_data.npy> <y_data.npy>")
        sys.exit(1)

    try:
        train_and_save_model(sys.argv[1], sys.argv[2])
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()