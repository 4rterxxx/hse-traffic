import numpy as np
import os


def load_trained_model():
    weights_path = "resources/model_weights.npy"
    scaler_path = "resources/scaler_params.npy"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл не найден: {weights_path}")
    
    weights = np.load(weights_path)
    scaler_params = np.load(scaler_path, allow_pickle=True).item()
    
    from .linear_reg import SalaryModel
    model = SalaryModel(
        alpha=scaler_params.get('alpha', 1.0),
        normalize=scaler_params.get('normalize', True),
        degree=scaler_params.get('degree', 1)
    )
    
    model.weights = weights
    model.X_mean = scaler_params['X_mean']
    model.X_std = scaler_params['X_std']
    model.y_mean = scaler_params['y_mean']
    model.y_std = scaler_params['y_std']
    
    return model


def predict_salaries(x_path):
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Файл не найден: {x_path}")
    
    X = np.load(x_path).astype(np.float64)
    model = load_trained_model()
    salaries = model.predict(X)
    
    return salaries.tolist()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Использование: python predictor.py <x_data.npy>")
        sys.exit(1)
    
    salaries = predict_salaries(sys.argv[1])
    
    for salary in salaries:
        print(f"{salary:.2f}")