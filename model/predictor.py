import numpy as np
import os
import pickle


def load_trained_model():
    model_type_path = "resources/model_type.npy"
    
    if not os.path.exists(model_type_path):
        raise FileNotFoundError("Модель не найдена. Сначала обучите модель.")
    
    model_type = np.load(model_type_path, allow_pickle=True)[0]
    
    from .linear_reg import LinearRegression, SalaryModel
    
    if model_type == "linear":
        model = LinearRegression()
        
        weights = np.load("resources/model_weights.npy")
        scaler_params = np.load("resources/scaler_params.npy", allow_pickle=True).item()
        
        model.weights = weights
        model.X_mean = scaler_params['X_mean']
        model.X_std = scaler_params['X_std']
        model.y_mean = scaler_params['y_mean']
        model.y_std = scaler_params['y_std']
        model.alpha = scaler_params.get('alpha', 1.0)
        model.degree = scaler_params.get('degree', 1)
        model.normalize = scaler_params.get('normalize', True)
        
        return model
    else:
        with open("resources/model.pkl", "rb") as f:
            model = pickle.load(f)
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