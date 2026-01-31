import numpy as np
import os


def train_and_save_model(x_path, y_path, test_size=0.2):
    X = np.load(x_path).astype(np.float64)
    y = np.load(y_path).astype(np.float64)
    
    mask = (y >= 10000) & (y <= 500000)
    X = X[mask]
    y = y[mask]
    
    print(f"Данные: {X.shape[0]} примеров")
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    from .linear_reg import LinearRegression, SalaryModel
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    # Линейная регрессия
    model = LinearRegression(alpha=1.0, normalize=True, degree=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"Linear Regression: R² = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = "linear"
    
    # Ridge
    ridge = SalaryModel(model_type='ridge', use_log=True)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"Ridge Regression: R² = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = ridge
        best_name = "ridge"
    
    # Random Forest
    rf = SalaryModel(model_type='random_forest', use_log=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"Random Forest: R² = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = rf
        best_name = "random_forest"
    
    # Gradient Boosting
    gb = SalaryModel(model_type='gradient_boosting', use_log=True)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"Gradient Boosting: R² = {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = gb
        best_name = "gradient_boosting"
    
    print(f"\nЛучшая модель: {best_name}, R² = {best_r2:.4f}")
    
    # Обучаем на всех данных
    print(f"\nДообучение {best_name} на всех данных...")
    
    if best_name == "linear":
        final_model = LinearRegression(alpha=1.0, normalize=True, degree=2)
    else:
        if best_name == "ridge":
            final_model = SalaryModel(model_type='ridge', use_log=True)
        elif best_name == "random_forest":
            final_model = SalaryModel(model_type='random_forest', use_log=True)
        else:
            final_model = SalaryModel(model_type='gradient_boosting', use_log=True)
    
    final_model.fit(X, y)
    
    # Сохраняем
    os.makedirs("resources", exist_ok=True)
    
    if best_name == "linear":
        np.save("resources/model_weights.npy", final_model.weights)
        np.save("resources/model_type.npy", np.array(["linear"]))
        
        scaler_params = {
            'X_mean': final_model.X_mean,
            'X_std': final_model.X_std,
            'y_mean': final_model.y_mean,
            'y_std': final_model.y_std,
            'alpha': final_model.alpha,
            'degree': final_model.degree,
            'normalize': final_model.normalize
        }
    else:
        # Для sklearn моделей сохраняем только тип
        np.save("resources/model_type.npy", np.array([best_name]))
        
        # Сохраняем веса модели и скейлер
        import pickle
        with open("resources/model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        
        scaler_params = {
            'use_log': final_model.use_log
        }
    
    np.save("resources/scaler_params.npy", scaler_params)
    
    print("Модель сохранена в resources/")