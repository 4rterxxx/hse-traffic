import numpy as np
import os


def train_and_save_model(x_path, y_path, test_size=0.2):
    X = np.load(x_path).astype(np.float64)
    y = np.load(y_path).astype(np.float64)
    
    # Фильтрация выбросов
    mask = (y >= 10000) & (y <= 500000)
    X = X[mask]
    y = y[mask]
    
    print(f"Данные для обучения: {X.shape[0]} примеров")
    print(f"Диапазон зарплат: {y.min():.0f} - {y.max():.0f} руб.")
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Разделение: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    from .linear_reg import SalaryModel
    
    best_model = None
    best_r2 = -float('inf')
    best_params = {}
    
    # Тестируем разные конфигурации
    for degree in [1, 2]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            model = SalaryModel(alpha=alpha, normalize=True, degree=degree)
            model.fit(X_train, y_train)
            
            y_test_pred = model.predict(X_test)
            
            ss_res = np.sum((y_test - y_test_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_params = {'alpha': alpha, 'degree': degree}
    
    print(f"\nЛучшая модель: degree={best_params['degree']}, alpha={best_params['alpha']}")
    print(f"R² на тесте: {best_r2:.4f}")
    
    # Обучаем на всех данных
    final_model = SalaryModel(
        alpha=best_params['alpha'],
        normalize=True,
        degree=best_params['degree']
    )
    final_model.fit(X, y)
    
    # Финальные метрики
    y_pred = final_model.predict(X)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\nФинальные метрики:")
    print(f"  MAE: {mae:.0f} руб.")
    print(f"  RMSE: {rmse:.0f} руб.")
    print(f"  R²: {r2:.4f}")
    
    os.makedirs("resources", exist_ok=True)
    
    np.save("resources/model_weights.npy", final_model.weights)
    
    scaler_params = {
        'X_mean': final_model.X_mean,
        'X_std': final_model.X_std,
        'y_mean': final_model.y_mean,
        'y_std': final_model.y_std,
        'alpha': final_model.alpha,
        'degree': final_model.degree,
        'normalize': final_model.normalize
    }
    np.save("resources/scaler_params.npy", scaler_params)
    
    print("\nМодель сохранена в resources/")