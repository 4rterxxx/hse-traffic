import numpy as np


class LinearRegression:
    def __init__(self, alpha=1.0, normalize=True, degree=1):
        self.alpha = alpha
        self.normalize = normalize
        self.degree = degree
        self.weights = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def _add_polynomial_features(self, X):
        """Добавляем полиномиальные признаки"""
        if self.degree <= 1:
            return X
        
        n_samples, n_features = X.shape
        poly_features = [X]
        
        # Квадраты признаков
        if self.degree >= 2:
            poly_features.append(X ** 2)
        
        # Взаимодействия признаков
        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction = X[:, i] * X[:, j]
                    poly_features.append(interaction.reshape(-1, 1))
        
        return np.hstack(poly_features)
    
    def fit(self, X, y):
        # Добавляем полиномиальные признаки
        X_poly = self._add_polynomial_features(X)
        
        if self.normalize:
            self.X_mean = np.mean(X_poly, axis=0)
            self.X_std = np.std(X_poly, axis=0)
            self.X_std[self.X_std == 0] = 1
            X_norm = (X_poly - self.X_mean) / self.X_std
        else:
            X_norm = X_poly.copy()
            self.X_mean = np.zeros(X_poly.shape[1])
            self.X_std = np.ones(X_poly.shape[1])
        
        # Нормализуем y
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std == 0:
            self.y_std = 1
        y_norm = (y - self.y_mean) / self.y_std
        
        X_with_bias = np.c_[np.ones(X_norm.shape[0]), X_norm]
        
        n_features = X_with_bias.shape[1]
        XTX = X_with_bias.T @ X_with_bias
        XTX_reg = XTX + self.alpha * np.eye(n_features)
        XTy = X_with_bias.T @ y_norm
        
        try:
            self.weights = np.linalg.solve(XTX_reg, XTy)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(XTX_reg) @ XTy
        
        return self
    
    def predict(self, X, denormalize=True):
        if self.weights is None:
            raise ValueError("Модель не обучена")
        
        # Добавляем те же полиномиальные признаки
        X_poly = self._add_polynomial_features(X)
        
        if self.normalize:
            X_norm = (X_poly - self.X_mean) / self.X_std
        else:
            X_norm = X_poly.copy()
        
        X_with_bias = np.c_[np.ones(X_norm.shape[0]), X_norm]
        y_pred_norm = X_with_bias @ self.weights
        
        if denormalize:
            y_pred = y_pred_norm * self.y_std + self.y_mean
            y_pred = np.clip(y_pred, 10000, 500000)
            return y_pred
        else:
            return y_pred_norm
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)


class SalaryModel(LinearRegression):
    def __init__(self, alpha=1.0, normalize=True, degree=1):
        super().__init__(alpha, normalize, degree)
    
    # Можно добавить специфичные для зарплат методы, если нужно
    pass