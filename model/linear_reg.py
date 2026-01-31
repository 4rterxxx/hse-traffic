import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


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
        if self.degree <= 1:
            return X
        
        n_samples, n_features = X.shape
        poly_features = [X]
        
        if self.degree >= 2:
            poly_features.append(X ** 2)
        
        if self.degree >= 2:
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interaction = X[:, i] * X[:, j]
                    poly_features.append(interaction.reshape(-1, 1))
        
        return np.hstack(poly_features)
    
    def fit(self, X, y):
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


class SalaryModel:
    def __init__(self, model_type='ridge', use_log=True):
        self.model_type = model_type
        self.use_log = use_log
        self.scaler = StandardScaler()
        self.model = None
        
    def fit(self, X, y):
        if self.use_log:
            y = np.log1p(y)
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'ridge':
            best_score = -np.inf
            best_alpha = 1.0
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                model = Ridge(alpha=alpha, random_state=42)
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            
            self.model = Ridge(alpha=best_alpha, random_state=42)
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        if self.use_log:
            y_pred = np.expm1(y_pred)
        
        y_pred = np.clip(y_pred, 10000, 500000)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0