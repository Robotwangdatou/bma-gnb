# -*- coding: utf-8 -*-
"""
NLD-IGNB: Neighbor-Driven Local Parameter and Prior Correction-Fused Imbalanced GNB
@author: robot

[STANDARD VERSION - as provided by the author; reused as the LOCAL model L in BMA-GNB]
"""
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize, StandardScaler
from scipy.stats import entropy
# -------------------------- 1. Basic Utility Functions --------------------------
def g_mean_score(y_true, y_pred):
    """Calculate G-mean for imbalanced data evaluation"""
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = np.diag(cm) / np.sum(cm, axis=1)
    sensitivities = np.nan_to_num(sensitivities, nan=0.0)
    g_mean = np.sqrt(np.prod(sensitivities))
    return g_mean
def calculate_metrics(y_true, y_pred, y_prob, minority_class):
    """Calculate 5 core metrics: AUC, Gmean, Recall, Precision, F1"""
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Initialize base metrics
    auc = 0.0
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    g_mean = g_mean_score(y_true=y_true, y_pred=y_pred)

    # AUC calculation for binary/multi-class
    if n_classes == 2:
        pos_label = minority_class
        pos_idx = np.where(classes == pos_label)[0][0]
        auc = roc_auc_score(y_true, y_prob[:, pos_idx])
    else:
        y_bin = label_binarize(y_true, classes=classes)
        auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')

    return [auc, g_mean, recall, precision, f1]
def calculate_weighted_local_stats(knn_model, point, k, X_class, weight_by_distance=True):
    """Calculate weighted local median and variance for robust parameter estimation"""
    if knn_model is None or X_class is None or len(X_class) == 0:
        return None

    # Get k nearest neighbors
    distances, indices = knn_model.kneighbors(point.reshape(1, -1), n_neighbors=k)
    indices = indices[0]
    distances = distances[0]

    # Extract neighbor samples
    if isinstance(X_class, pd.DataFrame):
        neighbor_points = X_class.iloc[indices].values
    else:
        neighbor_points = X_class[indices]

    # Calculate distance-based weights
    if weight_by_distance and np.sum(distances) > 0:
        weights = 1 / (distances + 1e-8)
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(k) / k

    # Weighted median (outlier-resistant)
    weighted_median = np.zeros(neighbor_points.shape[1])
    for col_idx in range(neighbor_points.shape[1]):
        col_vals = neighbor_points[:, col_idx]
        sorted_idx = np.argsort(col_vals)
        sorted_vals = col_vals[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)
        weighted_median[col_idx] = sorted_vals[np.argmax(cum_weights >= 0.5)]

    # Weighted variance
    weighted_mean = np.average(neighbor_points, axis=0, weights=weights)
    weighted_var = np.average((neighbor_points - weighted_mean) ** 2, axis=0, weights=weights)

    return {'median': weighted_median, 'var': weighted_var}
def calculate_neighbor_class_dist(knn_all, x, k, train_y, global_classes):
    """Calculate local class distribution for prior correction """
    distances, neighbor_indices = knn_all.kneighbors(x.reshape(1, -1), n_neighbors=k)
    neighbor_indices = neighbor_indices[0]

    # Extract neighbor classes (仅训练集标签)
    if isinstance(train_y, pd.Series):
        neighbor_classes = train_y.iloc[neighbor_indices].values
    else:
        neighbor_classes = train_y[neighbor_indices]

    # Count classes based on global class list
    class_counts = np.zeros(len(global_classes))
    for i, cls in enumerate(global_classes):
        class_counts[i] = np.sum(neighbor_classes == cls)

    # Normalize distribution
    total = np.sum(class_counts)
    if total == 0:
        class_counts = np.ones(len(global_classes)) / len(global_classes)
    else:
        class_counts = class_counts / total

    return class_counts
def calculate_local_heterogeneity_entropy(neighbor_class_dist):
    """Calculate local heterogeneity entropy for adaptive alpha weighting"""
    neighbor_class_dist = neighbor_class_dist + 1e-12
    he = entropy(neighbor_class_dist)
    return he
# -------------------------- 2. Core NLD-IGNB Algorithm --------------------------
class IntegratedGNB:
    """
    NLD-IGNB: Neighbor-Driven Local Parameter and Prior Correction-Fused Imbalanced GNB

    """

    def __init__(self,
                 k_local_factor=0.0005,
                 k_prior_factor=0.001,
                 min_k=20, max_k=250,
                 weight_strength=1.0,
                 alpha_base=0.05,
                 he_max=0.8,
                 minority_threshold=0.5,
                 var_smoothing=1e-9):

        # Algorithm parameters
        self.k_local_factor = k_local_factor
        self.k_prior_factor = k_prior_factor
        self.min_k = min_k
        self.max_k = max_k
        self.weight_strength = weight_strength
        self.alpha_base = alpha_base
        self.he_max = he_max
        self.minority_threshold = minority_threshold
        self.var_smoothing = var_smoothing

        # Model state (initialized during training)
        self.classes_ = None
        self.class_to_idx_ = None
        self.global_prior_ = None
        self.global_median_ = None
        self.global_var_ = None
        self.class_knn_models_ = {}
        self.knn_all_ = None
        self.class_samples_ = {}
        self.train_y_ = None
        self.k_local_per_class_ = {}
        self.k_local_unified_ = None
        self.k_prior_opt_ = None
        self.minority_class = None
    def fit(self, X, y):
        """Train NLD-IGNB model with class-adaptive neighbor selection """
        X = np.asarray(X)
        y = np.asarray(y) if not isinstance(y, pd.Series) else y.values
        self.train_y_ = y

        # Initialize class structure
        self.classes_ = np.sort(np.unique(y))
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Identify minority class
        class_counts = np.zeros(n_classes)
        for i, cls in enumerate(self.classes_):
            class_counts[i] = np.sum(y == cls)
        self.minority_class = self.classes_[np.argmin(class_counts)]

        # Global prior probability (仅训练集)
        self.global_prior_ = class_counts / np.sum(class_counts)

        # Initialize global statistics and class-specific models
        self.global_median_ = np.zeros((n_classes, n_features))
        self.global_var_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.class_samples_[cls] = X_c
            n_c = len(X_c)

            # Global statistics for fallback
            self.global_median_[i] = np.median(X_c, axis=0)
            self.global_var_[i] = np.var(X_c, axis=0) + self.var_smoothing

            # Class-specific KNN models (仅训练集)
            if n_c >= self.min_k:
                self.class_knn_models_[cls] = NearestNeighbors(algorithm='ball_tree').fit(X_c)
            else:
                self.class_knn_models_[cls] = None

        # Global KNN for prior correction (仅训练集)
        self.knn_all_ = NearestNeighbors(algorithm='ball_tree').fit(X)

        # Class-adaptive neighbor counts (仅训练集)
        for cls in self.classes_:
            n_c = len(self.class_samples_[cls])
            k_c = max(self.min_k, min(int(n_c * self.k_local_factor), self.max_k))
            self.k_local_per_class_[cls] = k_c

        self.k_local_unified_ = min(self.k_local_per_class_.values())
        self.k_prior_opt_ = max(self.min_k, min(int(n_samples * self.k_prior_factor), self.max_k))

        print(f"Training completed | Classes: {n_classes} | k_prior: {self.k_prior_opt_}")
        return self
    def _get_local_params(self, x, c):
        """Get weighted local parameters for class c (仅用训练集)"""
        X_c = self.class_samples_.get(c, None)
        knn_model = self.class_knn_models_.get(c, None)
        c_idx = self.class_to_idx_[c]

        # Fallback to global parameters if insufficient samples
        if X_c is None or knn_model is None or len(X_c) < self.min_k:
            return self.global_median_[c_idx], self.global_var_[c_idx]

        # Class-adaptive local parameter estimation (仅训练集)
        k_local = self.k_local_per_class_[c]
        local_stats = calculate_weighted_local_stats(knn_model, x, k_local, X_c, True)
        mu = local_stats['median']
        sigma = local_stats['var'] + self.var_smoothing

        return mu, sigma
    def _calculate_adaptive_alpha(self, neighbor_class_dist):
        """Calculate HE-adaptive fusion weight alpha"""
        he = calculate_local_heterogeneity_entropy(neighbor_class_dist)
        he_clamped = np.clip(he, 0, self.he_max)
        alpha = self.alpha_base + 0.4 * (he_clamped / self.he_max)
        return np.clip(alpha, 0.5, 0.9)
    def _get_corrected_prior(self, x):
        """Fuse global and local priors with HE-adaptive weighting (仅用训练集)"""

        local_dist = calculate_neighbor_class_dist(
            self.knn_all_, x, self.k_prior_opt_, self.train_y_, self.classes_
        )

        # Adaptive alpha based on local heterogeneity
        alpha = self._calculate_adaptive_alpha(local_dist)
        corrected_prior = (self.global_prior_ * (1 - alpha)) + (local_dist * alpha)
        return corrected_prior / np.sum(corrected_prior)
    def predict_proba(self, X):
        """Predict posterior probabilities (完全隔离测试集)"""
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            x = X[i]

            corrected_prior = self._get_corrected_prior(x)

            log_cond_prob = np.zeros(n_classes)
            for j, cls in enumerate(self.classes_):

                mu, sigma = self._get_local_params(x, cls)
                log_p = -0.5 * (np.log(2 * np.pi) + np.log(sigma)) - 0.5 * ((x - mu) ** 2) / sigma
                log_cond_prob[j] = np.sum(log_p)

            # Numerical stability for posterior calculation
            log_posterior = log_cond_prob + np.log(corrected_prior + 1e-12)
            log_posterior -= np.max(log_posterior)
            proba[i] = np.exp(log_posterior) / np.sum(np.exp(log_posterior))

        return proba
    def predict(self, X):
        """Predict class labels with minority class thresholding"""
        proba = self.predict_proba(X)
        n_classes = len(self.classes_)

        if n_classes == 2:
            pos_idx = self.class_to_idx_[self.minority_class]
            predictions = np.where(
                proba[:, pos_idx] >= self.minority_threshold,
                self.minority_class,
                self.classes_[1 - pos_idx]
            )
        else:
            pred_indices = np.argmax(proba, axis=1)
            predictions = self.classes_[pred_indices]

        return predictions
# -------------------------- 3. Evaluation Framework --------------------------
if __name__ == '__main__':
    # Data loading with fallback to synthetic data
    try:
        dataset = pd.read_csv('ecoli-0-1-4-7_vs_5-6.csv')
        print(f"Data loaded | Shape: {dataset.shape}")
    except FileNotFoundError:
        print("File not found, using synthetic data")
        # 生成模拟数据（匹配Liver.csv的不平衡特征）
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=500, n_features=10, n_informative=8,
            n_redundant=2, n_classes=2, weights=[0.7, 0.3], random_state=42
        )
        dataset = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
        dataset['class'] = y
    # Prepare data
    y = dataset['class']
    X_original = dataset.drop(columns='class')
    minority_class = y.value_counts().idxmin()

    # 5-fold cross-validation
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = pd.DataFrame(columns=['Fold', 'AUC', 'Gmean', 'Recall', 'Precision', 'F1'])
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_original, y), 1):
        # Step 1: 拆分fold内的训练/测试集（严格隔离）
        X_train, X_test = X_original.iloc[train_idx], X_original.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        integrated_gnb = IntegratedGNB(
            k_local_factor=0.5,
            k_prior_factor=0.001,
            min_k=5, max_k=30,
            alpha_base=0.8,
            he_max=1.2,
            minority_threshold=0.5,
            var_smoothing=1e-8
        )

        integrated_gnb.fit(X_train_scaled, y_train)

        y_pred = integrated_gnb.predict(X_test_scaled)
        y_prob = integrated_gnb.predict_proba(X_test_scaled)
        metrics = calculate_metrics(y_test, y_pred, y_prob, minority_class)
        # Step 5: 存储结果
        fold_result = pd.DataFrame({
            'Fold': [fold], 'AUC': [metrics[0]], 'Gmean': [metrics[1]],
            'Recall': [metrics[2]], 'Precision': [metrics[3]], 'F1': [metrics[4]]
        })
        results = pd.concat([results, fold_result], ignore_index=True)
        print(f"Fold {fold}: AUC={metrics[0]:.4f}, F1={metrics[4]:.4f}")
    # Final results summary
    print("\n=== Cross-validation Results (无泄露版) ===")
    avg_metrics = results[['AUC', 'Gmean', 'Recall', 'Precision', 'F1']].mean()
    std_metrics = results[['AUC', 'Gmean', 'Recall', 'Precision', 'F1']].std()
    # ==========================================
    final_result = pd.DataFrame({
        'AUC': [round(avg_metrics['AUC'], 4)],
        'AUC_std': [round(std_metrics['AUC'], 4) if not pd.isna(std_metrics['AUC']) else 0.0000],

        'Gmean': [round(avg_metrics['Gmean'], 4)],
        'Gmean_std': [round(std_metrics['Gmean'], 4) if not pd.isna(std_metrics['Gmean']) else 0.0000],

        'Recall': [round(avg_metrics['Recall'], 4)],
        'Recall_std': [round(std_metrics['Recall'], 4) if not pd.isna(std_metrics['Recall']) else 0.0000],

        'Precision': [round(avg_metrics['Precision'], 4)],
        'Precision_std': [round(std_metrics['Precision'], 4) if not pd.isna(std_metrics['Precision']) else 0.0000],

        'F1': [round(avg_metrics['F1'], 4)],
        'F1_std': [round(std_metrics['F1'], 4) if not pd.isna(std_metrics['F1']) else 0.0000],
    })
    for metric in avg_metrics.index:
        print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
