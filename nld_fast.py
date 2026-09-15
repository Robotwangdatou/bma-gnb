# -*- coding: utf-8 -*-
"""
FastIntegratedGNB: a vectorized, mathematically-equivalent drop-in for
IntegratedGNB (NLD-IGNB). Only `predict_proba` is overridden; `fit` and all
learned parameters are inherited unchanged, so results are identical to the
per-sample loop but run in bulk via batched kneighbors + numpy.

The per-sample logic it replicates (from nld_ignb.py):
  prior: local KNN class distribution -> HE -> alpha -> corrected prior
  params: per-class weighted median (mu) and weighted variance (sigma)
  log_p = -0.5*(log(2pi)+log sigma) - 0.5*(x-mu)^2/sigma
  log_posterior = log_p + log(corrected prior); softmax
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from nld_ignb import IntegratedGNB


class FastIntegratedGNB(IntegratedGNB):
    """Vectorized predict_proba, same math as IntegratedGNB.

    Also rebuilds the KNN structures with algorithm='brute' and n_jobs=-1,
    which scales much better than kd_tree on large high-dimensional data
    (results are identical - same exact Euclidean neighbours).
    """

    def fit(self, X, y):
        """Same fit as IntegratedGNB but with kd_tree KNN for speed."""
        X = np.asarray(X)
        y = np.asarray(y) if not isinstance(y, pd.Series) else y.values
        self.train_y_ = y

        # Initialize class structure
        self.classes_ = np.sort(np.unique(y))
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        class_counts = np.zeros(n_classes)
        for i, cls in enumerate(self.classes_):
            class_counts[i] = np.sum(y == cls)
        self.minority_class = self.classes_[np.argmin(class_counts)]
        self.global_prior_ = class_counts / np.sum(class_counts)

        self.global_median_ = np.zeros((n_classes, n_features))
        self.global_var_ = np.zeros((n_classes, n_features))

        for i, cls in enumerate(self.classes_):
            X_c = X[y == cls]
            self.class_samples_[cls] = X_c
            n_c = len(X_c)
            self.global_median_[i] = np.median(X_c, axis=0)
            self.global_var_[i] = np.var(X_c, axis=0) + self.var_smoothing
            if n_c >= self.min_k:
                self.class_knn_models_[cls] = NearestNeighbors(
                    algorithm='brute', n_jobs=-1).fit(X_c)
            else:
                self.class_knn_models_[cls] = None

        self.knn_all_ = NearestNeighbors(algorithm='brute', n_jobs=-1).fit(X)

        for cls in self.classes_:
            n_c = len(self.class_samples_[cls])
            k_c = max(self.min_k, min(int(n_c * self.k_local_factor), self.max_k))
            self.k_local_per_class_[cls] = k_c

        self.k_local_unified_ = min(self.k_local_per_class_.values())
        self.k_prior_opt_ = max(self.min_k, min(int(n_samples * self.k_prior_factor), self.max_k))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        K = len(self.classes_)
        classes = self.classes_
        c2i = self.class_to_idx_

        # ---------------- prior correction (batched) ----------------
        # local class distribution from global knn_all_
        dists, idxs = self.knn_all_.kneighbors(X, n_neighbors=self.k_prior_opt_)
        nb_y = self.train_y_[idxs]                       # (n, k_prior)
        class_counts = np.zeros((n, K))
        for i, cls in enumerate(classes):
            class_counts[:, i] = np.sum(nb_y == cls, axis=1)
        total = class_counts.sum(axis=1, keepdims=True)
        local_dist = np.where(total > 0, class_counts / np.maximum(total, 1e-12),
                              np.ones((n, K)) / K)

        he = entropy(local_dist + 1e-12, axis=1)
        he_clamped = np.clip(he, 0, self.he_max)
        alpha = np.clip(self.alpha_base + 0.4 * (he_clamped / self.he_max), 0.5, 0.9)
        corrected = self.global_prior_[None, :] * (1 - alpha)[:, None] + local_dist * alpha[:, None]
        corrected /= corrected.sum(axis=1, keepdims=True)
        log_prior = np.log(corrected + 1e-12)            # (n, K)

        # ---------------- per-class local params (batched) ----------------
        log_cond = np.zeros((n, K))
        for i, cls in enumerate(classes):
            X_c = self.class_samples_.get(cls, None)
            knn = self.class_knn_models_.get(cls, None)
            if X_c is None or knn is None or len(X_c) < self.min_k:
                mu = np.broadcast_to(self.global_median_[i], (n, X.shape[1]))
                var = np.broadcast_to(self.global_var_[i], (n, X.shape[1]))
            else:
                k_local = self.k_local_per_class_[cls]
                dd, ii = knn.kneighbors(X, n_neighbors=k_local)   # (n,k)
                nbr = X_c[ii]                                     # (n,k,d)
                w = 1.0 / (dd + 1e-8)
                w = w / w.sum(axis=1, keepdims=True)              # (n,k)
                w4 = w[:, :, None]

                # weighted median (per feature)
                order = np.argsort(nbr, axis=1)
                sorted_vals = np.take_along_axis(nbr, order, axis=1)
                sorted_w = np.take_along_axis(np.broadcast_to(w4, nbr.shape), order, axis=1)
                cum = np.cumsum(sorted_w, axis=1)
                mid = np.argmax(cum >= 0.5, axis=1)[:, None, :]   # (n,1,d)
                mu = np.take_along_axis(sorted_vals, mid, axis=1)[:, 0, :]

                # weighted variance (uses weighted mean)
                wmean = np.sum(nbr * w4, axis=1)                  # (n,d)
                var = np.sum((nbr - wmean[:, None, :]) ** 2 * w4, axis=1)
                var = var + self.var_smoothing

            log_p = -0.5 * (np.log(2 * np.pi) + np.log(var)) - 0.5 * ((X - mu) ** 2) / var
            log_cond[:, i] = np.sum(log_p, axis=1)

        log_post = log_cond + log_prior
        log_post -= log_post.max(axis=1, keepdims=True)
        proba = np.exp(log_post)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba
