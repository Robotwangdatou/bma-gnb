# -*- coding: utf-8 -*-
"""
BMA-GNB: Global-Local Gaussian Naive Bayes with a Learnable Logistic Gate
============================================================================

Framework (see Method theory draft):
  - Global model G : standard Gaussian Naive Bayes  -> P_G(c|x)
  - Local model  L : NLD-IGNB local-parameter module -> P_L(c|x)
                    (reused from nld_ignb.py, only the local-parameter part)
  - Gate: w_L(x) = sigmoid(phi^T z(x)), z(x) = [disagreement, local_density, global_entropy]
  - Mixture: P(c|x) = (1 - w_L(x)) * P_G(c|x) + w_L(x) * P_L(c|x)

Gate supervision: on a held-out validation set, the logistic gate learns
"when should we trust the local model", i.e. the binary target is whether
the local model L wins the prediction on the current sample. Samples where
G and L agree are excluded (the weight does not affect the outcome there).

Author: (user) / implementation helper
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from nld_ignb import IntegratedGNB, calculate_metrics
from nld_fast import FastIntegratedGNB


class BMA_GateGNB:
    """Global-Local GNB combined by a pointwise learnable logistic gate.

    Parameters
    ----------
    local_kwargs : dict
        Keyword arguments forwarded to IntegratedGNB (the local model L).
        Defaults match the NLD-IGNB evaluation setup.
    gate_C : float
        Inverse regularisation strength of the logistic gate (smaller = more
        regularisation). The gate has only 3 features, so keep it moderate.
    density_k : int
        Number of neighbours used to compute the local-density and
        neighbour-class-entropy gate features.
    """

    def __init__(self, local_kwargs=None, gate_C=1.0, density_k=10, gate_clf=None,
                 local_cls=None, local_model=None, gate_label_mode='binary',
                 gate_features='all', gate_wcap=None, gate_tau0=None,
                 global_model=None):
        # Aligned with the STANDARD NLD-IGNB configuration used in nld_ignb.py main
        defaults = dict(k_local_factor=0.5, k_prior_factor=0.001,
                        min_k=5, max_k=30, weight_strength=1.0,
                        alpha_base=0.8, he_max=1.2,
                        minority_threshold=0.5, var_smoothing=1e-8)
        if local_kwargs:
            defaults.update(local_kwargs)
        self.local_kwargs = defaults

        # --- model G : standard GNB (sklearn) by default; any sklearn
        #     classifier exposing predict_proba can be injected via
        #     global_model (used by the generality experiments) ---
        self.G_ = global_model if global_model is not None else GaussianNB()
        # --- model L : NLD-IGNB local-parameter module ---
        # local_cls: FastIntegratedGNB (vectorized, identical math) by default
        # for speed; pass IntegratedGNB to use the exact per-sample loop.
        # local_model: a pre-fitted local model to share with the NLD baseline
        # (avoids training + predicting the local model twice per fold).
        if local_model is not None:
            self.L_ = local_model
        else:
            self._local_cls = local_cls if local_cls is not None else FastIntegratedGNB
            self.L_ = self._local_cls(**defaults)

        # --- gate state ---
        self.gate_ = None
        self.gate_scaler_ = None
        self.constant_wL_ = 0.5
        self.constant_wL_min_ = None   # per-class fallback: L reliability on minority
        self.constant_wL_maj_ = None   # per-class fallback: L reliability on majority
        self.minority_cls_ = None
        self.knn_ = None
        self.density_k = density_k
        self.y_tr_ = None
        self.classes_ = None
        self.gate_C = gate_C
        self.gate_clf = gate_clf
        # gate supervision mode:
        #   'binary'        : y = 1[L correct on disagreement] (current baseline)
        #   'class_weighted': same binary y, but minority-class disagreement
        #                     samples are up-weighted so the gate does not learn
        #                     "globally distrust L" merely because L loses on
        #                     abundant majority samples (mechanism C).
        #   'conf_weighted' : continuous supervision, weight = L confidence on
        #                     correct samples (y = 1[correct] x conf_L), so the
        #                     gate keeps trusting L when L is right AND confident
        #                     even if L is weak overall (mechanisms B / C).
        #   'both'          : class_weighted x conf_weighted.
        #   'conf_split'    : soft-label continuous supervision. Each sample
        #                     where L is correct contributes TWO rows with
        #                     weights conf and (1-conf) labelled 1 and 0, i.e.
        #                     the exact cross-entropy of the continuous target
        #                     y = 1[L correct] x conf_L. Strong-confidence wins
        #                     dominate, low-confidence wins carry a soft
        #                     negative signal (mechanisms B / C).
        #   'class2x'       : mild class re-weighting - minority-class
        #                     disagreements get weight 2 (not full balancing,
        #                     which destroyed creditcard by swamping the gate
        #                     with minority samples).
        #   'conf_gate2x'   : mechanism-C targeted: up-weight ONLY minority
        #                     disagreement samples where L is highly confident
        #                     (conf_L >= 0.8); other minority disagreements
        #                     get 1.2, majority stay 1.0. This fixes "L weak
        #                     overall but confident on the minority" without
        #                     touching majority-dominated decisions.
        self.gate_label_mode = gate_label_mode
        # feature ablation mask over z = [disagreement, local_density, global_entropy]
        self.gate_features = gate_features
        self.gate_mask_ = np.array({'all': [1, 1, 1], 'no_delta': [0, 1, 1],
                                    'no_rho': [1, 0, 1], 'no_H': [1, 1, 0],
                                    'delta': [1, 0, 0], 'rho': [0, 1, 0],
                                    'H': [0, 0, 1], 'dr': [1, 1, 0],
                                    'dH': [1, 0, 1], 'rH': [0, 1, 1]}[gate_features],
                                   dtype=bool)
        # theory-driven remedies (Theorem 2 convex-mix bound / Theorem 3
        # separation condition). None = disabled (final paper model unchanged).
        self.gate_wcap = gate_wcap          # hard cap w_L <= wcap ("never abandon G")
        self.gate_tau0 = gate_tau0          # tau-shrink: w<-0.5+(w-0.5)*min(1,tau_hat/tau0)
        self.tau_hat_ = None                # estimated on validation set, no leakage

    # ------------------------------------------------------------------
    # Feature extraction: z(x) = [disagreement, local_density, global_entropy]
    # ------------------------------------------------------------------
    def _gate_features(self, X, PL=None, PG=None):
        """Compute the three gate features for each row of X.

        disagreement   : total-variation distance between P_G(·|x) and P_L(·|x)
                         in [0, 1]; large when the two models strongly conflict.
        local_density  : 1 / (1 + mean distance to the k nearest training points);
                         low in sparse regions where the local model is unreliable.
        global_entropy : entropy of the class distribution among the k nearest
                         training points (same quantity as the HE of NLD-IGNB),
                         reflecting local class heterogeneity.
        """
        PG = self.G_.predict_proba(X) if PG is None else PG
        # reuse precomputed PL when available (avoids re-running the expensive
        # local model twice per call)
        PL = self.L_.predict_proba(X) if PL is None else PL

        # 1) disagreement = total-variation distance
        disagreement = 0.5 * np.abs(PG - PL).sum(axis=1)

        # 2) local density from global KNN
        dist, idx = self.knn_.kneighbors(X)    # (n, k), (n, k)
        mean_dist = dist.mean(axis=1)
        local_density = 1.0 / (1.0 + mean_dist)

        # 3) entropy of the neighbour class distribution (vectorized)
        neigh_labels = self.y_tr_[idx]         # (n, k)
        n_classes = len(self.classes_)
        counts = np.zeros((X.shape[0], n_classes))
        for c in range(n_classes):
            counts[:, c] = np.sum(neigh_labels == c, axis=1)
        row_sum = counts.sum(axis=1, keepdims=True)
        probs = np.where(row_sum > 0, counts / np.maximum(row_sum, 1e-12),
                         np.ones((X.shape[0], n_classes)) / n_classes)
        global_entropy = entropy(probs + 1e-12, axis=1)

        Z = np.column_stack([disagreement, local_density, global_entropy])
        return Z

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X_tr, y_tr, X_val, y_val, PL_val=None):
        """Fit model G, model L and the logistic gate.

        Parameters
        ----------
        X_tr, y_tr : training split (used to fit G and L, and to build KNN)
        X_val, y_val : validation split (used to learn the gate phi)
        PL_val : optional precomputed L probabilities on X_val (shared across
                 gate variants to avoid recomputing the expensive local model)
        """
        X_tr = np.asarray(X_tr); y_tr = np.asarray(y_tr)
        X_val = np.asarray(X_val); y_val = np.asarray(y_val)

        self.y_tr_ = y_tr
        self.classes_ = np.sort(np.unique(y_tr))

        # 1) train global GNB
        self.G_.fit(X_tr, y_tr)
        # 2) train local NLD-IGNB (local-parameter module) unless one was
        #    passed in already fitted (shared with the NLD baseline to avoid
        #    paying the expensive local fit + predict twice per fold)
        if getattr(self.L_, 'classes_', None) is None:
            self.L_.fit(X_tr, y_tr)
        # KNN on training set for the gate features
        k = min(self.density_k, len(X_tr))
        self.knn_ = NearestNeighbors(n_neighbors=k, algorithm='brute', n_jobs=-1).fit(X_tr)

        # 3) build gate features AND derive the labels from the SAME
        #    G / L probability matrices (no second full local prediction)
        PG_val = self.G_.predict_proba(X_val)
        if PL_val is None:
            PL_val = self.L_.predict_proba(X_val)
        Z_val = self._gate_features(X_val, PL=PL_val, PG=PG_val)
        yG = self._hard_labels(PG_val, self.G_)
        yL = self._hard_labels(PL_val, self.L_)
        disagree_mask = yG != yL
        t = (yL[disagree_mask] == y_val[disagree_mask]).astype(int)

        # standardise the masked gate features, then fit the gate
        self.gate_scaler_ = StandardScaler().fit(Z_val[:, self.gate_mask_])
        Zs = self.gate_scaler_.transform(Z_val[:, self.gate_mask_])

        # Guard: if too few disagreeing samples or a single class remains,
        # fall back to per-class constant weights estimated on the FULL
        # validation set (not just the disagreeing samples). The old global
        # constant (L-win rate on disagreements) systematically under-weighted
        # the minority class, diluting high-confidence local predictions below
        # the 0.5 threshold (e.g. car-good). We instead estimate L's reliability
        # separately for minority / majority regions and blend by L's own
        # minority probability at test time.
        n_dis = int(disagree_mask.sum())
        if n_dis >= 20 and len(np.unique(t)) == 2:
            if self.gate_clf is not None:
                self.gate_ = clone(self.gate_clf)
            else:
                self.gate_ = LogisticRegression(C=self.gate_C, max_iter=2000)
            sw = None
            if self.gate_label_mode in ('conf_reg', 'sign_reg'):
                # TRUE continuous supervision (mechanisms B/C): the target is
                # not a binary "L wins" but L's confidence gated by correctness.
                #   conf_reg  : y = P_L(c*|x) x 1[L correct]   in [0, 1]
                #   sign_reg  : y = P_L(c*|x) x (2*1[L correct]-1)  in [-1, 1]
                # A Ridge regressor on the three gate features learns the
                # confidence-weighted correctness surface; sign_reg keeps the
                # "high-confidence WRONG" signal (negative y) so the gate can
                # learn to distrust L exactly where it is confidently wrong.
                from sklearn.linear_model import Ridge
                conf = PL_val[disagree_mask, yL[disagree_mask]]
                y_cont = conf * t if self.gate_label_mode == 'conf_reg' \
                    else conf * (2 * t - 1)
                self.gate_ = Ridge(alpha=1.0).fit(Zs[disagree_mask], y_cont)
            elif self.gate_label_mode == 'conf_split':
                # soft-label continuous supervision (exact CE of y = conf x 1[L ok])
                conf = PL_val[disagree_mask, yL[disagree_mask]]
                neg = ~t.astype(bool)
                Zpos = Zs[disagree_mask][t == 1]
                Zneg = Zs[disagree_mask][neg]
                X_fit = np.vstack([Zneg, Zpos, Zpos])
                y_fit = np.concatenate([np.zeros(int(neg.sum())),
                                        np.ones(int((~neg).sum())),
                                        np.zeros(int((~neg).sum()))])
                w_fit = np.concatenate([np.ones(int(neg.sum())),
                                        conf[t == 1],
                                        1.0 - conf[t == 1]])
                self.gate_.fit(X_fit, y_fit, sample_weight=w_fit)
            else:
                if self.gate_label_mode in ('class_weighted', 'both'):
                    # up-weight minority-class disagreement samples so the gate
                    # learns "L wins on minority" instead of being swamped by
                    # majority disagreements where L may lose often
                    yd = y_val[disagree_mask]
                    cls = list(self.classes_)
                    cls_counts = np.array([int((yd == c).sum()) for c in cls])
                    # balanced weights: each class contributes equal total weight
                    inv = len(yd) / (len(cls) * np.maximum(cls_counts, 1))
                    sw = np.array([inv[cls.index(c)] for c in yd])
                if self.gate_label_mode == 'class2x':
                    yd = y_val[disagree_mask]
                    cls = list(self.classes_)
                    cls_counts = np.array([int((yd == c).sum()) for c in cls])
                    m_idx = int(np.argmin(cls_counts))
                    sw = np.where(yd == cls[m_idx], 2.0, 1.0)
                if self.gate_label_mode == 'conf_gate2x':
                    yd = y_val[disagree_mask]
                    cls = list(self.classes_)
                    cls_counts = np.array([int((yd == c).sum()) for c in cls])
                    m_idx = int(np.argmin(cls_counts))
                    conf = PL_val[disagree_mask, yL[disagree_mask]]
                    sw = np.ones(n_dis)
                    is_min = yd == cls[m_idx]
                    hi_conf = conf >= 0.8
                    sw[is_min & hi_conf] = 2.5
                    sw[is_min & ~hi_conf] = 1.2
                if self.gate_label_mode in ('conf_weighted', 'both'):
                    # continuous supervision: weight = L confidence on samples L
                    # predicts correctly (y = 1[correct] x conf_L); incorrect
                    # samples keep weight 1 so the negative signal is preserved
                    conf = PL_val[disagree_mask, yL[disagree_mask]]
                    cw = np.where(t == 1, conf, 1.0)
                    sw = cw if sw is None else sw * cw
                if sw is not None:
                    self.gate_.fit(Zs[disagree_mask], t, sample_weight=sw)
                else:
                    self.gate_.fit(Zs[disagree_mask], t)
            self.constant_wL_ = None
            self.constant_wL_min_ = None
            self.constant_wL_maj_ = None
        else:
            self.gate_ = None
            self.constant_wL_ = None
            yv = np.asarray(y_val)
            cls_counts = [int((yv == c).sum()) for c in self.classes_]
            minority_idx = int(np.argmin(cls_counts))
            self.minority_cls_ = self.classes_[minority_idx]
            maj_idx = 1 - minority_idx
            yL_val = self._hard_labels(PL_val, self.L_)

            # Per-class reliability = L's precision on the samples L assigns to
            # that class (i.e. "when L says class c, how often is it right?").
            # Precision (not accuracy) is the right quantity for a mixing weight:
            # a sample that L labels as minority with high confidence is exactly
            # the region where w should be high. Using accuracy on the minority
            # class is unstable when the validation set has only a handful of
            # minority samples (poker: acc_min drops to 0.0 on 3-5 samples and
            # the minority gets diluted to death by G).
            def cls_precision(pred, true, cls):
                m = pred == cls
                if m.sum() == 0:
                    return 0.5
                return float((true[m] == cls).mean())

            w_min = cls_precision(yL_val, yv, self.minority_cls_)
            w_maj = cls_precision(yL_val, yv, self.classes_[maj_idx])
            # Floor at 0.5: in the fallback regime we never have enough data to
            # justify letting G unilaterally dilute a class (G's minority
            # likelihood is systematically suppressed by the global prior).
            self.constant_wL_min_ = float(np.clip(w_min, 0.5, 1.0))
            self.constant_wL_maj_ = float(np.clip(w_maj, 0.5, 1.0))
        # Theory-driven gate regularization (Theorem 3 separation; validation
        # labels only, no test leakage). tau_hat = separation of the LEARNED
        # gate on the validation set: min(E_B - lambda_bar, lambda_bar - E_A)
        # with A = {L right, G wrong}, B = {G right, L wrong}. tau_hat ~ 0
        # means the gate cannot separate even on its own training data ->
        # shrink activation toward the safe 0.5 mix (Theorem 2 convex bound).
        if self.gate_tau0 is not None:
            w_val = self._gate_weight(X_val, PL=PL_val)  # tau_hat_ unset -> raw
            yG_val = self._hard_labels(PG_val, self.G_)
            yL_val = self._hard_labels(PL_val, self.L_)
            A = (yG_val != y_val) & (yL_val == y_val)
            B = (yG_val == y_val) & (yL_val != y_val)
            lam_bar = float(w_val.mean())
            E_A = float(w_val[A].mean()) if A.sum() > 0 else np.nan
            E_B = float(w_val[B].mean()) if B.sum() > 0 else np.nan
            if (not np.isnan(E_A)) and (not np.isnan(E_B)):
                self.tau_hat_ = max(0.0, float(min(E_B - lam_bar, lam_bar - E_A)))
            else:
                self.tau_hat_ = 0.0
        return self

    @staticmethod
    def _hard_labels(proba, model):
        """Derive hard labels from a probability matrix, matching the model's
        own predict() decision rule (minority threshold for binary NLD)."""
        classes = model.classes_
        if len(classes) == 2:
            # NLD-style: threshold on the minority class (if present)
            mcl = getattr(model, 'minority_class', None)
            thr = getattr(model, 'minority_threshold', 0.5)
            if mcl is not None:
                pos_idx = np.where(classes == mcl)[0][0]
                return np.where(proba[:, pos_idx] >= thr, mcl, classes[1 - pos_idx])
        return classes[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _gate_weight(self, X, PL=None):
        """Pointwise local-model weight w_L(x) = sigmoid(phi^T z(x))."""
        if self.gate_ is None:
            # per-class fallback: blend majority/minority constants by L's own
            # minority probability (minority regions keep a higher w when L is
            # reliable there, without diluting high-confidence L outputs)
            if getattr(self, 'constant_wL_min_', None) is not None:
                if PL is None:
                    PL = self.L_.predict_proba(X)
                m_idx = list(self.classes_).index(self.minority_cls_)
                p_min = PL[:, m_idx]
                w = self.constant_wL_maj_ * (1.0 - p_min) + self.constant_wL_min_ * p_min
            else:
                w = np.full(X.shape[0], self.constant_wL_)
        else:
            Z = self.gate_scaler_.transform(self._gate_features(X, PL=PL)[:, self.gate_mask_])
            if hasattr(self.gate_, 'predict_proba'):
                # P(L wins | z) is exactly the sigmoid output of the logistic gate
                w = self.gate_.predict_proba(Z)[:, 1]
            else:
                # continuous-supervision gates: ridge regression, clip to a weight
                raw = self.gate_.predict(Z)
                if self.gate_label_mode == 'sign_reg':
                    w = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
                else:
                    w = np.clip(raw, 0.0, 1.0)
        # theory-driven remedies applied to every gate regime
        if self.gate_wcap is not None:
            w = np.clip(w, 0.0, self.gate_wcap)
        if self.gate_tau0 is not None and getattr(self, 'tau_hat_', None) is not None:
            r = min(1.0, self.tau_hat_ / self.gate_tau0)
            w = 0.5 + (w - 0.5) * r
        return w

    def predict_proba(self, X, PL=None):
        """Mixture posterior P(c|x) = (1-w_L)P_G(c|x) + w_L P_L(c|x)."""
        PG = self.G_.predict_proba(X)
        if PL is None:
            PL = self.L_.predict_proba(X)          # computed once, shared
        w = self._gate_weight(X, PL=PL)[:, None]
        return (1.0 - w) * PG + w * PL

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ======================================================================
# Demo / smoke test with synthetic imbalanced + locally heterogeneous data
# (5-fold CV). Replace with the real 39 datasets when running the actual
# experiments; this block only proves the pipeline runs and the gate acts.
# ======================================================================
if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.datasets import make_classification

    # --- data loading: try the standard ecoli csv, else synthetic ---
    try:
        dataset = pd.read_csv('ecoli-0-1-4-7_vs_5-6.csv')
        print(f"Data loaded | Shape: {dataset.shape}")
        X = dataset.drop(columns='class').to_numpy()
        y = dataset['class'].to_numpy()
    except FileNotFoundError:
        print("ecoli csv not found, using synthetic data")
        rng = 42
        # Challenging setup: strong class overlap + label noise, so the global
        # GNB and the local NLD-IGNB genuinely disagree on part of the space.
        X, y = make_classification(
            n_samples=1200, n_features=12, n_informative=8, n_redundant=2,
            n_clusters_per_class=2, weights=[0.80, 0.20], flip_y=0.12,
            class_sep=0.75, random_state=rng)
    rng = 42
    minority = pd.Series(y).value_counts().idxmin()

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    rows, gate_info = [], []
    for fold, (tr_idx, te_idx) in enumerate(kfold.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=rng)

        # standard scaler inside each fold (train-only fit, no leakage)
        scaler = StandardScaler().fit(X_tr2)
        X_tr2s, X_vals, X_tes = scaler.transform(X_tr2), scaler.transform(X_val), scaler.transform(X_te)

        # --- baseline: standard GNB ---
        g = GaussianNB().fit(X_tr2s, y_tr2)
        m_g = calculate_metrics(y_te, g.predict(X_tes), g.predict_proba(X_tes), minority)

        # --- baseline: NLD-IGNB (standard config) ---
        nld = IntegratedGNB(k_local_factor=0.5, k_prior_factor=0.001,
                            min_k=5, max_k=30, alpha_base=0.8, he_max=1.2,
                            var_smoothing=1e-8)
        nld.fit(X_tr2s, y_tr2)
        m_n = calculate_metrics(y_te, nld.predict(X_tes), nld.predict_proba(X_tes), minority)

        # --- proposed: BMA-GNB ---
        bma = BMA_GateGNB(gate_C=1.0, density_k=10)
        bma.fit(X_tr2s, y_tr2, X_vals, y_val)
        m_b = calculate_metrics(y_te, bma.predict(X_tes), bma.predict_proba(X_tes), minority)

        # diagnostic: did the gate act? (learned logistic gate vs constant fallback)
        w_test = bma._gate_weight(X_tes)
        if bma.gate_ is not None:
            gate_mode = f"logistic[{bma.gate_label_mode}]"
        elif getattr(bma, 'constant_wL_min_', None) is not None:
            gate_mode = f"const_min={bma.constant_wL_min_:.2f}/maj={bma.constant_wL_maj_:.2f}"
        else:
            gate_mode = f"const={bma.constant_wL_:.2f}"
        gate_info.append(gate_mode)
        rows.append([fold] + m_g + m_n + m_b)
        print(f"Fold {fold}: GNB F1={m_g[4]:.3f} | NLD F1={m_n[4]:.3f} | BMA F1={m_b[4]:.3f} "
              f"| gate={gate_mode} | w_L range=[{w_test.min():.2f},{w_test.max():.2f}]")

    cols = ['Fold',
            'G_AUC', 'G_Gmean', 'G_Recall', 'G_Prec', 'G_F1',
            'N_AUC', 'N_Gmean', 'N_Recall', 'N_Prec', 'N_F1',
            'B_AUC', 'B_Gmean', 'B_Recall', 'B_Prec', 'B_F1']
    res = pd.DataFrame(rows, columns=cols)
    print("\n=== 5-fold CV (AUC / Gmean / F1) ===")
    for tag, idx in [('GNB', slice(1, 6)), ('NLD', slice(6, 11)), ('BMA', slice(11, 16))]:
        sub = res.iloc[:, idx]
        print(f"{tag:>4}: AUC {sub.iloc[:,0].mean():.4f} | Gmean {sub.iloc[:,1].mean():.4f} | F1 {sub.iloc[:,4].mean():.4f}")
    print("Gate modes per fold:", gate_info)

