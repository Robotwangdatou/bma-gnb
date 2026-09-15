# -*- coding: utf-8 -*-
"""
Batch comparison: GNB vs NLD-IGNB vs BMA-GNB on the 39 imbalanced datasets.
Follows the standard NLD-IGNB evaluation protocol (5-fold StratifiedCV,
StandardScaler fit on train only, binary metrics, minority as positive).
"""
import sys, os, glob, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from nld_ignb import IntegratedGNB, calculate_metrics
from nld_fast import FastIntegratedGNB
from bma_gnb import BMA_GateGNB

# Vectorized drop-in, mathematically identical to IntegratedGNB (verified).
_LOCAL_CLS = FastIntegratedGNB

# Data directory: defaults to ./dataset (self-contained repo); override via env
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset'))
# Output dir + gate supervision mode overridable via env (keeps versions apart)
GATE_MODE = os.environ.get('GATE_MODE', 'binary')
OUT_DIR = os.environ.get('OUT_DIR', r"C:\Users\robot\Doubao\chats\2026-09-05\new-chat\nld_ignb_bma\code\results")
os.makedirs(OUT_DIR, exist_ok=True)

# Standard NLD-IGNB config (as in nld_ignb.py main)
NLD_KW = dict(k_local_factor=0.5, k_prior_factor=0.001, min_k=5, max_k=30,
              alpha_base=0.8, he_max=1.2, var_smoothing=1e-8)
N_SPLITS = 5
VAL_FRAC = 0.25
METRICS = ['AUC', 'Gmean', 'Recall', 'Precision', 'F1']

# Datasets whose per-sample local KNN is computationally prohibitive at full
# size (28万+ samples x high dim). We run them on a stratified subsample to
# keep the runtime sane; results are marked as approximate.
MAX_SAMPLES = 60000
APPROX_DATASETS = {'covtype_4vs_2', 'creditcard'}


def load_data(path, name):
    df = pd.read_csv(path)
    y = df['class'].to_numpy()
    X = df.drop(columns='class').to_numpy()
    approx = False
    if name in APPROX_DATASETS and len(y) > MAX_SAMPLES:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_SAMPLES,
                                     random_state=42)
        idx, _ = next(iter(sss.split(X, y)))
        X, y = X[idx], y[idx]
        approx = True
    return X, y, approx


def run_one_dataset(name, X, y):
    """Return DataFrame with per-fold rows for the 3 models."""
    rng = 42
    minority = pd.Series(y).value_counts().idxmin()
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rng)
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(kfold.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=rng)
        sc = StandardScaler().fit(X_tr2)
        X_tr2s = sc.transform(X_tr2)
        X_vals = sc.transform(X_val)
        X_tes = sc.transform(X_te)

        # GNB (predict_proba once, derive hard labels by argmax)
        g = GaussianNB().fit(X_tr2s, y_tr2)
        g_prob = g.predict_proba(X_tes)
        g_pred = g.classes_[np.argmax(g_prob, axis=1)]
        mg = calculate_metrics(y_te, g_pred, g_prob, minority)

        # NLD-IGNB (train the local model ONCE per fold; BMA shares it)
        nld = _LOCAL_CLS(**NLD_KW)
        nld.fit(X_tr2s, y_tr2)
        n_prob = nld.predict_proba(X_tes)
        if len(nld.classes_) == 2:
            pos_idx = nld.class_to_idx_[nld.minority_class]
            n_pred = np.where(n_prob[:, pos_idx] >= nld.minority_threshold,
                              nld.minority_class, nld.classes_[1 - pos_idx])
        else:
            n_pred = nld.classes_[np.argmax(n_prob, axis=1)]
        mn = calculate_metrics(y_te, n_pred, n_prob, minority)

        # BMA-GNB (reuses the same fitted local model L)
        bma = BMA_GateGNB(gate_C=1.0, density_k=10, local_cls=_LOCAL_CLS,
                          local_model=nld, gate_label_mode=GATE_MODE)
        bma.fit(X_tr2s, y_tr2, X_vals, y_val)
        b_prob = bma.predict_proba(X_tes)
        b_pred = bma.classes_[np.argmax(b_prob, axis=1)]
        mb = calculate_metrics(y_te, b_pred, b_prob, minority)

        rows.append({'dataset': name, 'fold': fold,
                     'G_AUC': mg[0], 'G_Gmean': mg[1], 'G_Recall': mg[2], 'G_Prec': mg[3], 'G_F1': mg[4],
                     'N_AUC': mn[0], 'N_Gmean': mn[1], 'N_Recall': mn[2], 'N_Prec': mn[3], 'N_F1': mn[4],
                     'B_AUC': mb[0], 'B_Gmean': mb[1], 'B_Recall': mb[2], 'B_Prec': mb[3], 'B_F1': mb[4]})
    return pd.DataFrame(rows)


def main(only=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if only:
        files = [f for f in files if os.path.basename(f) in only]
    print(f"Running on {len(files)} datasets", flush=True)

    all_rows = []
    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        t0 = time.time()
        try:
            X, y, approx = load_data(f, name)
            df = run_one_dataset(name, X, y)
            if approx:
                df['approx'] = True
            else:
                df['approx'] = False
            all_rows.append(df)
            # quick summary
            m = df[['G_F1', 'N_F1', 'B_F1']].mean()
            tag = ' [approx]' if approx else ''
            print(f"[{i:2d}/{len(files)}] {name:28s} n={len(y):6d}{tag} "
                  f"G_F1={m['G_F1']:.3f} N_F1={m['N_F1']:.3f} B_F1={m['B_F1']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{i:2d}/{len(files)}] {name:28s} ERROR: {e}", flush=True)

    if not all_rows:
        print("No results."); return
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(os.path.join(OUT_DIR, 'per_fold_all.csv'), index=False)

    # --- aggregate: mean over folds per dataset ---
    keys = ['dataset']
    groups = res.groupby('dataset')
    colmap = {'AUC': 'AUC', 'Gmean': 'Gmean', 'Recall': 'Recall',
              'Precision': 'Prec', 'F1': 'F1'}
    agg = groups[['G_AUC', 'N_AUC', 'B_AUC']].mean().round(4)
    for mt in ['Gmean', 'Recall', 'Precision', 'F1']:
        suf = colmap[mt]
        agg[f'G_{mt}'] = groups[f'G_{suf}'].mean().round(4)
        agg[f'N_{mt}'] = groups[f'N_{suf}'].mean().round(4)
        agg[f'B_{mt}'] = groups[f'B_{suf}'].mean().round(4)
    agg = agg.reset_index()

    # win/tie/loss vs NLD (BMA better/equal/worse on F1)
    def cmp_row(r, met='F1'):
        g, n, b = r[f'G_{met}'], r[f'N_{met}'], r[f'B_{met}']
        return 'B>N' if b > n + 1e-9 else ('B=N' if abs(b - n) <= 1e-9 else 'B<N')

    agg['F1_BvsN'] = agg.apply(lambda r: cmp_row(r, 'F1'), axis=1)
    agg['AUC_BvsN'] = agg.apply(lambda r: cmp_row(r, 'AUC'), axis=1)
    # average rank per metric (lower better? we use higher better -> rank by value)
    for mt in ['AUC', 'Gmean', 'F1']:
        colG, colN, colB = f'G_{mt}', f'N_{mt}', f'B_{mt}'
        ranks = agg[[colG, colN, colB]].rank(axis=1, ascending=False, method='average')
        agg[f'{mt}_rankG'] = ranks[colG].round(2)
        agg[f'{mt}_rankN'] = ranks[colN].round(2)
        agg[f'{mt}_rankB'] = ranks[colB].round(2)

    agg.to_csv(os.path.join(OUT_DIR, 'summary_per_dataset.csv'), index=False)
    print("\n=== Aggregate over datasets (mean of fold-means) ===")
    for mt in ['AUC', 'Gmean', 'F1']:
        print(f"{mt:6s} | GNB {agg[f'G_{mt}'].mean():.4f} | NLD {agg[f'N_{mt}'].mean():.4f} | BMA {agg[f'B_{mt}'].mean():.4f}")
    print("\n=== Win/Tie/Loss (BMA vs NLD) ===")
    for met in ['F1', 'AUC']:
        vc = agg[f'{met}_BvsN'].value_counts()
        w = vc.get('B>N', 0); t = vc.get('B=N', 0); l = vc.get('B<N', 0)
        print(f"{met}: BMA>NLD {w} | BMA=NLD {t} | BMA<NLD {l}")
    print("\n=== Average rank (lower better) ===")
    for mt in ['AUC', 'Gmean', 'F1']:
        print(f"{mt:6s} | GNB {agg[f'{mt}_rankG'].mean():.2f} | NLD {agg[f'{mt}_rankN'].mean():.2f} | BMA {agg[f'{mt}_rankB'].mean():.2f}")


if __name__ == '__main__':
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    main(only)
