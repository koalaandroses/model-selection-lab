"""Reproducible subset-selection experiment on explicitly synthetic data."""
from pathlib import Path
from itertools import combinations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def fit_score(x, y):
    """Gaussian OLS ICs; k includes intercept, slopes and residual variance."""
    design = np.column_stack([np.ones(len(y)), x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('Rank-deficient design')
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    n, k = len(y), design.shape[1] + 1
    rss = max(float(np.sum((y - design @ beta) ** 2)), 1e-12)
    # Common normal-density constants omitted; comparisons use identical rows.
    minus2loglik = n * np.log(rss / n)
    aic = minus2loglik + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    bic = minus2loglik + k * np.log(n)
    return beta, aic, aicc, bic

def experiment(seed=42, bootstraps=1000, output_dir=None):
    if bootstraps < 1:
        raise ValueError('bootstraps must be positive')
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(180, 4))
    x[:, 1] = 0.8 * x[:, 0] + 0.6 * x[:, 1]
    y = 3 + 2 * x[:, 0] - 1.5 * x[:, 2] + rng.normal(0, 2, len(x))
    # Holdout never participates in selection or bootstrap resampling.
    order = rng.permutation(len(x))
    train, test = order[:120], order[120:]
    subsets = [s for size in range(1, 5) for s in combinations(range(4), size)]
    records, predictions = [], []
    for s in subsets:
        beta, aic, aicc, bic = fit_score(x[train][:, s], y[train])
        pred = np.column_stack([np.ones(len(test)), x[test][:, s]]) @ beta
        predictions.append(pred)
        records.append(dict(model='+'.join(f'x{i+1}' for i in s), aic=aic,
                            aicc=aicc, bic=bic, holdout_rmse=np.sqrt(np.mean((pred-y[test])**2))))
    scores = pd.DataFrame(records)
    delta = scores.aic - scores.aic.min()
    weights = np.exp(-delta/2)
    scores['akaike_weight'] = weights / weights.sum()
    counts = np.zeros((len(subsets), 2), dtype=int)
    skipped = 0
    for _ in range(bootstraps):
        idx = rng.choice(train, len(train), replace=True)
        try:
            ic = np.array([fit_score(x[idx][:, s], y[idx])[2:] for s in subsets])
        except ValueError:
            skipped += 1
            continue
        for column in range(2):
            counts[np.argmin(ic[:, column]), column] += 1
    valid = bootstraps - skipped
    if valid == 0:
        raise ValueError('No valid bootstrap samples')
    scores['bootstrap_aicc_frequency'] = counts[:, 0] / valid
    scores['bootstrap_bic_frequency'] = counts[:, 1] / valid
    averaged = scores.akaike_weight.to_numpy() @ np.array(predictions)
    average_rmse = np.sqrt(np.mean((averaged-y[test])**2))
    out = Path(output_dir) if output_dir is not None else ROOT / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(x, columns=['x1','x2','x3','x4'])
    data['y'], data['split'] = y, 'test'
    data.loc[train, 'split'] = 'train'
    data.to_csv(out/'synthetic_data.csv', index=False)
    scores.sort_values('aicc').to_csv(out/'model_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    top = scores.sort_values('bootstrap_aicc_frequency', ascending=False).head(7)
    ax.bar(top.model, top.bootstrap_aicc_frequency, color='#286989')
    ax.set(ylabel='Selection frequency', title=f'Model stability across {valid} bootstrap samples', ylim=(0,1))
    fig.tight_layout()
    fig.savefig(out/'selection_stability.png', dpi=170)
    plt.close(fig)
    winner = scores.loc[scores.aicc.idxmin(), 'model']
    summary = f'''# Experiment results (synthetic data)

Seed: {seed}; training rows: {len(train)}; untouched holdout rows: {len(test)}.
Enumerated models: {len(subsets)} (nonempty subsets of four predictors).
Valid bootstrap samples: {valid}; skipped rank-deficient samples: {skipped}.
AICc winner on original training sample: {winner}.
AIC-weighted prediction holdout RMSE: {average_rmse:.4f}.

The simulation was generated with x1 and x3 as true signals. x2 is correlated
with x1; x4 is noise. Inspect the CSV to compare IC rankings and stability.
RMSE is diagnostic here, not used to tune models. Model averaging is weighted
prediction averaging, not averaging raw coefficients across different models.
These results do not establish general performance on real datasets.
'''
    (out/'results.md').write_text(summary)
    print(summary)
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstraps', type=int, default=1000)
    args = parser.parse_args()
    experiment(args.seed, args.bootstraps)
