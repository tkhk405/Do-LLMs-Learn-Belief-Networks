"""Original pairwise Spearman and exhaustive one-sided Mantel calculations."""
from itertools import permutations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .constants import ISSUES, ISSUE_COLUMNS_JP

def utas_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Pairwise-complete Spearman matrix with ranks recomputed for each pair."""
    values = frame[ISSUE_COLUMNS_JP].to_numpy(float)
    matrix = np.full((len(ISSUES), len(ISSUES)), np.nan, dtype=float)
    for i in range(len(ISSUES)):
        for j in range(i, len(ISSUES)):
            pairwise_complete = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            if pairwise_complete.sum() < 2:
                continue
            rho = float(
                spearmanr(
                    values[pairwise_complete, i], values[pairwise_complete, j]
                )[0]
            )
            matrix[i, j] = matrix[j, i] = rho
    return matrix

def exact_mantel(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=(6,6) or b.shape!=(6,6):raise ValueError('Expected two 6 x 6 matrices')
    for x in [a,b]:
        if not np.isfinite(x).all() or not np.allclose(x,x.T,atol=1e-12,rtol=0):raise ValueError('Finite symmetric matrices required')
    ix=np.triu_indices(6,1);obs=float(spearmanr(a[ix],b[ix]).statistic)
    if not np.isfinite(obs):raise ValueError('Undefined matrix correlation')
    null=np.array([spearmanr(a[np.ix_(q,q)][ix],b[ix]).statistic for q in permutations(range(6))])
    # Match the original exhaustive test: include identity, use >=, no plus-one correction.
    exceed=int(np.sum(null>=obs))
    return {'rho':obs,'p_one_sided':exceed/720,'extreme_permutations':exceed,'permutations':720,'issue_pairs':15},null
