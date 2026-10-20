"""
Métriques pharma pour virtual screening — au-delà de ROC-AUC.

Comble le gap identifié dans l'audit (src/validation/clinical_metrics.py n'a ni
enrichment factor ni hit rate top-k) et aligne avec les standards industriels :

  - Enrichment Factor @ X% (EF@1%, EF@5%, EF@10%)
  - BEDROC α=20 (Truchon-Bayly 2007) — early recognition
  - Hit rate top-K (K=100, 500, 1000)
  - Novelty vs training set (Tanimoto to nearest train active)
  - ROC-AUC et PRC-AUC pour référence

Ces métriques sont celles rapportées dans les papers HIV MoleculeNet (TDC,
Jiménez-Luna 2020, MolCLR, Uni-Mol) et requises pour une publication crédible.

Références scientifiques :
  - Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508.
  - Jain & Nicholls, J. Comput. Aided Mol. Des. 2008, 22, 133-139.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _validate(y_true: Sequence, y_score: Sequence) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_score {s.shape}")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must be binary {0, 1}")
    if not np.all(np.isfinite(s)):
        raise ValueError("y_score contains non-finite values")
    return y, s


def enrichment_factor(
    y_true: Sequence, y_score: Sequence, top_fraction: float = 0.01,
) -> Optional[float]:
    """
    EF@X% = (actifs trouvés dans top X% / top X%) / (actifs totaux / total)

    Mesure combien de fois mieux que le hasard le modèle enrichit les actifs
    dans le top-X des prédictions. Standard industriel virtual screening.

    EF=1 → aussi bon que le hasard. EF=10 → 10x mieux que random.
    Pour HIV (~1.4% actifs), EF@1% max théorique ≈ 72.
    """
    y, s = _validate(y_true, y_score)
    n = len(y)
    n_actives = int(y.sum())
    if n_actives == 0:
        return None
    k = max(1, int(round(n * top_fraction)))
    order = np.argsort(-s)[:k]  # top-k
    n_actives_top = int(y[order].sum())
    base_rate = n_actives / n
    ef = (n_actives_top / k) / base_rate
    return float(ef)


def bedroc(
    y_true: Sequence, y_score: Sequence, alpha: float = 20.0,
) -> Optional[float]:
    """
    BEDROC (Boltzmann-Enhanced Discrimination of ROC) — Truchon & Bayly 2007.

    Pondère les premiers hits exponentiellement — mesure early recognition.
    α=20 → poids concentré sur ~top 8% (standard pharma pour HIV/ADMET).
    α=320 → top 0.5% uniquement.

    BEDROC ∈ [0, 1]. BEDROC=0.5 ≈ random pour α modéré.
    """
    y, s = _validate(y_true, y_score)
    n = len(y)
    n_actives = int(y.sum())
    if n_actives == 0 or n_actives == n:
        return None

    r_a = n_actives / n
    order = np.argsort(-s)
    y_sorted = y[order]
    ranks = np.where(y_sorted == 1)[0] + 1  # 1-indexed

    # Terme principal Truchon-Bayly
    sum_exp = float(np.sum(np.exp(-alpha * ranks / n)))

    numerator = sum_exp * (1.0 - math.exp(-alpha)) / (
        r_a * (1.0 - math.exp(-alpha * r_a)) * (math.exp(alpha / n) - 1.0)
    )

    # Facteur de normalisation
    rie_max = (1.0 - math.exp(-alpha * r_a)) / (r_a * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * r_a)) / (r_a * (1.0 - math.exp(alpha)))

    rie = numerator
    if rie_max == rie_min:
        return 0.5
    bedroc_val = (rie - rie_min) / (rie_max - rie_min)
    return float(np.clip(bedroc_val, 0.0, 1.0))


def hit_rate_top_k(
    y_true: Sequence, y_score: Sequence, k: int,
) -> Optional[float]:
    """Fraction d'actifs dans le top-K prédictions. Interprétation chemists directe."""
    y, s = _validate(y_true, y_score)
    if k <= 0 or k > len(y):
        return None
    order = np.argsort(-s)[:k]
    hits = int(y[order].sum())
    return float(hits / k)


def recall_at_top_fraction(
    y_true: Sequence, y_score: Sequence, top_fraction: float = 0.01,
) -> Optional[float]:
    """Recall des actifs dans top-X%. Complément à hit_rate."""
    y, s = _validate(y_true, y_score)
    n_actives = int(y.sum())
    if n_actives == 0:
        return None
    k = max(1, int(round(len(y) * top_fraction)))
    order = np.argsort(-s)[:k]
    return float(y[order].sum() / n_actives)


def novelty_vs_training(
    predicted_smiles: Sequence[str],
    training_actives_smiles: Sequence[str],
    novelty_threshold: float = 0.4,
) -> Optional[dict]:
    """
    Pour chaque prédit, calcule la Tanimoto max vs les actifs d'entraînement.
    Molécules "nouvelles" = sim < novelty_threshold.

    Returns:
        dict {novel_fraction, mean_max_sim, median_max_sim}
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except ImportError:
        return None

    def _fp(smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    train_fps = [fp for fp in (_fp(s) for s in training_actives_smiles) if fp is not None]
    if not train_fps:
        return None

    max_sims = []
    for smi in predicted_smiles:
        fp = _fp(smi)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        max_sims.append(max(sims))

    if not max_sims:
        return None

    max_sims = np.array(max_sims)
    return {
        "novel_fraction": float((max_sims < novelty_threshold).mean()),
        "mean_max_sim": float(max_sims.mean()),
        "median_max_sim": float(np.median(max_sims)),
        "novelty_threshold": novelty_threshold,
    }


def full_screening_report(
    y_true: Sequence,
    y_score: Sequence,
    smiles: Optional[Sequence[str]] = None,
    training_actives: Optional[Sequence[str]] = None,
) -> dict:
    """Rapport complet pour un run de virtual screening HIV."""
    y, s = _validate(y_true, y_score)

    report = {
        "n_total": len(y),
        "n_actives": int(y.sum()),
        "actives_fraction": float(y.sum() / len(y)) if len(y) else 0.0,
        "ef_1pct": enrichment_factor(y, s, 0.01),
        "ef_5pct": enrichment_factor(y, s, 0.05),
        "ef_10pct": enrichment_factor(y, s, 0.10),
        "bedroc_alpha20": bedroc(y, s, alpha=20.0),
        "bedroc_alpha80": bedroc(y, s, alpha=80.0),
        "hit_rate_top_100": hit_rate_top_k(y, s, 100),
        "hit_rate_top_500": hit_rate_top_k(y, s, 500),
        "hit_rate_top_1000": hit_rate_top_k(y, s, 1000),
        "recall_top_1pct": recall_at_top_fraction(y, s, 0.01),
        "recall_top_5pct": recall_at_top_fraction(y, s, 0.05),
        "recall_top_10pct": recall_at_top_fraction(y, s, 0.10),
    }

    if _HAS_SKLEARN:
        try:
            report["roc_auc"] = float(roc_auc_score(y, s))
            report["prc_auc"] = float(average_precision_score(y, s))
        except Exception:
            pass

    if smiles is not None and training_actives is not None:
        nov = novelty_vs_training(smiles, training_actives)
        if nov is not None:
            report["novelty"] = nov

    return report


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    n = 10000
    y = np.random.binomial(1, 0.014, size=n)  # 1.4% actives (HIV-like)
    # Score corrélé pour simuler un modèle utile (AUC ~0.82)
    noise = np.random.randn(n)
    s = y * 2 + noise  # separation ~2 std

    print("=== Smoke test pharma_metrics ===")
    report = full_screening_report(y, s)
    for k, v in report.items():
        print(f"  {k}: {v}")
