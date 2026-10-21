"""
Conformal Prediction pour Applicability Domain + incertitude calibrée.

Référence : Alvarsson et al. 2019 (Predicting with confidence: using conformal
prediction in drug discovery, arXiv:1908.03569) + Vovk et al. 2005 (Algorithmic
Learning in a Random World).

Mantra : "garantie d'erreur ≤ α pour toute distribution de données".

Implémente :
  - Inductive Conformal Classifier (ICP) binaire pour HIV_active
  - Mondrian conformal (par classe) → gère class imbalance (crucial pour HIV ~1.4%)
  - Non-conformité par score logit calibré ou proba Platt
  - P-values + prediction sets {active, inactive, both, none}
  - Tanimoto-based AD comme AD complémentaire (sim to nearest train > seuil)

Usage :
    conformal = ConformalClassifier(alpha=0.1).fit(calib_scores, calib_labels)
    sets = conformal.predict_set(test_scores)
    # chaque sample reçoit un ensemble de classes couvertes au niveau 1-α
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class ConformalClassifier:
    """
    Inductive Conformal Prediction binaire — garantie coverage ≥ 1 - α.

    Nécessite un calibration set DISJOINT du train (recommandation: 10-20% du train
    strict, scaffold-split pour cohérence avec test).

    Score de non-conformité par défaut : -proba(y_true | x) → une proba haute
    pour la vraie classe = score bas = conformant.
    """

    def __init__(self, alpha: float = 0.1, mondrian: bool = True):
        """
        Args:
            alpha: niveau d'erreur (0.1 → coverage ≥ 90%)
            mondrian: calibration par classe (recommandé pour imbalance HIV)
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.mondrian = mondrian
        self._calib_nc: dict = {}  # {class: sorted nc scores}

    def fit(
        self,
        calib_scores: Sequence[float],
        calib_labels: Sequence[int],
    ) -> "ConformalClassifier":
        """
        Args:
            calib_scores: probabilités p(y=1) sur set calibration
            calib_labels: labels binaires {0, 1}
        """
        s = np.asarray(calib_scores, dtype=float).ravel()
        y = np.asarray(calib_labels, dtype=int).ravel()
        if s.shape != y.shape:
            raise ValueError("shape mismatch")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("labels must be binary {0, 1}")
        if not np.all((s >= 0) & (s <= 1)):
            raise ValueError("scores must be probabilities in [0, 1]")

        # Non-conformité : plus la proba de la VRAIE classe est basse, plus NC haut
        nc = np.where(y == 1, 1.0 - s, s)

        if self.mondrian:
            for cls in (0, 1):
                mask = (y == cls)
                self._calib_nc[cls] = np.sort(nc[mask])
        else:
            self._calib_nc["all"] = np.sort(nc)

        return self

    def _p_value(self, nc_test: float, cls: int) -> float:
        """p-value conformelle : fraction de calibration points ≥ nc_test."""
        if self.mondrian:
            calib = self._calib_nc.get(cls)
            if calib is None or len(calib) == 0:
                return 0.0
        else:
            calib = self._calib_nc["all"]
        n = len(calib)
        # +1 au num et dénom (smoothed conformal, exchangeability)
        n_greater_eq = int(np.sum(calib >= nc_test))
        return (n_greater_eq + 1) / (n + 1)

    def predict_set(self, test_scores: Sequence[float]) -> list[set]:
        """
        Retourne pour chaque sample l'ensemble des classes couvertes au niveau 1-α.

        4 cas possibles :
          - {1}      : actif prédit (confiance >= 1-α)
          - {0}      : inactif prédit
          - {0, 1}   : AMBIGU (modèle incertain, in AD mais pas tranchant)
          - {}       : REJECT (hors applicabilité, aucune classe plausible)
        """
        if not self._calib_nc:
            raise RuntimeError("Appelle .fit() avant .predict_set()")
        scores = np.asarray(test_scores, dtype=float).ravel()
        sets: list[set] = []
        for s in scores:
            # Test si "0 est plausible" : NC = s
            # Test si "1 est plausible" : NC = 1-s
            p0 = self._p_value(s, cls=0)
            p1 = self._p_value(1.0 - s, cls=1)
            pred_set = set()
            if p0 > self.alpha:
                pred_set.add(0)
            if p1 > self.alpha:
                pred_set.add(1)
            sets.append(pred_set)
        return sets

    def predict_with_pvalues(self, test_scores: Sequence[float]) -> list[dict]:
        """Prédictions enrichies avec p-values par classe + décision."""
        if not self._calib_nc:
            raise RuntimeError("Appelle .fit() avant .predict_with_pvalues()")
        scores = np.asarray(test_scores, dtype=float).ravel()
        out = []
        for s in scores:
            p0 = self._p_value(s, cls=0)
            p1 = self._p_value(1.0 - s, cls=1)
            if p0 > self.alpha and p1 > self.alpha:
                decision = "ambiguous"
            elif p0 > self.alpha:
                decision = "inactive"
            elif p1 > self.alpha:
                decision = "active"
            else:
                decision = "reject_ood"
            out.append({
                "score": float(s),
                "p_inactive": float(p0),
                "p_active": float(p1),
                "decision": decision,
                "credibility": float(max(p0, p1)),
                "confidence": float(1.0 - min(p0, p1)),
            })
        return out


class TanimotoApplicabilityDomain:
    """
    AD par similarité Tanimoto Morgan (ECFP4) au voisin le plus proche du train.

    Convention pharma :
      - sim >= 0.7 : haute confiance, prédiction fiable
      - 0.4 <= sim < 0.7 : confiance modérée (in AD mais frontière)
      - sim < 0.4 : OUT of AD, prédiction non fiable
    """

    def __init__(self, high_threshold: float = 0.7, low_threshold: float = 0.4):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self._train_fps = None

    def fit(self, train_smiles: Sequence[str]) -> "TanimotoApplicabilityDomain":
        try:
            from rdkit import Chem
        except ImportError as e:
            raise RuntimeError("RDKit required for Tanimoto AD") from e
        from src.utils.fingerprints import morgan_fp_bitvect

        fps = []
        for smi in train_smiles:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                fps.append(morgan_fp_bitvect(mol, 2, 2048))
        if not fps:
            raise ValueError("No valid training fingerprints")
        self._train_fps = fps
        return self

    def classify(self, smiles: Sequence[str]) -> list[dict]:
        from rdkit import Chem
        from rdkit.Chem import DataStructs
        from src.utils.fingerprints import morgan_fp_bitvect

        if self._train_fps is None:
            raise RuntimeError("Appelle .fit() avant .classify()")

        out = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                out.append({"max_sim": None, "ad_class": "invalid"})
                continue
            fp = morgan_fp_bitvect(mol, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, self._train_fps)
            max_sim = max(sims) if sims else 0.0
            if max_sim >= self.high_threshold:
                ad = "in_domain_high"
            elif max_sim >= self.low_threshold:
                ad = "in_domain_moderate"
            else:
                ad = "out_of_domain"
            out.append({"max_sim": float(max_sim), "ad_class": ad})
        return out


def evaluate_conformal_coverage(
    y_true: Sequence[int], prediction_sets: list[set], alpha: float = 0.1,
) -> dict:
    """
    Vérifie que la couverture empirique respecte la garantie théorique >= 1-α.

    Returns:
        {empirical_coverage, target_coverage, avg_set_size, ambiguous_rate, reject_rate}
    """
    y = np.asarray(y_true, dtype=int).ravel()
    n = len(y)
    covered = sum(1 for yi, ps in zip(y, prediction_sets, strict=True) if yi in ps)
    sizes = [len(ps) for ps in prediction_sets]
    return {
        "empirical_coverage": covered / n if n else 0.0,
        "target_coverage": 1.0 - alpha,
        "coverage_valid": (covered / n if n else 0.0) >= (1.0 - alpha) - 0.05,
        "avg_set_size": float(np.mean(sizes)) if sizes else 0.0,
        "ambiguous_rate": float(np.mean([s == 2 for s in sizes])),
        "reject_rate": float(np.mean([s == 0 for s in sizes])),
        "singleton_rate": float(np.mean([s == 1 for s in sizes])),
    }


if __name__ == "__main__":
    np.random.seed(42)
    # Smoke test : simule un classifier HIV calibré
    n_calib, n_test = 1000, 500
    y_calib = np.random.binomial(1, 0.014, size=n_calib)
    s_calib = np.clip(y_calib * 0.6 + np.random.randn(n_calib) * 0.15 + 0.3, 0, 1)
    y_test = np.random.binomial(1, 0.014, size=n_test)
    s_test = np.clip(y_test * 0.6 + np.random.randn(n_test) * 0.15 + 0.3, 0, 1)

    cp = ConformalClassifier(alpha=0.1, mondrian=True).fit(s_calib, y_calib)
    sets = cp.predict_set(s_test)
    cov = evaluate_conformal_coverage(y_test, sets, alpha=0.1)
    print("=== Smoke test conformal ===")
    for k, v in cov.items():
        print(f"  {k}: {v}")
