"""
Analyse approfondie de la calibration et de l'incertitude.

Évalue si les prédictions du modèle reflètent correctement l'incertitude réelle :
  - Calibration probability (Expected Calibration Error)
  - Confidence reliability diagrams
  - Decomposition aleatoric vs epistemic uncertainty
  - Brier score, log-loss
  - Reliability metrics
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss

# NB: matplotlib est importé paresseusement dans reliability_diagram (tracé
# optionnel) pour que ce module de MÉTRIQUES s'importe même sans matplotlib.

logger = logging.getLogger("panacee.calibration")


# ═══════════════════════════════════════════════════════════════
#  Calibration Metrics
# ═══════════════════════════════════════════════════════════════

class CalibrationAnalyzer:
    """
    Analyse la calibration des probabilités prédites.

    Une prédiction bien calibrée signifie que :
      - Si modèle dit P(y=1) = 0.7, alors ~70% des cas positifs
      - Le modèle quantifie correctement son incertitude
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: nombre de bins pour calibration
        """
        self.n_bins = n_bins
        if not isinstance(n_bins, int) or n_bins < 1:
            raise ValueError("n_bins doit être un entier positif")

    @staticmethod
    def _validate(y_true, y_proba):
        y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
        if y_true.shape != y_proba.shape or y_true.ndim != 1 or not len(y_true):
            raise ValueError("Cibles et probabilités doivent être des vecteurs non vides de même taille")
        if not np.isfinite(y_true).all() or not np.isfinite(y_proba).all():
            raise ValueError("Valeurs non finies dans la calibration")
        if not np.isin(y_true, [0, 1]).all() or ((y_proba < 0) | (y_proba > 1)).any():
            raise ValueError("Cibles binaires et probabilités dans [0, 1] requises")
        return y_true, y_proba

    def expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Expected Calibration Error (ECE).

        Mesure la différence moyenne entre confiance prédite et accuracy réelle.

        ECE = Σ |accuracy_i - confidence_i| * fraction_i

        Valeurs :
            0.0 = parfaitement calibré
            1.0 = complètement décalibré
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)

        total_samples = len(y_true)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_proba >= bin_edges[i]) & ((y_proba <= bin_edges[i + 1]) if i == self.n_bins - 1 else (y_proba < bin_edges[i + 1]))
            if mask.sum() == 0:
                continue

            bin_fraction = mask.sum() / total_samples
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_proba[mask].mean()

            ece += np.abs(bin_accuracy - bin_confidence) * bin_fraction

        return float(ece)

    def maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Maximum Calibration Error (MCE).

        Pire écart de confiance dans un bin.
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            mask = (y_proba >= bin_edges[i]) & ((y_proba <= bin_edges[i + 1]) if i == self.n_bins - 1 else (y_proba < bin_edges[i + 1]))
            if mask.sum() == 0:
                continue

            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_proba[mask].mean()

            mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return float(mce)

    def static_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Static Calibration Error (SCE).

        Version pondérée par la disparité des confiances.
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        predictions = (y_proba > 0.5).astype(int)
        overconfident = y_proba[predictions != y_true]

        if len(overconfident) == 0:
            return 0.0

        return float(np.mean(np.abs(overconfident - y_true[predictions != y_true])))

    def calibration_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Rapport complet de calibration.

        Returns:
            {metric_name: value}
        """
        return {
            "ece": self.expected_calibration_error(y_true, y_proba),
            "mce": self.maximum_calibration_error(y_true, y_proba),
            "sce": self.static_calibration_error(y_true, y_proba),
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        }

    def reliability_diagram(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """
        Génère un diagramme de reliability.

        Visualise la calibration : (confidence, accuracy) points et diagonal.
        """
        y_true, y_proba = self._validate(y_true, y_proba)
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = []
        bin_confs = []
        bin_counts = []
        used_centers = []

        for i in range(self.n_bins):
            mask = (y_proba >= bin_edges[i]) & ((y_proba <= bin_edges[i + 1]) if i == self.n_bins - 1 else (y_proba < bin_edges[i + 1]))
            if mask.sum() == 0:
                continue

            bin_accs.append(float(y_true[mask].mean()))
            bin_confs.append(float(y_proba[mask].mean()))
            bin_counts.append(int(mask.sum()))
            used_centers.append(float(bin_centers[i]))

        diagram_data = {
            "bin_centers": used_centers,
            "accuracies": bin_accs,
            "confidences": bin_confs,
            "bin_counts": bin_counts,
        }

        if save_path:
            import matplotlib.pyplot as plt  # import paresseux (tracé optionnel)
            _fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.scatter(bin_confs, bin_accs, s=bin_counts, alpha=0.6, label="Model")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Actual Frequency")
            ax.set_title("Reliability Diagram")
            ax.legend()
            ax.grid()
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close()

        return diagram_data


# ═══════════════════════════════════════════════════════════════
#  Uncertainty Decomposition
# ═══════════════════════════════════════════════════════════════

class UncertaintyDecomposition:
    """
    Décompose l'incertitude en composantes aleatoric et epistemic.

    Aleatoric : incertitude due au bruit dans les données
    Epistemic : incertitude due au manque de connaissance du modèle
    """

    @staticmethod
    def decompose_uncertainty(
        y_proba_mean: np.ndarray,
        y_proba_samples: np.ndarray,  # [n_samples, n_data]
    ) -> Dict[str, np.ndarray]:
        """
        Décompose l'incertitude (MC Dropout).

        Args:
            y_proba_mean: moyenne des prédictions
            y_proba_samples: échantillons de prédictions [n_monte_carlo, n_data]

        Returns:
            {total, aleatoric, epistemic}
        """
        samples = np.asarray(y_proba_samples, dtype=float)
        mean = np.asarray(y_proba_mean, dtype=float)
        if samples.ndim != mean.ndim + 1 or samples.shape[0] == 0 or samples.shape[1:] != mean.shape:
            raise ValueError("Dimensions des échantillons MC incompatibles")
        if not np.isfinite(samples).all() or ((samples < 0) | (samples > 1)).any():
            raise ValueError("Probabilités MC invalides")
        if not np.allclose(mean, samples.mean(axis=0), atol=1e-6):
            raise ValueError("La moyenne ne correspond pas aux échantillons MC")
        def entropy(prob):
            p = np.clip(prob, 1e-15, 1 - 1e-15)
            return -(p * np.log(p) + (1 - p) * np.log1p(-p))
        total = entropy(mean)
        aleatoric = entropy(samples).mean(axis=0)
        epistemic = np.maximum(total - aleatoric, 0.0)

        return {
            "total_uncertainty": total,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
        }

    @staticmethod
    def uncertainty_vs_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        total_uncertainty: np.ndarray,
    ) -> Dict[str, float]:
        """
        Vérifie si l'incertitude prédit bien les erreurs.

        Returns:
            {correlation, predictive_power}
        """
        errors = np.abs(y_true - y_proba)

        # Corrélation de Spearman
        correlation, p_value = stats.spearmanr(total_uncertainty, errors)

        # Capacité prédictive : prédictions avec haute incertitude sont-elles moins fiables?
        high_uncertainty = total_uncertainty > np.median(total_uncertainty)
        low_uncertainty = ~high_uncertainty

        error_high = errors[high_uncertainty].mean()
        error_low = errors[low_uncertainty].mean()

        predictive_power = error_high / (error_low + 1e-8)

        return {
            "uncertainty_error_correlation": float(correlation),
            "p_value": float(p_value),
            "predictive_power": float(predictive_power),  # ratio
        }


# ═══════════════════════════════════════════════════════════════
#  Confidence Thresholding
# ═══════════════════════════════════════════════════════════════

class ConfidenceThreshold:
    """
    Analyse le trade-off entre couverture et accuracy avec des seuils.
    """

    @staticmethod
    def coverage_accuracy_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, List]:
        """
        Génère la courbe couverture vs accuracy.

        À mesure qu'on augmente le seuil de confiance :
          - Couverture diminue (moins de prédictions)
          - Accuracy augmente (plus fiable sur les prédictions restantes)
        """
        thresholds = np.linspace(0.5, 1.0, 20)
        coverages = []
        accuracies = []
        y_true, y_proba = CalibrationAnalyzer._validate(y_true, y_proba)
        used_thresholds = []

        for threshold in thresholds:
            mask = np.maximum(y_proba, 1 - y_proba) >= threshold
            if mask.sum() == 0:
                continue

            coverage = mask.sum() / len(y_true)
            accuracy = (y_true[mask] == (y_proba[mask] > 0.5)).mean()

            coverages.append(float(coverage))
            accuracies.append(float(accuracy))
            used_thresholds.append(float(threshold))

        # AUC sur (couverture, accuracy) : trier par couverture croissante,
        # sinon np.trapz sur des x décroissants renvoie une aire négative.
        auc_val = 0.0
        if len(coverages) >= 2:
            order = np.argsort(coverages)
            cov_sorted = np.array(coverages)[order]
            acc_sorted = np.array(accuracies)[order]
            auc_val = float(np.trapezoid(acc_sorted, cov_sorted) if hasattr(np, "trapezoid")
                            else np.trapz(acc_sorted, cov_sorted))

        return {
            "thresholds": used_thresholds,
            "coverages": coverages,
            "accuracies": accuracies,
            "area_under_curve": auc_val,
        }

    @staticmethod
    def optimal_threshold(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Trouve le seuil optimal pour une métrique donnée.

        Args:
            metric: "f1", "accuracy", "precision", "recall"

        Returns:
            (optimal_threshold, metric_value)
        """
        thresholds = np.linspace(0.0, 1.0, 100)
        best_score = -1
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            if metric == "f1":
                from sklearn.metrics import f1_score
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "accuracy":
                score = (y_true == y_pred).mean()
            elif metric == "precision":
                from sklearn.metrics import precision_score
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                from sklearn.metrics import recall_score
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score


# ═══════════════════════════════════════════════════════════════
#  Selective Prediction
# ═══════════════════════════════════════════════════════════════

class SelectivePrediction:
    """
    Refuse les prédictions peu confiantes pour maintenir haute accuracy.
    """

    @staticmethod
    def prediction_with_rejection(
        y_proba: np.ndarray,
        rejection_threshold: float = 0.5,
    ) -> Dict:
        """
        Marque les prédictions comme acceptées/rejetées.

        Args:
            rejection_threshold: seuil de confiance

        Returns:
            {accepted_indices, rejected_indices, metrics}
        """
        y_proba = np.asarray(y_proba)
        CalibrationAnalyzer._validate(np.zeros_like(y_proba), y_proba)
        if not 0.5 <= rejection_threshold <= 1.0:
            raise ValueError("Seuil de confiance attendu dans [0.5, 1]")
        accepted = np.maximum(y_proba, 1 - y_proba) >= rejection_threshold
        rejected = ~accepted

        return {
            "accepted_indices": np.where(accepted)[0].tolist(),
            "rejected_indices": np.where(rejected)[0].tolist(),
            "acceptance_rate": float(accepted.mean()),
            "rejection_rate": float(rejected.mean()),
        }

    @staticmethod
    def coverage_rejection_rate(
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, List]:
        """
        Trade-off : accuracy augmente avec le taux de rejet.
        """
        thresholds = np.linspace(0.5, 1.0, 20)
        accuracies = []
        rejection_rates = []
        y_true, y_proba = CalibrationAnalyzer._validate(y_true, y_proba)

        for threshold in thresholds:
            accepted = np.maximum(y_proba, 1 - y_proba) >= threshold
            if accepted.sum() == 0:
                continue

            y_pred = (y_proba[accepted] > 0.5).astype(int)
            accuracy = (y_true[accepted] == y_pred).mean()
            rejection_rate = (~accepted).mean()

            accuracies.append(float(accuracy))
            rejection_rates.append(float(rejection_rate))

        return {
            "accuracies": accuracies,
            "rejection_rates": rejection_rates,
        }
