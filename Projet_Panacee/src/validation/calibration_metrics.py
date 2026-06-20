"""
Analyse approfondie de la calibration et de l'incertitude.

Évalue si les prédictions du modèle reflètent correctement l'incertitude réelle :
  - Calibration probability (Expected Calibration Error)
  - Confidence reliability diagrams
  - Decomposition aleatoric vs epistemic uncertainty
  - Brier score, log-loss
  - Reliability metrics
"""
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import logging

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
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total_samples = len(y_true)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
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
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
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
            "log_loss": float(log_loss(y_true, y_proba)),
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
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(self.n_bins):
            mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if mask.sum() == 0:
                continue

            bin_accs.append(y_true[mask].mean())
            bin_confs.append(y_proba[mask].mean())
            bin_counts.append(mask.sum())

        diagram_data = {
            "bin_centers": bin_centers[: len(bin_accs)].tolist(),
            "accuracies": bin_accs,
            "confidences": bin_confs,
            "bin_counts": bin_counts,
        }

        if save_path:
            fig, ax = plt.subplots(figsize=(8, 8))
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
        # Epistemic (aleatoric) : variance across MC samples
        epistemic = np.var(y_proba_samples, axis=0)

        # Aleatoric : average entropy
        entropy = -y_proba_mean * np.log(y_proba_mean + 1e-10) - (1 - y_proba_mean) * np.log(1 - y_proba_mean + 1e-10)

        # Total : entropy + epistemic
        total = entropy + epistemic

        return {
            "total_uncertainty": total,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": entropy,
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

        for threshold in thresholds:
            mask = y_proba >= threshold
            if mask.sum() == 0:
                continue

            coverage = mask.sum() / len(y_true)
            accuracy = (y_true[mask] == (y_proba[mask] > 0.5)).mean()

            coverages.append(float(coverage))
            accuracies.append(float(accuracy))

        return {
            "thresholds": thresholds[: len(coverages)].tolist(),
            "coverages": coverages,
            "accuracies": accuracies,
            "area_under_curve": float(np.trapz(accuracies, coverages)) if coverages else 0.0,
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
        accepted = y_proba >= rejection_threshold
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

        for threshold in thresholds:
            accepted = y_proba >= threshold
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
