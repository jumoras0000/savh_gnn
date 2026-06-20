"""
Framework de validation rigoureuse pour études académiques avancées.

Fonctionnalités :
  - Cross-validation (k-fold, stratifiée)
  - Ablation studies (impact de chaque module)
  - Comparaison avec baselines
  - Statistical significance testing
  - Courbes d'apprentissage
  - AUC/AUROC, calibration
"""
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass, field
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, f1_score, precision_recall_curve,
    brier_score_loss, log_loss, mean_squared_error, r2_score,
)
from scipy import stats
from pathlib import Path
import json

logger = logging.getLogger("panacee.validation")


# ═══════════════════════════════════════════════════════════════
#  Structures de données
# ═══════════════════════════════════════════════════════════════

@dataclass
class MetricResult:
    """Résultat d'une métrique avec statistiques."""
    name: str
    values: List[float]  # Scores par fold
    mean: float = 0.0
    std: float = 0.0
    ci_lower: float = 0.0  # Intervalle de confiance 95%
    ci_upper: float = 0.0

    def __post_init__(self):
        if self.values:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values)
            if len(self.values) > 1:
                se = self.std / np.sqrt(len(self.values))
                t_crit = stats.t.ppf(0.975, len(self.values) - 1)
                self.ci_lower = self.mean - t_crit * se
                self.ci_upper = self.mean + t_crit * se
            else:
                self.ci_lower = self.ci_upper = self.mean

    def __str__(self):
        return f"{self.name}: {self.mean:.4f} ± {self.std:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"


@dataclass
class ValidationResult:
    """Résultats complets d'une validation."""
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AblationStudyResult:
    """Résultats d'une étude d'ablation."""
    component: str
    metrics_with: Dict[str, MetricResult]
    metrics_without: Dict[str, MetricResult]
    impact: Dict[str, float]  # Différence de performance


# ═══════════════════════════════════════════════════════════════
#  Cross-Validation Framework
# ═══════════════════════════════════════════════════════════════

class CrossValidator:
    """
    Framework de cross-validation k-fold pour évaluation rigoureuse.
    
    Supporte :
      - K-fold standard et stratifiée
      - Métriques classification et régression
      - Statistiques d'erreur
    """

    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            n_splits: nombre de folds
            stratified: utiliser stratification (pour data déséquilibrée)
            random_state: seed pour reproducibilité
        """
        self.n_splits = n_splits
        self.stratified = stratified
        self.random_state = random_state
        self.results: List[ValidationResult] = []

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        metric_fn: Callable,
        y_proba_fn: Optional[Callable] = None,
    ) -> Dict[str, MetricResult]:
        """
        Lance la cross-validation k-fold.

        Args:
            X: features [N, D]
            y: labels [N]
            model_factory: fonction qui crée un nouveau modèle
            metric_fn: fonction qui calcule les métriques
            y_proba_fn: (optionnel) fonction pour obtenir les probabilités

        Returns:
            Dictionnaire {metric_name: MetricResult}
        """
        if self.stratified and len(np.unique(y)) < len(y):
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

        self.results = []
        all_metrics = {}

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            logger.info(f"Fold {fold+1}/{self.n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Entraîner
            model = model_factory()
            model.fit(X_train, y_train)

            # Prédire
            y_pred = model.predict(X_test)
            y_proba = y_proba_fn(model, X_test) if y_proba_fn else None

            # Métriques
            metrics = metric_fn(y_test, y_pred, y_proba)

            result = ValidationResult(
                fold=fold,
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                metrics=metrics,
            )
            self.results.append(result)

            # Agréger
            for m_name, m_val in metrics.items():
                if m_name not in all_metrics:
                    all_metrics[m_name] = []
                all_metrics[m_name].append(m_val)

        # Convertir en MetricResult
        metric_results = {}
        for m_name, m_vals in all_metrics.items():
            metric_results[m_name] = MetricResult(name=m_name, values=m_vals)

        return metric_results

    def get_learning_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        metric_fn: Callable,
        train_sizes: List[float] = None,
    ) -> Dict:
        """
        Génère une courbe d'apprentissage (taille vs performance).

        Args:
            train_sizes: fractions de données d'entraînement à tester

        Returns:
            {train_size: {metric_name: MetricResult}}
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)

        learning_curve_results = {}

        for train_size in train_sizes:
            n_train = int(len(X) * train_size)
            
            fold_scores = {m: [] for m in ["auc", "f1"]}
            for fold, (train_idx, test_idx) in enumerate(
                KFold(n_splits=5, shuffle=True, random_state=42).split(X)
            ):
                X_train, X_test = X[train_idx[:n_train]], X[test_idx]
                y_train, y_test = y[train_idx[:n_train]], y[test_idx]

                model = model_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = metric_fn(y_test, y_pred, None)
                for m_name, m_val in metrics.items():
                    if m_name in fold_scores:
                        fold_scores[m_name].append(m_val)

            learning_curve_results[train_size] = {
                m_name: MetricResult(name=m_name, values=m_vals)
                for m_name, m_vals in fold_scores.items()
                if m_vals
            }

        return learning_curve_results


# ═══════════════════════════════════════════════════════════════
#  Ablation Study Framework
# ═══════════════════════════════════════════════════════════════

class AblationStudy:
    """
    Étude d'ablation pour évaluer l'impact de chaque composant.
    
    Compare les performances avec/sans chaque module.
    """

    def __init__(
        self,
        base_model_factory: Callable,
        variant_factories: Dict[str, Callable],
        metric_fn: Callable,
        n_repeats: int = 3,
    ):
        """
        Args:
            base_model_factory: fonction créant le modèle complet
            variant_factories: {component_name: factory_without_component}
            metric_fn: fonction de calcul des métriques
            n_repeats: nombre de répétitions pour robustesse
        """
        self.base_factory = base_model_factory
        self.variants = variant_factories
        self.metric_fn = metric_fn
        self.n_repeats = n_repeats
        self.results: List[AblationStudyResult] = []

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> List[AblationStudyResult]:
        """
        Lance l'étude d'ablation.

        Args:
            X: données
            y: labels
            test_size: proportion test

        Returns:
            Résultats par composant
        """
        from sklearn.model_selection import train_test_split

        self.results = []

        # Évaluer le modèle complet
        base_scores = {m: [] for m in ["auc", "f1"]}
        for rep in range(self.n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42 + rep
            )

            model = self.base_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = self.metric_fn(y_test, y_pred, None)

            for m_name, m_val in metrics.items():
                if m_name in base_scores:
                    base_scores[m_name].append(m_val)

        base_metrics = {
            m_name: MetricResult(name=m_name, values=m_vals)
            for m_name, m_vals in base_scores.items()
            if m_vals
        }

        # Évaluer chaque variante (sans un composant)
        for comp_name, variant_factory in self.variants.items():
            logger.info(f"Ablation test: sans {comp_name}")

            variant_scores = {m: [] for m in ["auc", "f1"]}
            for rep in range(self.n_repeats):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42 + rep
                )

                model = variant_factory()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = self.metric_fn(y_test, y_pred, None)

                for m_name, m_val in metrics.items():
                    if m_name in variant_scores:
                        variant_scores[m_name].append(m_val)

            variant_metrics = {
                m_name: MetricResult(name=m_name, values=m_vals)
                for m_name, m_vals in variant_scores.items()
                if m_vals
            }

            # Calculer l'impact (différence)
            impact = {}
            for m_name in base_metrics:
                base_val = base_metrics[m_name].mean
                variant_val = variant_metrics[m_name].mean
                impact[m_name] = base_val - variant_val

            result = AblationStudyResult(
                component=comp_name,
                metrics_with=base_metrics,
                metrics_without=variant_metrics,
                impact=impact,
            )
            self.results.append(result)

        return self.results


# ═══════════════════════════════════════════════════════════════
#  Baseline & Comparaison
# ═══════════════════════════════════════════════════════════════

class BaselineComparison:
    """
    Compare le modèle avancé avec des baselines simples.
    
    Baselines :
      - Prédictionnaive (classe majoritaire)
      - Régression linéaire
      - Random forest simple
    """

    @staticmethod
    def compare_models(
        X: np.ndarray,
        y: np.ndarray,
        advanced_model_factory: Callable,
        metric_fn: Callable,
        test_size: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare les performances.

        Returns:
            {model_name: {metric_name: value}}
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.dummy import DummyClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        results = {}

        # Baseline 1: Prédiction naïve
        baseline_naive = DummyClassifier(strategy="most_frequent")
        baseline_naive.fit(X_train, y_train)
        y_pred = baseline_naive.predict(X_test)
        results["Naive"] = metric_fn(y_test, y_pred, None)

        # Baseline 2: Régression logistique
        baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
        baseline_lr.fit(X_train, y_train)
        y_pred = baseline_lr.predict(X_test)
        results["Logistic Regression"] = metric_fn(y_test, y_pred, None)

        # Baseline 3: Random forest simple
        baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_rf.fit(X_train, y_train)
        y_pred = baseline_rf.predict(X_test)
        results["Random Forest"] = metric_fn(y_test, y_pred, None)

        # Modèle avancé
        advanced = advanced_model_factory()
        advanced.fit(X_train, y_train)
        y_pred = advanced.predict(X_test)
        results["Advanced Model"] = metric_fn(y_test, y_pred, None)

        return results


# ═══════════════════════════════════════════════════════════════
#  Statistical Significance Testing
# ═══════════════════════════════════════════════════════════════

class SignificanceTest:
    """
    Tests statistiques pour évaluer la significativité des différences.
    """

    @staticmethod
    def paired_t_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05,
    ) -> Dict:
        """
        T-test appairé (pour cross-validation).

        Returns:
            {t_statistic, p_value, significant, effect_size}
        """
        t_stat, p_val = stats.ttest_rel(scores1, scores2)
        significant = p_val < alpha

        # Cohen's d
        d = (np.mean(scores1) - np.mean(scores2)) / np.std(scores1 - np.array(scores2))

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(significant),
            "cohens_d": float(d),
        }

    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Test de McNemar (comparaison de 2 classifieurs).

        Returns:
            {chi2, p_value, significant}
        """
        # Contingency table
        disagreement = (y_pred1 != y_pred2)
        n_01 = ((y_pred1[disagreement] == y_true[disagreement]) & (y_pred2[disagreement] != y_true[disagreement])).sum()
        n_10 = ((y_pred1[disagreement] != y_true[disagreement]) & (y_pred2[disagreement] == y_true[disagreement])).sum()

        chi2 = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10 + 1e-8)
        p_val = 1 - stats.chi2.cdf(chi2, 1)

        return {
            "chi2": float(chi2),
            "p_value": float(p_val),
            "significant": p_val < alpha,
        }
