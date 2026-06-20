"""
advanced_algorithms.py – Algorithmes de pointe
===============================================
Algorithmes avancés pour la recherche de combinaisons optimales,
basés sur la littérature scientifique récente.

Modules :
  1. MCTSCombinationSearch  – Monte Carlo Tree Search pour combinaisons
     Réf. : Silver et al. (2016) "Mastering the game of Go with MCTS"

  2. BayesianOptimizer      – Optimisation bayésienne (acquisition EI)
     Réf. : Snoek et al. (2012) "Practical Bayesian Optimization of ML algorithms"
             Ramsundar et al. (2019) "Deep Learning for the Life Sciences" (Chap. 9)

  3. ParetoOptimizer        – Optimisation multi-objectif (front de Pareto)
     Réf. : Daulton et al. (2020) "Differentiable Expected Hypervolume Improvement"

  4. EnsemblePredictor      – Ensemble de modèles avec calibration
     Réf. : Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty"

  5. GradientActivationMap  – Carte d'activation pour interprétabilité GNN
     Réf. : Pope et al. (2019) "Explainability Methods for Graph Convolutional Neural Networks"
"""
import os
import math
import random
import logging
from typing import List, Optional, Dict, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("panacee.advanced")


# ══════════════════════════════════════════════════════════════════════
#  1. MONTE CARLO TREE SEARCH (MCTS)
# ══════════════════════════════════════════════════════════════════════

class MCTSNode:
    """Nœud d'arbre pour MCTS."""

    def __init__(self, mol_indices: List[int], parent=None, c_puct: float = 1.41):
        self.mol_indices = mol_indices
        self.parent      = parent
        self.children: List["MCTSNode"] = []
        self.visits      = 0
        self.total_score = 0.0
        self.c_puct      = c_puct

    @property
    def mean_score(self) -> float:
        return self.total_score / max(self.visits, 1)

    def ucb1(self) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        return self.mean_score + self.c_puct * math.sqrt(
            math.log(parent_visits + 1) / (self.visits + 1)
        )

    def select_best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1())

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def expand(self, all_indices: List[int], max_size: int):
        """Génère tous les enfants en ajoutant une molécule."""
        current = set(self.mol_indices)
        for idx in all_indices:
            if idx not in current and len(self.mol_indices) < max_size:
                child = MCTSNode(self.mol_indices + [idx], parent=self)
                self.children.append(child)

    def backpropagate(self, score: float):
        """Remonte le score vers la racine."""
        self.visits      += 1
        self.total_score += score
        if self.parent:
            self.parent.backpropagate(score)


class MCTSCombinationSearch:
    """
    Recherche MCTS de la meilleure combinaison de molécules.

    Utilise l'UCB1 pour équilibrer exploration / exploitation.
    Bien adapté aux grands espaces de recherche (> 100 molécules).

    Référence :
      Réimplémenter l'idée de MCTS pour la combinaison chimique,
      inspiré de REINVENT (Blaschke et al. 2020, ChemRxiv).
    """

    def __init__(
        self,
        score_fn: Callable[[List[int]], float],
        n_simulations: int  = 200,
        max_combo_size: int = 4,
        c_puct: float       = 1.41,
    ):
        self.score_fn      = score_fn
        self.n_simulations = n_simulations
        self.max_combo_size = max_combo_size
        self.c_puct        = c_puct

    def search(self, mol_indices: List[int]) -> Tuple[List[int], float]:
        """
        Cherche la meilleure combinaison parmi les mol_indices.

        Returns:
            best_indices, best_score
        """
        root = MCTSNode([], c_puct=self.c_puct)
        root.expand(mol_indices, self.max_combo_size)

        best_indices, best_score = [], 0.0

        for sim in range(self.n_simulations):
            # ── Sélection ─────────────────────────────────────────────
            node = root
            while not node.is_leaf():
                node = node.select_best_child()

            # ── Expansion ─────────────────────────────────────────────
            if node.visits > 0 and len(node.mol_indices) < self.max_combo_size:
                node.expand(mol_indices, self.max_combo_size)
                if node.children:
                    node = random.choice(node.children)

            # ── Simulation (rollout) ───────────────────────────────────
            if len(node.mol_indices) >= 2:
                score = self.score_fn(node.mol_indices)
                if score > best_score:
                    best_score   = score
                    best_indices = node.mol_indices.copy()
            else:
                score = 0.0

            # ── Rétropropagation ──────────────────────────────────────
            node.backpropagate(score)

            if (sim + 1) % 50 == 0:
                logger.debug(f"  MCTS simulation {sim+1}/{self.n_simulations} — best={best_score:.4f}")

        return best_indices, best_score


# ══════════════════════════════════════════════════════════════════════
#  2. OPTIMISATION BAYÉSIENNE
# ══════════════════════════════════════════════════════════════════════

class GaussianProcessSimple:
    """
    GP simplifié (noyau RBF) pour l'optimisation bayésienne.
    Utilisé en interne par BayesianOptimizer.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-3):
        self.length_scale = length_scale
        self.noise        = noise
        self.X_train      = None
        self.y_train      = None

    def _rbf(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        return np.exp(-0.5 * (diff ** 2).sum(-1) / self.length_scale ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))

        K_tt = self._rbf(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        K_st = self._rbf(X, self.X_train)

        K_inv = np.linalg.pinv(K_tt)
        mu    = K_st @ K_inv @ self.y_train
        var   = 1.0 - np.diag(K_st @ K_inv @ K_st.T)
        return mu, np.maximum(var, 1e-8)


class BayesianOptimizer:
    """
    Optimisation bayésienne pour maximiser un score (par ex. combo_score).

    Algorithme :
      1. Évaluer quelques points aléatoires (exploration initiale)
      2. Ajuster un GP sur les observations
      3. Choisir le prochain point via Expected Improvement (EI)
      4. Répéter jusqu'au budget

    Référence :
      Snoek et al. (2012) "Practical Bayesian Optimization of Machine Learning Algorithms"
      NeurIPS 2012.
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], float],
        bounds: np.ndarray,          # [D, 2] min/max par dimension
        n_initial: int = 10,
        n_iterations: int = 50,
        xi: float = 0.01,            # paramètre d'exploration EI
    ):
        self.score_fn    = score_fn
        self.bounds      = bounds
        self.n_initial   = n_initial
        self.n_iterations = n_iterations
        self.xi          = xi
        self.gp          = GaussianProcessSimple()
        self.X_obs       = []
        self.y_obs       = []

    def _expected_improvement(self, X: np.ndarray, y_best: float) -> np.ndarray:
        """EI = E[max(f(x) - y_best, 0)]"""
        from scipy.stats import norm
        mu, var  = self.gp.predict(X)
        std      = np.sqrt(var)
        Z        = (mu - y_best - self.xi) / (std + 1e-9)
        ei       = (mu - y_best - self.xi) * norm.cdf(Z) + std * norm.pdf(Z)
        return np.maximum(ei, 0.0)

    def _random_sample(self, n: int) -> np.ndarray:
        D = self.bounds.shape[0]
        X = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(n, D)
        )
        return X

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Lance l'optimisation.

        Returns:
            best_x, best_y
        """
        # Exploration initiale aléatoire
        X_init = self._random_sample(self.n_initial)
        for x in X_init:
            y = self.score_fn(x)
            self.X_obs.append(x)
            self.y_obs.append(y)

        best_idx = np.argmax(self.y_obs)
        best_x   = self.X_obs[best_idx]
        best_y   = self.y_obs[best_idx]

        for i in range(self.n_iterations):
            X_all = np.array(self.X_obs)
            y_all = np.array(self.y_obs)
            self.gp.fit(X_all, y_all)

            # Candidats aléatoires (approx. de l'optimisation EI)
            X_cand = self._random_sample(1000)
            ei     = self._expected_improvement(X_cand, best_y)
            x_next = X_cand[np.argmax(ei)]

            y_next = self.score_fn(x_next)
            self.X_obs.append(x_next)
            self.y_obs.append(y_next)

            if y_next > best_y:
                best_x = x_next
                best_y = y_next
                logger.debug(f"  BayesOpt iter {i+1}: nouveau max = {best_y:.4f}")

        return best_x, best_y


# ══════════════════════════════════════════════════════════════════════
#  3. OPTIMISATION MULTI-OBJECTIF (PARETO)
# ══════════════════════════════════════════════════════════════════════

class ParetoOptimizer:
    """
    Identifie le front de Pareto dans un espace multi-objectif.

    Pour des molécules, on veut optimiser simultanément :
      - Efficacité ↑
      - Toxicité ↓
      - Solubilité ↑
      - etc.

    Référence :
      Daulton et al. (2020) "Differentiable Expected Hypervolume Improvement
      for Parallel Multi-Objective Bayesian Optimization", NeurIPS 2020.
    """

    @staticmethod
    def is_dominated(point: np.ndarray, others: np.ndarray) -> bool:
        """Un point est dominé s'il existe un autre point meilleur sur TOUS les objectifs."""
        return np.any(np.all(others >= point, axis=1) & np.any(others > point, axis=1))

    @classmethod
    def pareto_front(cls, objectives: np.ndarray) -> np.ndarray:
        """
        Calcule le front de Pareto.

        Args:
            objectives : [N, M] — N points, M objectifs (à maximiser)

        Returns:
            Indices sur le front de Pareto
        """
        n = len(objectives)
        pareto_mask = np.ones(n, dtype=bool)

        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                # i est-il dominé par j ?
                if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    pareto_mask[i] = False
                    break

        return np.where(pareto_mask)[0]

    @classmethod
    def hypervolume(cls, objectives: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Calcule le hypervolume du front de Pareto par rapport à un point de référence.
        Mesure standard de la qualité d'un front de Pareto.
        (Approximation pour M ≤ 4 par inclusion-exclusion.)
        """
        pareto_idx = cls.pareto_front(objectives)
        pf         = objectives[pareto_idx]
        n, m       = pf.shape

        if m == 2:
            # Exact en 2D : tri + intégration
            pf_sorted = pf[np.argsort(pf[:, 0])]
            hv = 0.0
            y_prev = reference_point[1]
            for point in reversed(pf_sorted):
                if point[0] > reference_point[0]:
                    break
                hv    += (reference_point[0] - point[0]) * max(0, point[1] - y_prev)
                y_prev = max(y_prev, point[1])
            return float(hv)

        # Pour M > 2 : approximation Monte Carlo
        n_samples = 50_000
        samples   = np.random.uniform(reference_point, pf.max(axis=0), size=(n_samples, m))
        dominated = np.any(
            np.all(pf[:, None, :] >= samples[None, :, :], axis=2), axis=0
        )
        vol_total = float(np.prod(pf.max(axis=0) - reference_point))
        return float(vol_total * dominated.mean())


# ══════════════════════════════════════════════════════════════════════
#  4. ENSEMBLE PREDICTOR (avec calibration d'incertitude)
# ══════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Ensemble de modèles pour estimer l'incertitude des prédictions.

    Implémente "Deep Ensembles" (Lakshminarayanan et al. 2017) :
      - Entraîne plusieurs modèles avec différentes initialisations
      - La variance entre les prédictions mesure l'incertitude

    Usage :
        ensemble = EnsemblePredictor(predictors_list)
        mean, std = ensemble.predict_with_uncertainty("CCO")
    """

    def __init__(self, predictors: list):
        """
        Args:
            predictors : list de PanaceePredictor déjà chargés
        """
        if len(predictors) < 2:
            raise ValueError("EnsemblePredictor requiert au moins 2 prédicteurs")
        self.predictors = predictors

    def predict_with_uncertainty(self, smiles: str) -> Tuple[dict, dict]:
        """
        Returns:
            mean_predictions, std_predictions
        """
        all_preds = [p.predict(smiles) for p in self.predictors]

        # Exclure les erreurs
        valid = [r for r in all_preds if "error" not in r]
        if not valid:
            return {"error": f"SMILES invalide : {smiles}"}, {}

        # Clés scalaires (hors dict de toxicité)
        scalar_keys = ["efficacy", "solubility", "lipophilicity",
                       "bioavailability", "metabolic_stability",
                       "toxicity_max", "toxicity_mean", "global_score"]

        means, stds = {}, {}
        for k in scalar_keys:
            vals = [r[k] for r in valid if k in r]
            if vals:
                means[k] = round(float(np.mean(vals)), 4)
                stds[k]  = round(float(np.std(vals)), 4)

        return means, stds

    def calibration_error(self, smiles_list: List[str], true_values: dict) -> dict:
        """
        Calcule l'erreur de calibration (ECE) de l'ensemble.
        Plus l'ECE est proche de 0, meilleure est la calibration.
        """
        ece_values = {}
        for key, truth in true_values.items():
            preds_all = []
            for smi, y in zip(smiles_list, truth):
                means, stds = self.predict_with_uncertainty(smi)
                if key in means:
                    preds_all.append((means[key], stds[key], y))

            if not preds_all:
                continue

            # ECE simplifié : moyenne |pred - true| / std
            calibrations = [
                abs(m - t) / max(s, 1e-6)
                for m, s, t in preds_all
            ]
            ece_values[key] = round(float(np.mean(calibrations)), 4)

        return ece_values


# ══════════════════════════════════════════════════════════════════════
#  5. GRADIENT ACTIVATION MAP (Interprétabilité GNN)
# ══════════════════════════════════════════════════════════════════════

class GradientActivationMap:
    """
    Calcule l'importance de chaque atome par rétropropagation.

    Basé sur GradCAM adapté aux GNN :
      Pope et al. (2019) "Explainability Methods for Graph Convolutional Neural Networks"
      CVPR 2019.

    Usage :
        gam = GradientActivationMap(predictor.model, target_prop="toxicity", target_task=0)
        atom_scores = gam.compute("CC(=O)Nc1ccc(O)cc1")
        # atom_scores[i] = importance relative de l'atome i
    """

    def __init__(
        self,
        model: nn.Module,
        target_prop: str = "toxicity",
        target_task: int = 0,
    ):
        self.model       = model
        self.target_prop = target_prop
        self.target_task = target_task
        self._gradients  = None
        self._activations = None

        # Hooks sur la dernière couche de l'encodeur
        self._register_hooks()

    def _register_hooks(self):
        def save_grad(grad):
            self._gradients = grad.detach().cpu()

        def save_activation(module, input, output):
            # output peut être un tenseur ou un tuple
            out = output[0] if isinstance(output, (tuple, list)) else output
            out.register_hook(save_grad)
            self._activations = out.detach().cpu()

        # Trouver la dernière couche conv
        for name, module in reversed(list(self.model.encoder.named_modules())):
            if "conv" in name.lower() or "layer" in name.lower():
                module.register_forward_hook(save_activation)
                logger.debug(f"GAM hook enregistré sur : {name}")
                break

    def compute(self, smiles: str) -> Optional[np.ndarray]:
        """
        Calcule les scores d'importance par atome.

        Returns:
            np.ndarray [n_atoms] ou None si SMILES invalide
        """
        from graph_builder import smiles_to_graph
        from torch_geometric.data import Batch

        g = smiles_to_graph(smiles)
        if g is None:
            return None

        batch = Batch.from_data_list([g])
        # device
        device = next(self.model.parameters()).device
        batch  = batch.to(device)

        self.model.eval()

        # Forward + collecte des gradients
        batch.x.requires_grad_(True)
        preds = self.model(batch)

        if self.target_prop == "toxicity":
            score = preds[self.target_prop][0, self.target_task]
        else:
            score = preds[self.target_prop].squeeze()

        score.backward()

        if self._gradients is None or self._activations is None:
            return None

        # GradCAM : moyenne des gradients × activations
        weights = self._gradients.mean(dim=0, keepdim=True)      # [1, H]
        cam     = (weights * self._activations).sum(dim=1)        # [N]
        cam     = torch.relu(cam).numpy()

        # Normaliser en [0, 1]
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam
