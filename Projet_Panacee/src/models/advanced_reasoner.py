"""
Module de Raisonnement Avancé pour la découverte de médicaments.

Algorithmes implémentés :
  1. Monte Carlo Tree Search (MCTS) - exploration combinatoire intelligente
  2. Optimisation Bayésienne - optimisation dose/combinaison
  3. Optimisation multi-objectif de Pareto - compromis multi-propriétés
  4. Méthodes d'ensemble avec calibration de confiance
  5. Méta-raisonnement avec chaîne de pensée

Références :
  - Silver et al. (2016) "Mastering the game of Go with DNNs and tree search"
  - Snoek et al. (2012) "Practical Bayesian Optimization"
  - Preuer et al. (2018) "DeepSynergy"
"""
import math
import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger("panacee.advanced_reasoner")


# ═══════════════════════════════════════════════════
#  1. MONTE CARLO TREE SEARCH (MCTS)
# ═══════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """Nœud de l'arbre MCTS pour exploration combinatoire."""
    molecules: List[int]           # Indices des molécules sélectionnées
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: List[int] = field(default_factory=list)

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound pour sélection."""
        if self.visits == 0:
            return float("inf")
        exploit = self.total_reward / self.visits
        explore = math.sqrt(2 * math.log(self.parent.visits + 1) / self.visits)
        return exploit + 1.41 * explore

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.visits)


class MCTSCombinationSearch:
    """
    MCTS adapté à la recherche de combinaisons moléculaires optimales.

    L'arbre explore les combinaisons possibles de molécules,
    guidé par le modèle de score du raisonneur.

    Usage:
        mcts = MCTSCombinationSearch(score_fn, n_molecules=100, combo_size=3)
        best = mcts.search(n_iterations=500)
    """

    def __init__(
        self,
        score_function,  # Callable[[List[int]], float]
        n_molecules: int,
        combo_size: int = 3,
        exploration_weight: float = 1.41,
    ):
        """
        Args:
            score_function: fonction qui score une combinaison (indices → float)
            n_molecules: nombre total de molécules disponibles
            combo_size: taille des combinaisons à explorer
            exploration_weight: poids de l'exploration (C dans UCB1)
        """
        self.score_fn = score_function
        self.n_molecules = n_molecules
        self.combo_size = combo_size
        self.C = exploration_weight
        self._best_combo = None
        self._best_score = -float("inf")

    def search(self, n_iterations: int = 500) -> Tuple[List[int], float]:
        """
        Lance la recherche MCTS.

        Args:
            n_iterations: nombre d'itérations

        Returns:
            (meilleure combinaison, score)
        """
        root = MCTSNode(
            molecules=[],
            untried_actions=list(range(self.n_molecules)),
        )

        for i in range(n_iterations):
            node = self._select(root)
            child = self._expand(node)
            reward = self._simulate(child)
            self._backpropagate(child, reward)

            if (i + 1) % 100 == 0:
                logger.debug(
                    f"MCTS iter {i+1}/{n_iterations} | "
                    f"best={self._best_score:.4f} combo={self._best_combo}"
                )

        return self._best_combo or [], self._best_score

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Sélection : descendre dans l'arbre en choisissant le meilleur UCB1."""
        while node.children and not node.untried_actions:
            if len(node.molecules) >= self.combo_size:
                return node
            node = max(node.children, key=lambda c: c.ucb1)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion : ajouter un enfant avec une action non essayée."""
        if not node.untried_actions or len(node.molecules) >= self.combo_size:
            return node

        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        child = MCTSNode(
            molecules=node.molecules + [action],
            parent=node,
            untried_actions=[
                a for a in range(self.n_molecules)
                if a not in node.molecules and a != action
            ],
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Simulation : compléter aléatoirement et évaluer."""
        combo = list(node.molecules)
        available = [i for i in range(self.n_molecules) if i not in combo]

        while len(combo) < self.combo_size and available:
            idx = random.randint(0, len(available) - 1)
            combo.append(available.pop(idx))

        if len(combo) == self.combo_size:
            reward = self.score_fn(combo)

            if reward > self._best_score:
                self._best_score = reward
                self._best_combo = list(combo)

            return reward
        return 0.0

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Rétro-propagation : mettre à jour les compteurs vers la racine."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def get_top_k(self, k: int = 10) -> List[Tuple[List[int], float]]:
        """
        Retourne les k meilleures combinaisons trouvées.
        Note: nécessite de refaire la recherche avec suivi.
        """
        return [(self._best_combo or [], self._best_score)]


# ═══════════════════════════════════════════════════
#  2. OPTIMISATION BAYÉSIENNE (sans dépendance externe)
# ═══════════════════════════════════════════════════

class BayesianDoseOptimizer:
    """
    Optimisation bayésienne pour trouver les doses optimales.

    Utilise un processus gaussien simplifié (noyau RBF)
    pour modéliser la relation dose → efficacité.

    Usage:
        opt = BayesianDoseOptimizer(dose_levels=[0.1, 0.5, 1.0, 5.0, 10.0])
        for _ in range(20):
            dose = opt.suggest_next()
            efficacy = evaluate(dose)  # ton modèle
            opt.observe(dose, efficacy)
        best_dose = opt.get_best()
    """

    def __init__(
        self,
        dose_levels: List[float],
        length_scale: float = 1.0,
        noise: float = 0.1,
    ):
        self.dose_levels = np.array(dose_levels, dtype=np.float64)
        self.length_scale = length_scale
        self.noise = noise

        self.X_observed: List[float] = []
        self.Y_observed: List[float] = []

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Noyau RBF (Radial Basis Function)."""
        sq_dist = np.subtract.outer(x1, x2) ** 2
        return np.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def _predict(self, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prédiction GP : retourne (mean, std) pour chaque point test.
        """
        if len(self.X_observed) == 0:
            return np.zeros(len(x_test)), np.ones(len(x_test))

        X = np.array(self.X_observed)
        Y = np.array(self.Y_observed)

        K = self._rbf_kernel(X, X) + self.noise ** 2 * np.eye(len(X))
        K_star = self._rbf_kernel(x_test, X)
        K_ss = self._rbf_kernel(x_test, x_test)

        try:
            K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))
        except np.linalg.LinAlgError:
            return np.zeros(len(x_test)), np.ones(len(x_test))

        mu = K_star @ K_inv @ Y
        cov = K_ss - K_star @ K_inv @ K_star.T
        std = np.sqrt(np.maximum(np.diag(cov), 1e-8))

        return mu, std

    def _acquisition_ucb(self, mu: np.ndarray, std: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        return mu + beta * std

    def suggest_next(self) -> float:
        """Suggère la prochaine dose à évaluer."""
        mu, std = self._predict(self.dose_levels)
        ucb = self._acquisition_ucb(mu, std)
        best_idx = np.argmax(ucb)
        return float(self.dose_levels[best_idx])

    def observe(self, dose: float, efficacy: float):
        """Enregistre une observation."""
        self.X_observed.append(dose)
        self.Y_observed.append(efficacy)

    def get_best(self) -> Tuple[float, float]:
        """Retourne la meilleure dose observée."""
        if not self.Y_observed:
            return self.dose_levels[len(self.dose_levels) // 2], 0.0
        best_idx = int(np.argmax(self.Y_observed))
        return self.X_observed[best_idx], self.Y_observed[best_idx]

    def get_dose_response_curve(self) -> Dict[str, np.ndarray]:
        """Retourne la courbe dose-réponse estimée."""
        fine_grid = np.linspace(
            self.dose_levels.min(), self.dose_levels.max(), 50
        )
        mu, std = self._predict(fine_grid)
        return {"doses": fine_grid, "mean": mu, "std": std}


# ═══════════════════════════════════════════════════
#  3. OPTIMISATION MULTI-OBJECTIF DE PARETO
# ═══════════════════════════════════════════════════

@dataclass
class Solution:
    """Une solution avec ses objectifs."""
    index: int
    objectives: List[float]     # scores pour chaque objectif
    metadata: Dict = field(default_factory=dict)


def dominates(a: Solution, b: Solution, maximize: List[bool]) -> bool:
    """Vérifie si la solution a domine b."""
    at_least_one_better = False
    for i, (va, vb) in enumerate(zip(a.objectives, b.objectives, strict=False)):
        if maximize[i]:
            if va < vb:
                return False
            if va > vb:
                at_least_one_better = True
        else:
            if va > vb:
                return False
            if va < vb:
                at_least_one_better = True
    return at_least_one_better


def pareto_front(
    solutions: List[Solution],
    maximize: Optional[List[bool]] = None,
) -> List[Solution]:
    """
    Calcule le front de Pareto d'un ensemble de solutions.

    Args:
        solutions: liste de solutions avec objectifs
        maximize: pour chaque objectif, True si on maximise

    Returns:
        Solutions non-dominées (front de Pareto)
    """
    if not solutions:
        return []

    n_obj = len(solutions[0].objectives)
    if maximize is None:
        maximize = [True] * n_obj

    front = []
    for candidate in solutions:
        is_dominated = False
        new_front = []
        for existing in front:
            if dominates(existing, candidate, maximize):
                is_dominated = True
                new_front.append(existing)
            elif dominates(candidate, existing, maximize):
                continue  # existing est dominé, on l'enlève
            else:
                new_front.append(existing)

        if not is_dominated:
            new_front.append(candidate)
        front = new_front

    return front


class MultiObjectiveOptimizer:
    """
    Optimiseur multi-objectif pour la découverte de médicaments.

    Objectifs typiques :
      - Maximiser efficacité
      - Minimiser toxicité
      - Maximiser solubilité
      - Maximiser biodisponibilité orale
      - Minimiser effets secondaires
    """

    def __init__(self, objective_names: List[str], maximize: List[bool]):
        """
        Args:
            objective_names: noms des objectifs
            maximize: True pour maximiser, False pour minimiser
        """
        self.objective_names = objective_names
        self.maximize = maximize
        self.solutions: List[Solution] = []

    def add_solution(self, index: int, objectives: List[float], metadata: Dict | None = None):
        """Ajoute une solution évaluée."""
        self.solutions.append(Solution(
            index=index,
            objectives=objectives,
            metadata=metadata or {},
        ))

    def get_pareto_front(self) -> List[Solution]:
        """Retourne le front de Pareto actuel."""
        return pareto_front(self.solutions, self.maximize)

    def rank_solutions(self) -> List[Tuple[Solution, int]]:
        """
        Classe les solutions par rang de domination.
        Rang 0 = front de Pareto, rang 1 = second front, etc.

        Returns:
            Liste de (solution, rang)
        """
        remaining = list(self.solutions)
        ranked = []
        current_rank = 0

        while remaining:
            front = pareto_front(remaining, self.maximize)
            for sol in front:
                ranked.append((sol, current_rank))
            front_set = {id(s) for s in front}
            remaining = [s for s in remaining if id(s) not in front_set]
            current_rank += 1

        return ranked

    def suggest_best_compromise(self, weights: Optional[List[float]] = None) -> Optional[Solution]:
        """
        Suggère le meilleur compromis sur le front de Pareto.

        Args:
            weights: poids pour chaque objectif (optionnel)

        Returns:
            Meilleure solution compromis
        """
        front = self.get_pareto_front()
        if not front:
            return None

        if weights is None:
            weights = [1.0] * len(self.objective_names)

        # Normaliser les objectifs sur le front
        n_obj = len(self.objective_names)
        obj_matrix = np.array([s.objectives for s in front])

        if len(front) > 1:
            mins = obj_matrix.min(axis=0)
            maxs = obj_matrix.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normalized = (obj_matrix - mins) / ranges
        else:
            normalized = np.ones((1, n_obj)) * 0.5

        # Score pondéré (inverser les objectifs à minimiser)
        scores = np.zeros(len(front))
        for i in range(n_obj):
            col = normalized[:, i]
            if not self.maximize[i]:
                col = 1 - col
            scores += weights[i] * col

        best_idx = int(np.argmax(scores))
        return front[best_idx]


# ═══════════════════════════════════════════════════
#  4. ENSEMBLE DE CONFIANCE
# ═══════════════════════════════════════════════════

class EnsembleConfidence(nn.Module):
    """
    Combine les prédictions de multiples sources/modèles
    avec calibration de confiance.

    Méthodes :
      - MC Dropout : multiples passes avec dropout actif
      - Température scaling pour calibration
      - Incertitude épistémique vs aléatoire
    """

    def __init__(self, d_model: int, n_sources: int = 3):
        super().__init__()
        self.n_sources = n_sources

        # Poids d'attention pour chaque source
        self.source_attention = nn.Sequential(
            nn.Linear(d_model * n_sources, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_sources),
            nn.Softmax(dim=-1),
        )

        # Température apprise pour calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Incertitude
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 2),  # [épistémique, aléatoire]
            nn.Softplus(),
        )

    def forward(self, source_predictions: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            source_predictions: liste de [B, D] prédictions de chaque source

        Returns:
            combined: [B, D] prédiction combinée
            confidence: [B, 1] confiance calibrée
            uncertainties: [B, 2] épistémique et aléatoire
        """
        # Empiler et combiner
        stacked = torch.stack(source_predictions, dim=1)  # [B, n_sources, D]
        B, S, D = stacked.shape

        # Concaténer pour attention
        concat = stacked.reshape(B, S * D)
        weights = self.source_attention(concat)  # [B, n_sources]

        # Combinaison pondérée
        combined = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # Incertitude
        uncertainties = self.uncertainty_head(combined)

        # Confiance = fonction inverse de l'incertitude totale
        total_uncertainty = uncertainties.sum(dim=-1, keepdim=True)
        confidence = torch.sigmoid(-total_uncertainty / self.temperature.abs().clamp(min=0.1))

        return {
            "combined": combined,
            "confidence": confidence,
            "uncertainties": uncertainties,
            "source_weights": weights,
        }

    @staticmethod
    def mc_dropout_predict(model: nn.Module, x, n_forward: int = 10) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout : multiples passes en mode train
        pour estimer l'incertitude.

        Args:
            model: modèle avec dropout
            x: entrée
            n_forward: nombre de passes

        Returns:
            mean, std des prédictions
        """
        model.train()  # Active le dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_forward):
                out = model(x)
                if isinstance(out, dict):
                    out = out.get("success_score", out.get("confidence"))
                predictions.append(out)

        model.eval()

        preds = torch.stack(predictions)
        return {
            "mean": preds.mean(dim=0),
            "std": preds.std(dim=0),
            "epistemic_uncertainty": preds.var(dim=0),
        }


# ═══════════════════════════════════════════════════
#  5. MÉTA-RAISONNEMENT (Chaîne de Pensée)
# ═══════════════════════════════════════════════════

@dataclass
class ReasoningStep:
    """Étape de raisonnement dans la chaîne de pensée."""
    step_name: str
    description: str
    score: float
    evidence: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)


class ChainOfThought:
    """
    Raisonnement structuré en chaîne de pensée pour l'évaluation
    de combinaisons moléculaires.

    Étapes :
    1. Évaluation individuelle de chaque molécule
    2. Analyse des interactions par paires
    3. Vérification des contraintes ADMET/Lipinski
    4. Analyse de synergie/antagonisme
    5. Optimisation de dose
    6. Score de confiance final
    """

    def __init__(self):
        self.steps: List[ReasoningStep] = []
        self._scores: Dict[str, float] = {}

    def reset(self):
        """Réinitialise la chaîne."""
        self.steps = []
        self._scores = {}

    def add_step(
        self, name: str, description: str, score: float,
        evidence: List[str] | None = None, sub_scores: Dict[str, float] | None = None
    ):
        """Ajoute une étape de raisonnement."""
        step = ReasoningStep(
            step_name=name,
            description=description,
            score=score,
            evidence=evidence or [],
            sub_scores=sub_scores or {},
        )
        self.steps.append(step)
        self._scores[name] = score

    def get_final_score(self, weights: Dict[str, float] | None = None) -> float:
        """
        Score final pondéré de la chaîne de pensée.

        Args:
            weights: poids pour chaque étape

        Returns:
            Score final [0, 1]
        """
        if not self.steps:
            return 0.0

        if weights is None:
            weights = {s.step_name: 1.0 for s in self.steps}

        total_weight = sum(weights.get(s.step_name, 1.0) for s in self.steps)
        if total_weight == 0:
            return 0.0

        score = sum(
            s.score * weights.get(s.step_name, 1.0)
            for s in self.steps
        ) / total_weight

        return max(0.0, min(1.0, score))

    def generate_report(self) -> str:
        """Génère un rapport textuel de la chaîne de pensée."""
        lines = ["═" * 60, "  CHAÎNE DE PENSÉE - RAPPORT DE RAISONNEMENT", "═" * 60, ""]

        for i, step in enumerate(self.steps, 1):
            grade = "★★★" if step.score > 0.7 else "★★" if step.score > 0.4 else "★"
            lines.append(f"Étape {i}: {step.step_name} {grade} ({step.score:.2f})")
            lines.append(f"  → {step.description}")

            if step.sub_scores:
                for k, v in step.sub_scores.items():
                    lines.append(f"    • {k}: {v:.3f}")

            if step.evidence:
                for e in step.evidence:
                    lines.append(f"    📋 {e}")

            lines.append("")

        final = self.get_final_score()
        verdict = (
            "EXCELLENTE COMBINAISON" if final > 0.8 else
            "BONNE COMBINAISON" if final > 0.6 else
            "COMBINAISON ACCEPTABLE" if final > 0.4 else
            "COMBINAISON RISQUÉE"
        )
        lines.extend([
            "─" * 60,
            f"  SCORE FINAL: {final:.3f} — {verdict}",
            "─" * 60,
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════
#  6. RAISONNEUR AVANCÉ INTÉGRÉ
# ═══════════════════════════════════════════════════

class AdvancedMolecularReasoner:
    """
    Orchestrateur de haut niveau qui combine :
      - Le MolecularReasoner (réseau de neurones)
      - MCTS (exploration combinatoire)
      - Optimisation Bayésienne (doses)
      - Pareto (multi-objectif)
      - Chaîne de pensée (raisonnement structuré)
      - Base de connaissances médicales
      - Recherche web (vérification)

    Usage:
        reasoner = AdvancedMolecularReasoner(nn_reasoner, encoder, device)
        results = reasoner.full_analysis(
            smiles_list=["CCO", "CC(=O)O"],
            indication="anti-inflammatory"
        )
    """

    def __init__(
        self,
        nn_reasoner: nn.Module,
        encoder: nn.Module,
        device: torch.device,
        dose_levels: List[float] | None = None,
    ):
        self.nn_reasoner = nn_reasoner
        self.encoder = encoder
        self.device = device
        self.dose_levels = dose_levels or [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    def _encode_molecules(self, graphs) -> torch.Tensor:
        """Encode une liste de graphes moléculaires."""
        self.encoder.eval()
        embeddings = []
        with torch.no_grad():
            for g in graphs:
                g = g.to(self.device)
                emb = self.encoder(g)
                embeddings.append(emb)
        return torch.stack(embeddings).unsqueeze(0)  # [1, N, D]

    def full_analysis(
        self,
        mol_embeddings: torch.Tensor,
        mol_names: List[str] | None = None,
        indication: str = "",
        use_mcts: bool = False,
        use_bayesian: bool = True,
        use_web_search: bool = False,
    ) -> Dict:
        """
        Analyse complète d'une combinaison de molécules.

        Args:
            mol_embeddings: [1, N, D] ou [N, D] embeddings
            mol_names: noms/SMILES des molécules
            indication: indication thérapeutique visée
            use_mcts: utiliser MCTS pour explorer d'autres combinaisons
            use_bayesian: optimiser les doses avec Bayesian opt
            use_web_search: vérifier avec les APIs web

        Returns:
            Résultats d'analyse complets avec chaîne de pensée
        """
        if mol_embeddings.dim() == 2:
            mol_embeddings = mol_embeddings.unsqueeze(0)

        B, N, D = mol_embeddings.shape
        mol_names = mol_names or [f"Mol_{i}" for i in range(N)]

        # Initialiser la chaîne de pensée
        cot = ChainOfThought()

        # ── Étape 1 : Prédictions du réseau de neurones ──
        self.nn_reasoner.eval()
        with torch.no_grad():
            nn_results = self.nn_reasoner(mol_embeddings.to(self.device))

        synergy_score = nn_results["synergy_matrix"][0].mean().item()
        success_score = nn_results["success_score"][0, 0].item()
        confidence = nn_results["confidence"][0, 0].item()

        cot.add_step(
            name="neural_network_prediction",
            description=f"Prédictions du Transformer ({N} molécules analysées)",
            score=success_score,
            sub_scores={
                "synergie_moyenne": synergy_score,
                "score_réussite": success_score,
                "confiance": confidence,
            },
        )

        # ── Étape 2 : MC Dropout pour incertitude ──
        mc_results = EnsembleConfidence.mc_dropout_predict(
            self.nn_reasoner, mol_embeddings.to(self.device), n_forward=5
        )
        uncertainty = mc_results["std"].mean().item()

        cot.add_step(
            name="uncertainty_estimation",
            description="Estimation d'incertitude par MC Dropout (5 passes)",
            score=max(0, 1 - uncertainty * 5),  # Haute certitude = bon score
            sub_scores={
                "incertitude_épistémique": uncertainty,
                "stabilité_prédiction": max(0, 1 - uncertainty),
            },
        )

        # ── Étape 3 : Analyse des interactions par paires ──
        synergy_matrix = nn_results["synergy_matrix"][0].cpu().numpy()
        pair_analyses = []
        for i in range(N):
            for j in range(i + 1, N):
                pair_analyses.append({
                    "pair": f"{mol_names[i]} + {mol_names[j]}",
                    "synergy": float(synergy_matrix[i, j]),
                    "interpretation": (
                        "synergie" if synergy_matrix[i, j] > 0.6 else
                        "neutre" if synergy_matrix[i, j] > 0.4 else
                        "antagonisme possible"
                    ),
                })

        avg_pair_score = np.mean([p["synergy"] for p in pair_analyses]) if pair_analyses else 0.5
        cot.add_step(
            name="pairwise_interaction",
            description=f"Analyse de {len(pair_analyses)} paires d'interactions",
            score=avg_pair_score,
            evidence=[
                f"{p['pair']}: {p['interpretation']} ({p['synergy']:.2f})"
                for p in pair_analyses[:5]
            ],
        )

        # ── Étape 4 : Optimisation Bayésienne des doses ──
        dose_results = {}
        if use_bayesian:
            dose_dists = nn_results["dose_distributions"][0].cpu().numpy()
            for i in range(N):
                opt = BayesianDoseOptimizer(self.dose_levels)
                # Utiliser les prédictions du NN comme observations initiales
                for j, dose in enumerate(self.dose_levels):
                    if j < len(dose_dists[i]):
                        opt.observe(dose, float(dose_dists[i, j]))

                best_dose, best_eff = opt.get_best()
                dose_results[mol_names[i]] = {
                    "optimal_dose": best_dose,
                    "predicted_efficacy": best_eff,
                }

            dose_score = np.mean([
                d["predicted_efficacy"] for d in dose_results.values()
            ])
            cot.add_step(
                name="dose_optimization",
                description="Optimisation Bayésienne des doses (GP + UCB)",
                score=dose_score,
                sub_scores={
                    name: info["optimal_dose"]
                    for name, info in dose_results.items()
                },
                evidence=[
                    f"{name}: dose optimale = {info['optimal_dose']} mg/kg"
                    for name, info in dose_results.items()
                ],
            )

        # ── Étape 5 : Connaissances médicales ──
        knowledge_score = 0.5  # Par défaut
        try:
            from src.knowledge.medical_rules import comprehensive_evaluation
            if mol_names and not mol_names[0].startswith("Mol_"):
                evaluations = []
                for smiles in mol_names:
                    ev = comprehensive_evaluation(smiles)
                    if ev.get("valid"):
                        evaluations.append(ev)

                if evaluations:
                    knowledge_score = np.mean([
                        e["overall_score"] for e in evaluations
                    ])
                    cot.add_step(
                        name="medical_knowledge",
                        description="Évaluation par base de connaissances (Lipinski, ADMET)",
                        score=knowledge_score,
                        sub_scores={
                            e["smiles"][:20]: e["overall_score"]
                            for e in evaluations[:5]
                        },
                        evidence=[
                            f"{e['smiles'][:20]}: drug-likeness={e['drug_likeness']:.2f}, "
                            f"biodisp={e['estimated_bioavailability_pct']:.0f}%"
                            for e in evaluations[:5]
                        ],
                    )
        except ImportError:
            pass

        # ── Étape 6 : Recherche web (optionnel) ──
        if use_web_search:
            try:
                from src.knowledge.web_search import WebResearchEngine
                engine = WebResearchEngine()
                web_evidence = []

                for smiles in mol_names[:3]:  # Limiter à 3 pour la vitesse
                    if not smiles.startswith("Mol_"):
                        research = engine.research_molecule(smiles)
                        if research.get("sources"):
                            web_evidence.append(
                                f"{smiles[:20]}: trouvé dans {', '.join(research['sources'])}"
                            )

                if web_evidence:
                    cot.add_step(
                        name="web_verification",
                        description="Vérification dans les bases de données publiques",
                        score=0.7,
                        evidence=web_evidence,
                    )
            except Exception as e:
                logger.debug(f"Recherche web ignorée: {e}")

        # ── Résultat final ──
        final_score = cot.get_final_score(weights={
            "neural_network_prediction": 3.0,
            "uncertainty_estimation": 1.5,
            "pairwise_interaction": 2.0,
            "dose_optimization": 1.5,
            "medical_knowledge": 2.0,
            "web_verification": 1.0,
        })

        return {
            "final_score": final_score,
            "nn_predictions": {
                "synergy_matrix": synergy_matrix.tolist(),
                "success_score": success_score,
                "confidence": confidence,
                "dose_distributions": nn_results["dose_distributions"][0].cpu().numpy().tolist(),
            },
            "pair_analyses": pair_analyses,
            "dose_optimization": dose_results,
            "uncertainty": uncertainty,
            "chain_of_thought": cot,
            "report": cot.generate_report(),
        }
