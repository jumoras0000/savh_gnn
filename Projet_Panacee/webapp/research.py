# -*- coding: utf-8 -*-
"""
Module Recherche — analyse de molécules réelles pour la découverte de médicaments.

S'appuie sur src.analysis.combinatorial_engine.PanaceeAnalyzer (modèle Phase 3) :
  - predict()  : propriétés + évaluation de risque par molécule
  - combo()    : synergie / doses / score de réussite d'une combinaison

L'analyseur est coûteux à charger → on le met en cache par checkpoint. Tout est
importé paresseusement : si torch / le checkpoint manquent, on renvoie une erreur
explicite au lieu de planter.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CACHE: dict = {}

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def _phase3_default() -> str:
    from src.config import CHECKPOINT_DIR, PHASE3
    return str(CHECKPOINT_DIR / "phase3" / PHASE3["checkpoint_name"])


def _phase2_default() -> str:
    from src.config import CHECKPOINT_DIR
    return str(CHECKPOINT_DIR / "phase2" / "best_toxicity_model.pth")


def detect_checkpoint_kind(path: str) -> str:
    """phase3 (multi-propriétés+IA) / phase2 (toxicité) / encoder (Phase 1) / unknown."""
    import torch
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return "unknown"
    if not isinstance(ckpt, dict):
        return "unknown"
    if "reasoner_state_dict" in ckpt or "reasoner_config" in ckpt or "property_dims" in ckpt:
        return "phase3"
    if "task_names" in ckpt or "num_tasks" in ckpt or "optimal_thresholds" in ckpt:
        return "phase2"
    if "model_state_dict" in ckpt:
        return "encoder"
    return "unknown"


def get_analyzer(checkpoint: str | None = None):
    """Renvoie un PanaceeAnalyzer Phase 3 (mis en cache). Lève si indisponible."""
    ckpt = checkpoint or _phase3_default()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint Phase 3 introuvable : {ckpt}. "
            f"Entraîne d'abord la Phase 3 (onglet Entraînement) ou importe un .pth."
        )
    key = ("phase3", ckpt)
    if key in _CACHE:
        return _CACHE[key]
    from src.analysis.combinatorial_engine import PanaceeAnalyzer
    analyzer = PanaceeAnalyzer(checkpoint_path=ckpt, device="cpu")
    _CACHE[key] = analyzer
    return analyzer


class _Phase2Predictor:
    """Prédicteur toxicité seule, reconstruit depuis un checkpoint Phase 2."""

    def __init__(self, ckpt_path: str):
        import torch
        from src.models.encoder import MolecularEncoder
        from src.models.toxicity_classifier import ToxicityClassifier
        from src.config import (
            ATOM_FEATURE_DIM, BOND_FEATURE_DIM, HIDDEN_DIM,
            NUM_GNN_LAYERS, OUTPUT_DIM, DROPOUT, CONV_TYPE, ATTENTION_HEADS,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.task_names = ckpt.get("task_names") or TOX21_TASKS
        num_tasks = ckpt.get("num_tasks", len(self.task_names))
        self.thresholds = ckpt.get("optimal_thresholds") or [0.5] * num_tasks

        encoder = MolecularEncoder(
            atom_dim=ATOM_FEATURE_DIM, hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
            num_layers=cfg.get("num_layers", NUM_GNN_LAYERS), edge_dim=BOND_FEATURE_DIM,
            output_dim=cfg.get("output_dim", OUTPUT_DIM), dropout=cfg.get("dropout", DROPOUT),
            conv_type=cfg.get("conv_type", CONV_TYPE),
            attention_heads=cfg.get("attention_heads", ATTENTION_HEADS),
        )
        self.model = ToxicityClassifier(encoder=encoder, num_tasks=num_tasks,
                                        hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM))
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()
        self._torch = torch

    def predict_properties(self, smiles: str):
        from src.preprocessing.graph_builder import smiles_to_graph
        from torch_geometric.data import Batch
        torch = self._torch
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None
        with torch.no_grad():
            logits = self.model(Batch.from_data_list([graph]))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        result = {"smiles": smiles, "toxicity": {}}
        for i, name in enumerate(self.task_names):
            if i < len(probs):
                thr = self.thresholds[i] if i < len(self.thresholds) else 0.5
                result["toxicity"][name] = {"probabilite": round(float(probs[i]) * 100, 1),
                                            "toxique": bool(probs[i] > thr)}
        result["safety_score"] = round((1 - float(probs.mean())) * 100, 1)
        return result


def get_phase2(checkpoint: str | None = None) -> _Phase2Predictor:
    ckpt = checkpoint or _phase2_default()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint Phase 2 introuvable : {ckpt}.")
    key = ("phase2", ckpt)
    if key not in _CACHE:
        _CACHE[key] = _Phase2Predictor(ckpt)
    return _CACHE[key]


def _resolve_model(checkpoint: str | None):
    """
    Choisit le meilleur prédicteur disponible.
    Renvoie (mode, predictor) avec mode ∈ {phase3, phase2, descriptors}.
    Un checkpoint explicitement fourni mais introuvable lève FileNotFoundError.
    """
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")
        kind = detect_checkpoint_kind(checkpoint)
        if kind == "phase3":
            return "phase3", get_analyzer(checkpoint)
        if kind == "phase2":
            return "phase2", get_phase2(checkpoint)
        return "descriptors", None  # encodeur Phase 1 / inconnu → descripteurs seuls
    # auto : Phase 3 > Phase 2 > descripteurs
    if os.path.exists(_phase3_default()):
        return "phase3", get_analyzer()
    if os.path.exists(_phase2_default()):
        return "phase2", get_phase2()
    return "descriptors", None


# ──────────────────────────────────────────────────────────────────────
# Évaluation de risque (observations cliniques par molécule)
# ──────────────────────────────────────────────────────────────────────

def assess_risk(result: dict) -> dict:
    """
    Transforme une prédiction brute en observations + niveau de risque.
    Renvoie {level: OK|WARN|DANGER, score, observations:[{level,text}]}.
    """
    obs: list[dict] = []
    level = "OK"

    def bump(new):
        nonlocal level
        order = {"OK": 0, "WARN": 1, "DANGER": 2}
        if order[new] > order[level]:
            level = new

    # Toxicité par endpoint
    tox = result.get("toxicity", {})
    toxic_eps = [name for name, d in tox.items() if d.get("toxique")]
    if toxic_eps:
        sev = "DANGER" if len(toxic_eps) >= 3 else "WARN"
        bump(sev)
        obs.append({"level": sev,
                    "text": f"Toxicité prédite sur {len(toxic_eps)} endpoint(s) : "
                            f"{', '.join(toxic_eps[:6])}{'…' if len(toxic_eps) > 6 else ''}."})
    else:
        obs.append({"level": "OK", "text": "Aucun signal de toxicité Tox21 détecté."})

    # Score de sécurité
    safety = result.get("safety_score")
    if safety is not None:
        if safety < 50:
            bump("DANGER"); obs.append({"level": "DANGER", "text": f"Score de sécurité bas ({safety}%)."})
        elif safety < 70:
            bump("WARN"); obs.append({"level": "WARN", "text": f"Score de sécurité moyen ({safety}%)."})

    # Biodisponibilité
    bio = result.get("bioavailability", {}).get("probabilite")
    if bio is not None and bio < 40:
        bump("WARN"); obs.append({"level": "WARN", "text": f"Biodisponibilité orale faible ({bio}%)."})

    # Lipophilicité (Ro5 : LogP élevé = accumulation)
    logp = result.get("lipophilicity", {}).get("log_p")
    if logp is not None and logp > 5:
        bump("WARN"); obs.append({"level": "WARN",
                                  "text": f"LogP élevé ({logp}) : risque d'accumulation / faible solubilité."})

    # Solubilité
    sol = result.get("solubility", {}).get("interpretation", "")
    if sol in ("Peu soluble", "Insoluble"):
        bump("WARN"); obs.append({"level": "WARN", "text": f"Solubilité défavorable ({sol})."})

    # Efficacité
    eff = result.get("efficacy", {}).get("probabilite_activite")
    if eff is not None and eff < 40:
        obs.append({"level": "WARN", "text": f"Activité biologique prédite faible ({eff}%)."})

    dl = result.get("drug_likeness", {}).get("score_global", "—")
    return {"level": level, "drug_likeness": dl, "observations": obs}


# ──────────────────────────────────────────────────────────────────────
# API haut-niveau (appelée par le serveur)
# ──────────────────────────────────────────────────────────────────────

def predict(smiles_list: list[str], checkpoint: str | None = None) -> dict:
    """
    Analyse des molécules. Descripteurs RDKit TOUJOURS calculés ; les propriétés
    prédites dépendent du modèle disponible (Phase 3 complet, Phase 2 toxicité,
    sinon descripteurs seuls).
    """
    from webapp import cheminfo
    mode, predictor = _resolve_model(checkpoint)

    results, invalid = [], []
    for smi in smiles_list:
        desc = cheminfo.descriptors(smi)
        if not desc.get("valid"):
            invalid.append(smi)
            continue
        r = {"smiles": smi}
        if predictor is not None:
            pr = predictor.predict_properties(smi)
            if pr is not None:
                r = pr
        r["descriptors"] = desc
        r["risk"] = assess_risk(r)
        results.append(r)

    notes = {
        "phase3": "Modèle Phase 3 complet (toxicité, efficacité, ADME, raisonnement).",
        "phase2": "Modèle Phase 2 (toxicité seule) + descripteurs RDKit.",
        "descriptors": "Aucun modèle entraîné détecté → descripteurs RDKit uniquement. "
                       "Entraîne la Phase 2/3 pour les prédictions de toxicité/efficacité.",
    }
    return {"results": results, "invalid": invalid, "mode": mode, "note": notes[mode],
            "checkpoint": checkpoint or ""}


def combo(smiles_list: list[str], checkpoint: str | None = None) -> dict:
    """Analyse de combinaison (synergie/doses) — nécessite un modèle Phase 3."""
    analyzer = get_analyzer(checkpoint)
    res = analyzer.analyze_combination(smiles_list)
    if res is None:
        return {"error": "combinaison invalide (≥ 2 molécules valides requises)"}
    for mol in res.get("molecules", []):
        mol["risk"] = assess_risk(mol)
    return res


# ──────────────────────────────────────────────────────────────────────
# Criblage virtuel (dont anti-VIH)
# ──────────────────────────────────────────────────────────────────────

_OBJECTIVES = {
    "efficacy": "Efficacité anti-VIH (Phase 3)",
    "safety": "Sécurité (faible toxicité)",
    "drug_likeness": "Drug-likeness (QED, sans modèle)",
}


def screen(molecules: list, objective: str = "drug_likeness",
           checkpoint: str | None = None, top_k: int = 50) -> dict:
    """
    Crible une bibliothèque de molécules et la classe selon un objectif.
      - drug_likeness : QED RDKit (aucun modèle requis).
      - safety        : score de sécurité (Phase 2 ou 3).
      - efficacy      : efficacité anti-VIH (Phase 3 requis).
    `molecules` : liste de str (SMILES) ou de {name, smiles}.
    """
    from webapp import cheminfo
    if objective not in _OBJECTIVES:
        return {"error": f"objectif inconnu: {objective}"}

    # Normaliser les entrées
    items = []
    for m in molecules:
        if isinstance(m, dict):
            items.append({"name": m.get("name", ""), "smiles": m.get("smiles", "")})
        else:
            items.append({"name": "", "smiles": str(m)})

    # Modèle requis ?
    predictor, mode = None, "descriptors"
    if objective in ("safety", "efficacy"):
        mode, predictor = _resolve_model(checkpoint)
        if objective == "efficacy" and mode != "phase3":
            raise FileNotFoundError(
                "Le criblage par efficacité anti-VIH nécessite un modèle Phase 3. "
                "Entraîne la Phase 3 ou choisis l'objectif « drug_likeness » / « safety »."
            )
        if objective == "safety" and predictor is None:
            raise FileNotFoundError("Le criblage par sécurité nécessite un modèle Phase 2 ou 3.")

    ranked, invalid = [], []
    for it in items:
        desc = cheminfo.descriptors(it["smiles"])
        if not desc.get("valid"):
            invalid.append(it["smiles"])
            continue
        row = {"name": it["name"], "smiles": desc["canonical"],
               "mw": desc["mw"], "logp": desc["logp"], "qed": desc["qed"],
               "lipinski_pass": desc["lipinski_pass"]}
        score = None
        if objective == "drug_likeness":
            score = round((desc["qed"] or 0) * 100, 1)
        else:
            pr = predictor.predict_properties(it["smiles"]) if predictor else None
            if pr:
                if objective == "safety":
                    score = pr.get("safety_score")
                elif objective == "efficacy":
                    score = (pr.get("efficacy") or {}).get("probabilite_activite")
                row["safety_score"] = pr.get("safety_score")
                row["risk"] = assess_risk(pr)["level"]
        row["score"] = score if score is not None else 0.0
        ranked.append(row)

    ranked.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked[:top_k], 1):
        r["rank"] = i
    return {"objective": objective, "objective_label": _OBJECTIVES[objective],
            "mode": mode, "n_input": len(items), "n_valid": len(ranked),
            "invalid": invalid, "ranked": ranked[:top_k]}
