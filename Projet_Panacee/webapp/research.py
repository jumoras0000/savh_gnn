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


def _default_checkpoint() -> str:
    from src.config import CHECKPOINT_DIR, PHASE3
    return str(CHECKPOINT_DIR / "phase3" / PHASE3["checkpoint_name"])


def get_analyzer(checkpoint: str | None = None):
    """Renvoie un PanaceeAnalyzer (mis en cache). Lève si modèle indisponible."""
    ckpt = checkpoint or _default_checkpoint()
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint Phase 3 introuvable : {ckpt}. "
            f"Entraîne d'abord la Phase 3 (onglet Entraînement) ou importe un .pth."
        )
    if ckpt in _CACHE:
        return _CACHE[ckpt]
    from src.analysis.combinatorial_engine import PanaceeAnalyzer
    analyzer = PanaceeAnalyzer(checkpoint_path=ckpt, device="cpu")
    _CACHE[ckpt] = analyzer
    return analyzer


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
    analyzer = get_analyzer(checkpoint)
    results, invalid = [], []
    for smi in smiles_list:
        r = analyzer.predict_properties(smi)
        if r is None:
            invalid.append(smi)
            continue
        r["risk"] = assess_risk(r)
        results.append(r)
    return {"results": results, "invalid": invalid,
            "checkpoint": checkpoint or _default_checkpoint()}


def combo(smiles_list: list[str], checkpoint: str | None = None) -> dict:
    analyzer = get_analyzer(checkpoint)
    res = analyzer.analyze_combination(smiles_list)
    if res is None:
        return {"error": "combinaison invalide (≥ 2 molécules valides requises)"}
    for mol in res.get("molecules", []):
        mol["risk"] = assess_risk(mol)
    return res
