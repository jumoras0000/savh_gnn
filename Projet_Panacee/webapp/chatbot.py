# -*- coding: utf-8 -*-
"""
Chatbot Panacée — converser avec le modèle GNN entraîné, synchronisé avec Claude.

Deux modes, transparents pour l'appelant :
  • "claude"  : si le SDK Anthropic + ANTHROPIC_API_KEY sont présents, Claude
                (claude-opus-4-8, thinking adaptatif) orchestre une boucle d'outils
                qui appellent les fonctions du modèle GNN (toxicité, efficacité VIH,
                descripteurs, criblage, synergie). Analyses très avancées.
  • "local"   : sans clé/SDK, un assistant local de secours détecte l'intention +
                les SMILES et appelle DIRECTEMENT les mêmes outils GNN. On peut donc
                toujours « parler au modèle » même hors-ligne.

Les outils sont partagés entre les deux modes (_dispatch_tool).
"""
from __future__ import annotations

import json
import os

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "Tu es « Panacée Copilot », un assistant de découverte de médicaments adossé à "
    "un modèle GNN entraîné (toxicité Tox21, efficacité anti-VIH, ADME, synergie de "
    "combinaisons) et à la cheminformatique RDKit. Réponds en français, de façon "
    "claire et concise. Utilise les outils pour toute question sur des molécules "
    "(SMILES) : propriétés, toxicité, efficacité anti-VIH, descripteurs, criblage, "
    "combinaisons. Rappelle, quand c'est pertinent, que les prédictions sont "
    "in-silico et doivent être validées en laboratoire. Si aucun modèle n'est "
    "entraîné, appuie-toi sur les descripteurs RDKit (toujours disponibles) et "
    "invite à entraîner la Phase 2/3."
)

# ──────────────────────────────────────────────────────────────────────
# Définition des outils (schémas JSON partagés Claude + fallback)
# ──────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "compute_descriptors",
        "description": "Descripteurs physico-chimiques RDKit (MW, LogP, TPSA, HBD/HBA, "
                       "QED, Lipinski) d'une ou plusieurs molécules. Ne nécessite aucun modèle.",
        "input_schema": {
            "type": "object",
            "properties": {"smiles": {"type": "array", "items": {"type": "string"},
                                       "description": "Liste de SMILES"}},
            "required": ["smiles"],
        },
    },
    {
        "name": "predict_molecule",
        "description": "Prédit toxicité (12 endpoints Tox21), efficacité anti-VIH, ADME, "
                       "drug-likeness et le RISQUE d'une ou plusieurs molécules via le modèle "
                       "GNN (Phase 3 complet, ou Phase 2 toxicité seule, sinon descripteurs).",
        "input_schema": {
            "type": "object",
            "properties": {"smiles": {"type": "array", "items": {"type": "string"}}},
            "required": ["smiles"],
        },
    },
    {
        "name": "screen_library",
        "description": "Criblage virtuel : classe des molécules par objectif. "
                       "objective='efficacy' (anti-VIH, Phase 3), 'safety' (Phase 2-3) ou "
                       "'drug_likeness' (QED, sans modèle). Fournir smiles OU library "
                       "('hiv_reference'/'reference_drugs').",
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {"type": "string", "enum": ["efficacy", "safety", "drug_likeness"]},
                "smiles": {"type": "array", "items": {"type": "string"}},
                "library": {"type": "string", "enum": ["hiv_reference", "reference_drugs"]},
            },
            "required": ["objective"],
        },
    },
    {
        "name": "analyze_combination",
        "description": "Analyse une combinaison de molécules (≥2) : synergie par paire, "
                       "doses optimales, score de réussite. Nécessite un modèle Phase 3.",
        "input_schema": {
            "type": "object",
            "properties": {"smiles": {"type": "array", "items": {"type": "string"}}},
            "required": ["smiles"],
        },
    },
    {
        "name": "training_status",
        "description": "État des entraînements : runs détectés, dernière epoch, AUC, FNR, "
                       "verdict de sécurité, et statut du process en cours.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ──────────────────────────────────────────────────────────────────────
# Exécution des outils (appellent les modules existants)
# ──────────────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, args: dict) -> dict:
    try:
        if name == "compute_descriptors":
            from webapp import cheminfo
            return {"results": [cheminfo.descriptors(s) for s in args.get("smiles", [])]}
        if name == "predict_molecule":
            from webapp import research
            return research.predict(args.get("smiles", []))
        if name == "screen_library":
            from webapp import cheminfo, research
            mols = args.get("smiles")
            if not mols and args.get("library"):
                mols = cheminfo.library(args["library"])
            if not mols:
                return {"error": "fournir 'smiles' ou 'library'"}
            return research.screen(mols, objective=args.get("objective", "drug_likeness"))
        if name == "analyze_combination":
            from webapp import research
            return research.combo(args.get("smiles", []))
        if name == "training_status":
            from webapp import service
            from webapp.trainer import MANAGER
            root = os.environ.get("PANACEE_CKPT_ROOT", "checkpoints")
            return {"runs": service.list_runs(root), "process": MANAGER.status()}
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:  # pragma: no cover
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": f"outil inconnu: {name}"}


# ──────────────────────────────────────────────────────────────────────
# Mode Claude (Anthropic SDK + tool-use)
# ──────────────────────────────────────────────────────────────────────

def _claude_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def _chat_claude(history: list[dict], max_steps: int = 6) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": m["role"], "content": m["content"]} for m in history
                if m.get("role") in ("user", "assistant")]
    tool_trace = []

    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL, max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = _dispatch_tool(block.name, dict(block.input))
                    tool_trace.append({"tool": block.name, "input": dict(block.input)})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out, default=str, ensure_ascii=False)})
            messages.append({"role": "user", "content": results})
            continue
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"reply": text, "mode": "claude", "tools": tool_trace}

    return {"reply": "Analyse interrompue (trop d'étapes d'outils).",
            "mode": "claude", "tools": tool_trace}


# ──────────────────────────────────────────────────────────────────────
# Mode local de secours (sans clé Claude) — détecte intention + SMILES
# ──────────────────────────────────────────────────────────────────────

def _looks_like_smiles(token: str) -> bool:
    """Heuristique : RDKit parse, en écartant les mots tout en minuscules sans
    caractère spécial (évite 'code', 'data'… ; garde 'CCO', 'c1ccccc1', etc.)."""
    if len(token) < 3:
        return False
    has_special = any(c in token for c in "()=#[]@+/\\0123456789")
    if token.isalpha() and token.islower() and not has_special:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(token) is not None
    except Exception:
        return False


def _extract_smiles(text: str) -> list[str]:
    out = []
    for raw in text.replace(",", " ").replace(";", " ").split():
        tok = raw.strip().strip(".")
        if _looks_like_smiles(tok):
            out.append(tok)
    return out


def _fmt_descriptors(d: dict) -> str:
    if not d.get("valid"):
        return f"  • {d.get('smiles')} : SMILES invalide."
    return (f"  • {d['canonical']} — {d['formula']}, MW {d['mw']} g/mol, LogP {d['logp']}, "
            f"TPSA {d['tpsa']} Å², QED {d['qed']}, Lipinski {'OK' if d['lipinski_pass'] else 'NON'}")


def _chat_local(history: list[dict]) -> dict:
    user = ""
    for m in reversed(history):
        if m.get("role") == "user":
            user = m.get("content", "")
            break
    low = user.lower()
    smiles = _extract_smiles(user)
    tools = []

    def done(reply):
        return {"reply": reply, "mode": "local", "tools": tools}

    # Intentions
    if any(k in low for k in ("statut", "status", "entrainement", "entraînement", "run")) and not smiles:
        res = _dispatch_tool("training_status", {})
        tools.append({"tool": "training_status", "input": {}})
        runs = res.get("runs", [])
        if not runs:
            return done("Aucun run détecté. Lance un entraînement (onglet Entraînement).")
        lines = [f"  • {r['id']} · {r.get('status')} · AUC {r.get('val_auc')} · "
                 f"FNR {r.get('macro_fnr')} · danger {r.get('n_danger')}" for r in runs[:6]]
        return done("Runs détectés :\n" + "\n".join(lines))

    if any(k in low for k in ("cribl", "screen", "vih", "hiv", "candidat")):
        objective = ("efficacy" if any(k in low for k in ("vih", "hiv", "effica")) else
                     "safety" if any(k in low for k in ("sûr", "sur", "sécur", "secur", "toxic")) else
                     "drug_likeness")
        library = None
        if not smiles:
            library = "hiv_reference" if objective == "efficacy" else "reference_drugs"
        args = {"objective": objective, **({"smiles": smiles} if smiles else {"library": library})}
        res = _dispatch_tool("screen_library", args)
        tools.append({"tool": "screen_library", "input": args})
        if res.get("error"):
            return done("Criblage : " + res["error"])
        top = res.get("ranked", [])[:5]
        lines = [f"  {r['rank']}. {r.get('name') or r['smiles'][:32]} — score {r['score']} "
                 f"(QED {r.get('qed')})" for r in top]
        return done(f"Criblage « {res['objective_label']} » (mode {res['mode']}) — top {len(top)} :\n"
                    + "\n".join(lines))

    if smiles and len(smiles) >= 2 and any(k in low for k in ("synerg", "combin", "associ")):
        res = _dispatch_tool("analyze_combination", {"smiles": smiles})
        tools.append({"tool": "analyze_combination", "input": {"smiles": smiles}})
        if res.get("error"):
            return done("Combinaison : " + res["error"] +
                        " (la synergie nécessite un modèle Phase 3).")
        return done(f"Combinaison — score de réussite {res['success_score']}%, "
                    f"confiance {res['confidence']}%, sécurité {res['combined_safety']}%.")

    if smiles and any(k in low for k in ("toxic", "effica", "vih", "hiv", "sûr", "sur",
                                          "sécur", "secur", "risque", "médicament", "predict", "prédi")):
        res = _dispatch_tool("predict_molecule", {"smiles": smiles})
        tools.append({"tool": "predict_molecule", "input": {"smiles": smiles}})
        if res.get("error"):
            res = _dispatch_tool("compute_descriptors", {"smiles": smiles})
            tools.append({"tool": "compute_descriptors", "input": {"smiles": smiles}})
            return done("Pas de modèle entraîné → descripteurs RDKit :\n"
                        + "\n".join(_fmt_descriptors(d) for d in res.get("results", [])))
        lines = []
        for r in res.get("results", []):
            risk = (r.get("risk") or {}).get("level", "?")
            safety = r.get("safety_score", "—")
            eff = (r.get("efficacy") or {}).get("probabilite_activite", "—")
            lines.append(f"  • {r.get('smiles')} — risque {risk}, sécurité {safety}%, "
                         f"efficacité {eff}%")
        note = res.get("note", "")
        return done(f"Analyse ({res.get('mode')}) :\n" + "\n".join(lines) +
                    (f"\n({note})" if note else ""))

    if smiles:  # SMILES sans intention claire → descripteurs
        res = _dispatch_tool("compute_descriptors", {"smiles": smiles})
        tools.append({"tool": "compute_descriptors", "input": {"smiles": smiles}})
        return done("Descripteurs :\n" + "\n".join(_fmt_descriptors(d) for d in res.get("results", [])))

    # Aucun SMILES, aucune intention reconnue → aide
    return done(
        "Je suis le copilote Panacée (mode local — définis ANTHROPIC_API_KEY pour la "
        "synchronisation Claude et des analyses avancées). Donne-moi un ou plusieurs "
        "SMILES et une intention. Exemples :\n"
        "  • « toxicité et risque de CC(=O)Nc1ccc(O)cc1 »\n"
        "  • « crible la bibliothèque anti-VIH par efficacité »\n"
        "  • « synergie de CCO et CC(=O)O »\n"
        "  • « statut de l'entraînement »")


# ──────────────────────────────────────────────────────────────────────
# Entrée publique
# ──────────────────────────────────────────────────────────────────────

def chat(history: list[dict]) -> dict:
    """history = [{role, content}, ...]. Renvoie {reply, mode, tools}."""
    if not history:
        return {"reply": "Pose-moi une question sur une molécule.", "mode": "local", "tools": []}
    if _claude_available():
        try:
            return _chat_claude(history)
        except Exception as e:  # pragma: no cover - dépend de l'API
            return {"reply": f"Claude indisponible ({e}). Bascule en mode local. "
                             "Réessaie ta question.", "mode": "local", "tools": []}
    return _chat_local(history)
