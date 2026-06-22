# -*- coding: utf-8 -*-
"""
Cheminformatique (RDKit) — analyses qui ne dépendent PAS du modèle entraîné.
Toujours disponibles : descripteurs physico-chimiques, règle de Lipinski, QED,
canonicalisation, et rendu 2D de la structure (SVG).

RDKit est importé paresseusement pour ne pas alourdir le démarrage du serveur.
"""
from __future__ import annotations

from functools import lru_cache

from webapp.catalog import LIBRARIES


def descriptors(smiles: str) -> dict:
    """Descripteurs physico-chimiques + Lipinski + QED. {valid:False} si SMILES KO."""
    from rdkit import Chem
    from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot = Lipinski.NumRotatableBonds(mol)

    # Règle de Lipinski (Ro5) : 0 ou 1 violation tolérée
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])

    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = None

    return {
        "valid": True,
        "smiles": smiles,
        "canonical": Chem.MolToSmiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(mw, 1),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 1),
        "hbd": int(hbd),
        "hba": int(hba),
        "rotatable_bonds": int(rot),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "qed": round(qed, 3) if qed is not None else None,
        "lipinski_violations": int(violations),
        "lipinski_pass": bool(violations <= 1),
    }


# ──────────────────────────────────────────────────────────────────────
# Alertes structurelles par endpoint clinique (toxicophores) — SANS modèle.
# Complète le GNN avec des règles de chimie médicinale établies. Une alerte
# n'est PAS une preuve de toxicité : c'est un motif à surveiller / contre-vérifier.
# ──────────────────────────────────────────────────────────────────────
TOXICOPHORES = {
    "ames_mutagenicity": {
        "label": "Mutagénicité (test d'Ames)",
        "why": "Motifs réactifs vis-à-vis de l'ADN, associés à un risque mutagène.",
        "patterns": [
            ("nitro_aromatique", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][a]",
             "Nitro aromatique — toxicophore d'Ames classique."),
            ("amine_aromatique", "[NX3;H2,H1;!$(NC=O);!$(N=O)][c]",
             "Amine aromatique libre — activable en nitrénium mutagène."),
            ("azo", "[#6][NX2]=[NX2][#6]", "Liaison azo — clivable en amines aromatiques."),
            ("nitroso", "[#6,#7][NX2]=O", "Groupe nitroso — alkylant potentiel."),
            ("azide", "[$([NX1]=[NX2+]=[NX1-]),$([NX1-]=[NX2+]=[NX1-])]", "Azoture réactif."),
            ("epoxide", "[OX2r3]1[#6r3][#6r3]1", "Époxyde — alkylant de l'ADN."),
            ("aldehyde", "[CX3H1](=O)[#6]", "Aldéhyde — réactif électrophile."),
            ("halogenure_alkyle", "[CX4][Cl,Br,I]", "Halogénure d'alkyle — agent alkylant."),
            ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]", "Accepteur de Michael — électrophile."),
        ],
    },
    "dili_hepatotox": {
        "label": "Hépatotoxicité (DILI)",
        "why": "Motifs liés à des métabolites réactifs hépatiques (idiosyncrasie).",
        "patterns": [
            ("aniline_anilide", "[NX3;H2,H1]c1ccccc1",
             "Aniline / anilide — bioactivation hépatique possible (ex. paracétamol)."),
            ("hydrazine", "[NX3][NX3]", "Hydrazine — hépatotoxique connu."),
            ("thiophene", "c1ccsc1", "Thiophène — époxydation réactive possible."),
            ("nitro_aromatique", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][a]",
             "Nitro aromatique — stress oxydatif hépatique."),
        ],
    },
}

# hERG (cardiotoxicité) : mal capturé par un seul SMARTS → heuristique
# pharmacophore = azote basique (protonable) + lipophilie + masse suffisante.
_HERG_BASIC_AMINE = "[NX3;!$(NC=O);!$(N=*);!$([N+]);!$(Nc);!$(NS=O)]"


@lru_cache(maxsize=256)
def _compile(smarts: str):
    # Mise en cache : les motifs SMARTS sont constants. Sans cache, structural_alerts
    # recompilait ~25 motifs PAR molécule (coûteux lors d'un criblage de bibliothèque).
    from rdkit import Chem
    return Chem.MolFromSmarts(smarts)


def structural_alerts(smiles: str) -> dict:
    """
    Alertes structurelles par endpoint clinique (Ames, DILI, hERG).

    Renvoie {valid, level, endpoints:[{key,label,why,hits:[{name,description}]}],
             herg}. `level` ∈ OK/WARN/DANGER agrège la gravité.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles}

    endpoints = []
    total_hits = 0
    for key, spec in TOXICOPHORES.items():
        hits = []
        for name, smarts, desc in spec["patterns"]:
            patt = _compile(smarts)
            if patt is not None and mol.HasSubstructMatch(patt):
                hits.append({"name": name, "description": desc})
        if hits:
            total_hits += len(hits)
            endpoints.append({"key": key, "label": spec["label"],
                              "why": spec["why"], "hits": hits})

    # hERG : amine basique + logP élevé + MW suffisante
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    amine = _compile(_HERG_BASIC_AMINE)
    has_basic_amine = bool(amine is not None and mol.HasSubstructMatch(amine))
    herg_risk = bool(has_basic_amine and logp > 3.0 and mw > 250)
    herg = {
        "risk": herg_risk,
        "basic_amine": has_basic_amine,
        "logp": round(float(logp), 2),
        "reason": ("Azote basique + lipophilie élevée (LogP>3) + MW>250 : "
                   "pharmacophore hERG typique." if herg_risk
                   else "Pas de pharmacophore hERG net."),
    }
    if herg_risk:
        endpoints.append({
            "key": "herg_cardiotox", "label": "Cardiotoxicité (hERG)",
            "why": "Blocage du canal potassique hERG → risque d'arythmie.",
            "hits": [{"name": "pharmacophore_herg", "description": herg["reason"]}],
        })

    # Agrégation du niveau
    n_ep = len(endpoints)
    if n_ep >= 2 or total_hits >= 3 or herg_risk:
        level = "DANGER"
    elif n_ep >= 1:
        level = "WARN"
    else:
        level = "OK"

    return {"valid": True, "smiles": smiles, "level": level,
            "endpoints": endpoints, "herg": herg}


# ──────────────────────────────────────────────────────────────────────
# Domaine d'applicabilité : la molécule est-elle proche de ce que le modèle
# « connaît » ? Une prédiction loin du domaine = extrapolation peu fiable.
# Référence = union des molécules réelles du catalogue (médicaments connus).
# ──────────────────────────────────────────────────────────────────────
_AD_CACHE: dict = {}
_AD_THRESHOLD = 0.30  # Tanimoto (Morgan r2) : en-deçà → hors domaine


def _morgan_fp(mol):
    """Empreinte Morgan (rayon 2, 2048 bits) via l'API moderne, repli sinon."""
    try:
        from rdkit.Chem import rdFingerprintGenerator
        gen = _AD_CACHE.get("gen")
        if gen is None:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            _AD_CACHE["gen"] = gen
        return gen.GetFingerprint(mol)
    except Exception:
        from rdkit.Chem import rdMolDescriptors
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _reference_fingerprints():
    """Empreintes Morgan des molécules de référence (mises en cache)."""
    if "fps" in _AD_CACHE:
        return _AD_CACHE["fps"]
    from rdkit import Chem
    fps = []
    for lib in LIBRARIES.values():
        for entry in lib["molecules"]:
            m = Chem.MolFromSmiles(entry["smiles"])
            if m is not None:
                fps.append((entry["name"], _morgan_fp(m)))
    _AD_CACHE["fps"] = fps
    return fps


def applicability_domain(smiles: str) -> dict:
    """
    Estime si une molécule est dans le domaine d'applicabilité du modèle via la
    similarité de Tanimoto (empreinte Morgan) au jeu de référence.

    Renvoie {valid, max_similarity, nearest, in_domain, level, note}.
    """
    from rdkit import Chem
    from rdkit.Chem import DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles}

    fp = _morgan_fp(mol)
    best_sim, best_name = 0.0, None
    for name, ref in _reference_fingerprints():
        s = DataStructs.TanimotoSimilarity(fp, ref)
        if s > best_sim:
            best_sim, best_name = s, name

    in_domain = best_sim >= _AD_THRESHOLD
    level = "OK" if best_sim >= 0.5 else ("WARN" if in_domain else "DANGER")
    if best_sim >= 0.5:
        note = "Proche de molécules connues : prédiction dans le domaine."
    elif in_domain:
        note = "Similarité modérée : prédiction à interpréter avec prudence."
    else:
        note = ("Très éloignée des molécules connues : la prédiction est une "
                "EXTRAPOLATION, fiabilité réduite.")
    return {"valid": True, "smiles": smiles,
            "max_similarity": round(float(best_sim), 3),
            "nearest": best_name, "in_domain": bool(in_domain),
            "level": level, "note": note}


def depict_svg(smiles: str, width: int = 260, height: int = 200) -> str | None:
    """Rendu 2D de la molécule en SVG (None si SMILES invalide)."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    width = max(80, min(600, int(width)))
    height = max(80, min(600, int(height)))
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def library(name: str) -> list[dict]:
    """SMILES d'une bibliothèque de référence, canonicalisés (invalides ignorés)."""
    lib = LIBRARIES.get(name)
    if not lib:
        return []
    from rdkit import Chem
    out = []
    for entry in lib["molecules"]:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is not None:
            out.append({"name": entry["name"], "smiles": Chem.MolToSmiles(mol)})
    return out


def list_libraries() -> dict:
    """Métadonnées + comptes valides de chaque bibliothèque."""
    out = {}
    for key, lib in LIBRARIES.items():
        valid = library(key)
        out[key] = {"label": lib["label"], "note": lib["note"],
                    "count": len(valid), "molecules": valid}
    return out
