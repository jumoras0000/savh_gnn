# -*- coding: utf-8 -*-
"""
Cheminformatique (RDKit) — analyses qui ne dépendent PAS du modèle entraîné.
Toujours disponibles : descripteurs physico-chimiques, règle de Lipinski, QED,
canonicalisation, et rendu 2D de la structure (SVG).

RDKit est importé paresseusement pour ne pas alourdir le démarrage du serveur.
"""
from __future__ import annotations

from webapp.catalog import LIBRARIES


def descriptors(smiles: str) -> dict:
    """Descripteurs physico-chimiques + Lipinski + QED. {valid:False} si SMILES KO."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

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
