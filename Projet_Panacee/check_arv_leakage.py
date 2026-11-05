# -*- coding: utf-8 -*-
"""
Vérifie la FUITE des 34 ARV DrugBank dans les splits HIV MoleculeNet.

Question de cohérence critique : diagnose_arv montre Zidovudine/Stavudine avec
sim=1.000 vs actifs training → molécule IDENTIQUE déjà dans le training. Si des
ARV du gate sont dans le train, une partie des "24 trouvés" = MÉMORISATION, pas
généralisation → le gate DrugBank est partiellement biaisé.

Pour chaque ARV (SMILES canonique), vérifie présence exacte dans train/val/test
+ son label (actif/inactif). Croise avec found/missed.

Sans training. Rapide.

Usage :
    python check_arv_leakage.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles
from stack_gnn_rf import load_hiv_data


def _canon(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def canon_set(smiles_series):
    out = {}
    for s in smiles_series:
        c = _canon(s)
        if c is not None:
            out[c] = s
    return out


def main():
    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    print(f"Splits: train={len(tr)} val={len(va)} test={len(te)} | label={lbl}")

    print("\n▶ Canonicalisation des splits (peut prendre ~1min)")
    tr_canon = canon_set(tr["smiles"])
    va_canon = canon_set(va["smiles"])
    te_canon = canon_set(te["smiles"])

    # Map canon → label pour chaque split
    def canon_label_map(df):
        m = {}
        for s, y in zip(df["smiles"], df[lbl]):
            c = _canon(s)
            if c is not None:
                m[c] = int(y)
        return m
    tr_lab = canon_label_map(tr)
    va_lab = canon_label_map(va)
    te_lab = canon_label_map(te)

    arvs = canonicalize_arv_smiles()
    print(f"\n▶ Vérification des {len(arvs)} ARV dans les splits")
    print("═" * 92)
    print(f"{'ARV':<22}{'classe':<9}{'train':>10}{'val':>8}{'test':>8}{'statut':>22}")
    print("─" * 92)

    counts = {"train_active": 0, "train_inactive": 0, "val": 0, "test": 0, "absent": 0}
    rows = []
    for d in arvs:
        c = d["smiles_canonical"]
        name = d["name"]
        cls = d.get("class", "?")
        in_tr = c in tr_lab
        in_va = c in va_lab
        in_te = c in te_lab

        tr_s = f"OUI(lbl={tr_lab[c]})" if in_tr else "-"
        va_s = f"OUI({va_lab[c]})" if in_va else "-"
        te_s = f"OUI({te_lab[c]})" if in_te else "-"

        if in_tr:
            statut = "⚠ FUITE TRAIN" + (" (actif)" if tr_lab[c] == 1 else " (inactif)")
            if tr_lab[c] == 1:
                counts["train_active"] += 1
            else:
                counts["train_inactive"] += 1
        elif in_va:
            statut = "⚠ dans val"
            counts["val"] += 1
        elif in_te:
            statut = "⚠ dans test"
            counts["test"] += 1
        else:
            statut = "✓ absent (propre)"
            counts["absent"] += 1

        rows.append({"name": name, "class": cls, "in_train": in_tr,
                     "train_label": tr_lab.get(c), "in_val": in_va, "in_test": in_te,
                     "statut": statut})
        print(f"{name:<22}{cls:<9}{tr_s:>10}{va_s:>8}{te_s:>8}{statut:>22}")

    print("═" * 92)
    print("SYNTHÈSE FUITE")
    print("═" * 92)
    print(f"  Dans TRAIN comme ACTIF   : {counts['train_active']}  ← mémorisation (gate biaisé ↑)")
    print(f"  Dans TRAIN comme INACTIF : {counts['train_inactive']} ← contradiction (ARV labellisé inactif !)")
    print(f"  Dans VAL                 : {counts['val']}")
    print(f"  Dans TEST                : {counts['test']}")
    print(f"  Absents (propre)         : {counts['absent']}")
    leaked = counts["train_active"] + counts["train_inactive"]
    print(f"\n  → {leaked}/{len(arvs)} ARV présents dans le TRAIN")
    print(f"  → Gate généralisation honnête = sur les {len(arvs)-leaked} ARV hors-train")

    out = {"counts": counts, "arvs": rows, "n_arv": len(arvs), "leaked_train": leaked}
    save_dir = CHECKPOINT_DIR / "phase3_stacking"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "arv_leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {save_dir / 'arv_leakage_report.json'}")


if __name__ == "__main__":
    main()
