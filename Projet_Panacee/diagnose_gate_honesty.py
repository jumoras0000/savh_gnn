# -*- coding: utf-8 -*-
"""
Honnêteté du gate A' (31/34) — transfert légitime ou assisté par analogues ?

A' passe le gate (31/34 vs 24 pour A, +7). Question critique : les ARV
nouvellement trouvés le sont-ils par transfert de pharmacophore (légitime) ou
parce qu'on a entraîné sur des quasi-copies (analogues très proches) ?

Pour chaque ARV : Tanimoto MAX vs les actifs ChEMBL AJOUTÉS au train + found/missed
sous A et A'. Croisement → identifie les 7 nouveaux et juge leur proximité.

Échelle (Morgan r=2) :
  sim < 0.70  → transfert légitime (pharmacophore de classe, pas copie)
  0.70–0.85   → modéré (chemotype proche)
  > 0.85      → ANALOGUE PROCHE → gate assisté, à divulguer/resserrer

Reconstruit le set ajouté avec EXACTEMENT le même filtrage anti-fuite.

Usage :
    python diagnose_gate_honesty.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles
from src.preprocessing.scaffold_split import generate_scaffold
from stack_gnn_rf import load_hiv_data
from model_a import load_model_a, score_model_a


def _canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None


def canon_series(series):
    out = set()
    for s in series:
        c = _canon(s)
        if c:
            out.add(c)
    return out


def rebuild_added(chembl_csv, arv_canon, va_canon, te_canon, tr_canon):
    """Reconstruit EXACTEMENT le set ajouté (même filtrage que train_augmented_rf)."""
    arv_scaffolds = {generate_scaffold(c) for c in arv_canon}
    eval_scaffolds = {generate_scaffold(s) for s in list(va_canon) + list(te_canon)}
    chembl = pd.read_csv(chembl_csv)
    added, seen = [], set()
    for s in chembl["smiles"]:
        c = _canon(s)
        if c is None or c in seen:
            continue
        if c in arv_canon:
            continue
        scaf = generate_scaffold(c)
        if scaf in eval_scaffolds:
            continue
        if c in tr_canon:
            continue
        seen.add(c)
        added.append(c)
    return added


def found_arvs(bundle, arv_smi, te_smi, device):
    """Retourne le set d'indices ARV dans le top-10% du pool [arv+test]."""
    both = arv_smi + te_smi
    scores, valid = score_model_a(bundle, both, device)
    n_arv = len(arv_smi)
    pool = scores[valid]
    # position des ARV valides
    compact = np.cumsum(valid) - 1
    n_top = max(1, int(len(pool) * 0.10))
    order = np.argsort(-pool)
    top_rank = np.empty(len(pool), dtype=int)
    top_rank[order] = np.arange(1, len(pool) + 1)
    found = set()
    for i in range(n_arv):
        if valid[i] and top_rank[compact[i]] <= n_top:
            found.add(i)
    return found


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    te_smi = te["smiles"].tolist()

    # ── Reconstruire le set ajouté ──
    print("\n▶ Reconstruction du set ChEMBL ajouté (même filtrage)")
    arv_canon = set(arv_smi)
    va_canon = canon_series(va["smiles"])
    te_canon = canon_series(te["smiles"])
    tr_canon = canon_series(tr["smiles"])
    added = rebuild_added(str(EXTERNAL_DIR / "chembl_hiv" / "chembl_hiv_merged.csv"),
                          arv_canon, va_canon, te_canon, tr_canon)
    print(f"  Actifs ajoutés reconstruits: {len(added)}")

    # ── Fingerprints des ajoutés ──
    print("▶ Fingerprints Morgan des ajoutés")
    from rdkit import Chem
    from rdkit.Chem import DataStructs
    from src.utils.fingerprints import morgan_fp_bitvect

    added_fps = []
    for c in added:
        m = Chem.MolFromSmiles(c)
        if m is not None:
            added_fps.append(morgan_fp_bitvect(m, 2, 2048))
    print(f"  {len(added_fps)} FP ajoutés")

    # ── found/missed sous A et A' ──
    print("\n▶ Found/missed sous A et A'")
    bundle_A = load_model_a(CHECKPOINT_DIR / "model_A_rank_avg_5", device)
    bundle_Ap = load_model_a(CHECKPOINT_DIR / "model_A_prime_augmented", device)
    found_A = found_arvs(bundle_A, arv_smi, te_smi, device)
    found_Ap = found_arvs(bundle_Ap, arv_smi, te_smi, device)
    print(f"  A: {len(found_A)}/34 | A': {len(found_Ap)}/34")
    newly = found_Ap - found_A
    lost = found_A - found_Ap
    print(f"  Nouveaux trouvés par A': {len(newly)} | perdus: {len(lost)}")

    # ── Tanimoto max ARV vs ajoutés ──
    print("\n▶ Tanimoto max ARV vs actifs ajoutés")
    rows = []
    for i, d in enumerate(arvs):
        m = Chem.MolFromSmiles(arv_smi[i])
        if m is None or not added_fps:
            max_sim = None
        else:
            fp = morgan_fp_bitvect(m, 2, 2048)
            sims = DataStructs.BulkTanimotoSimilarity(fp, added_fps)
            max_sim = float(max(sims)) if sims else None
        rows.append({
            "name": d["name"], "class": d.get("class", "?"),
            "found_A": i in found_A, "found_Ap": i in found_Ap,
            "newly_found": i in newly, "max_sim_added": max_sim,
        })

    # ── Tableau : focus sur les nouveaux ──
    def verdict(sim):
        if sim is None:
            return "?"
        if sim > 0.85:
            return "⚠ ANALOGUE PROCHE"
        if sim >= 0.70:
            return "~ modéré"
        return "✓ transfert"

    print("\n" + "═" * 84)
    print("ARV NOUVELLEMENT TROUVÉS PAR A' (les 7 qui font passer le gate)")
    print("═" * 84)
    print(f"{'ARV':<22}{'classe':<9}{'sim_max_ajouté':>16}{'verdict':>22}")
    print("─" * 84)
    newly_rows = [r for r in rows if r["newly_found"]]
    newly_rows.sort(key=lambda r: -(r["max_sim_added"] or 0))
    for r in newly_rows:
        s = f"{r['max_sim_added']:.3f}" if r["max_sim_added"] is not None else "N/A"
        print(f"{r['name']:<22}{r['class']:<9}{s:>16}{verdict(r['max_sim_added']):>22}")

    # ── Synthèse honnêteté ──
    n_analog = sum(1 for r in newly_rows if r["max_sim_added"] and r["max_sim_added"] > 0.85)
    n_moderate = sum(1 for r in newly_rows if r["max_sim_added"] and 0.70 <= r["max_sim_added"] <= 0.85)
    n_legit = sum(1 for r in newly_rows if r["max_sim_added"] is not None and r["max_sim_added"] < 0.70)

    print("\n" + "═" * 84)
    print("SYNTHÈSE HONNÊTETÉ DU GATE")
    print("═" * 84)
    print(f"  Gain brut A'−A : +{len(newly)} ARV")
    print(f"  Parmi les nouveaux :")
    print(f"    transfert légitime (sim < 0.70)   : {n_legit}")
    print(f"    modéré (0.70–0.85)                : {n_moderate}")
    print(f"    ⚠ analogue proche (sim > 0.85)    : {n_analog}")
    gate_honest = len(found_Ap) - n_analog
    print(f"\n  Gate brut A'            : {len(found_Ap)}/34")
    print(f"  Gate hors analogues proches : {gate_honest}/34 "
          f"({'PASS' if gate_honest >= 25 else 'FAIL'} au seuil 25)")
    if n_analog > 0:
        print(f"\n  ⚠ {n_analog} ARV trouvés via analogue proche → à divulguer dans le rapport.")
    else:
        print(f"\n  ✓ Aucun ARV nouveau via analogue proche → gain = transfert légitime.")

    # ── Sauvegarde ──
    out = {
        "n_added": len(added), "found_A": len(found_A), "found_Ap": len(found_Ap),
        "newly_found": [r["name"] for r in newly_rows],
        "lost": [rows[i]["name"] for i in range(len(arvs)) if i in lost],
        "n_analog_assisted": n_analog, "n_moderate": n_moderate, "n_legit": n_legit,
        "gate_honest_excl_analog": gate_honest,
        "arvs": rows,
    }
    save_dir = CHECKPOINT_DIR / "phase3_stacking"
    with open(save_dir / "gate_honesty_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {save_dir / 'gate_honesty_report.json'}")


if __name__ == "__main__":
    main()
