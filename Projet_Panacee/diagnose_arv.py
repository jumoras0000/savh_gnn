# -*- coding: utf-8 -*-
"""
Diagnostic ARV manqués — OOD ou in-domain mal classés ?

Question : rank_avg_5 atteint 24/34 (manque 1 pour le gate 25). Les ARV encore
manqués sont-ils HORS domaine d'applicabilité (scaffolds jamais vus en training
→ plafond réel, fix = données) ou IN-DOMAIN mais mal classés (→ marge modèle) ?

Pour chaque ARV, calcule :
  - rang dans le pool [34 ARV + test] via l'ensemble rank_avg_5 (position / 3812)
  - found (dans top-10%) ou missed
  - Tanimoto max vs actifs du training (a-t-on vu une molécule active proche ?)
  - classe mécanistique + année d'approbation

Verdict par ARV manqué :
  - sim < 0.4 (seuil novelty projet) → OOD, scaffold inédit, non rattrapable par FP
  - sim >= 0.4 → in-domain, mal classé, marge d'amélioration légitime

Usage :
    python diagnose_arv.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.stats import rankdata

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR
from src.validation.drugbank_antiretroviral import canonicalize_arv_smiles

from stack_gnn_rf import gnn_scores_aligned, load_gnn, load_hiv_data
from push_ensemble import (
    FP_KINDS, featurize, rf_scores_aligned_fp, train_rf, rank_avg,
)

NOVELTY_THRESHOLD = 0.4  # seuil projet (cf. pharma_metrics novelty)


def tanimoto_max_to_actives(query_smiles, train_actives_smiles):
    """Pour chaque query, Tanimoto max vs les actifs du training. None si invalide."""
    from rdkit import Chem
    from rdkit.Chem import DataStructs

    from src.utils.fingerprints import morgan_fp_bitvect

    def _fp(smi):
        mol = Chem.MolFromSmiles(str(smi))
        return morgan_fp_bitvect(mol, 2, 2048) if mol is not None else None

    train_fps = [fp for fp in (_fp(s) for s in train_actives_smiles) if fp is not None]
    out = []
    for smi in query_smiles:
        fp = _fp(smi)
        if fp is None:
            out.append(None)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        out.append(float(max(sims)) if sims else None)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr, va, te, lbl = load_hiv_data(str(EXTERNAL_DIR / "phase3"))
    train_actives = tr[tr[lbl] == 1]["smiles"].tolist()
    tr_smi = tr["smiles"].tolist()
    ytr = tr[lbl].values
    print(f"Train actifs: {len(train_actives)} | test: {len(te)}")

    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]

    # ── 5 modèles de base (identique push_ensemble) ──
    gnn_ckpt = str(CHECKPOINT_DIR / "phase3_improved" / "best_val.pth")
    print(f"\n▶ GNN: {gnn_ckpt}")
    gnn = load_gnn(gnn_ckpt, device)
    sc = {"gnn": {
        "test": gnn_scores_aligned(gnn, te["smiles"].tolist(), device),
        "arv": gnn_scores_aligned(gnn, arv_smi, device),
    }}
    for kind in FP_KINDS:
        print(f"▶ RF + {kind.upper()}")
        Xtr, mtr = featurize(tr_smi, kind)
        rf = train_rf(Xtr[mtr], ytr[mtr])
        sc[f"rf_{kind}"] = {
            "test": rf_scores_aligned_fp(rf, te["smiles"].tolist(), kind),
            "arv": rf_scores_aligned_fp(rf, arv_smi, kind),
        }
    models = ["gnn"] + [f"rf_{k}" for k in FP_KINDS]

    # Intersection validité
    te_valid = np.ones(len(te), dtype=bool)
    arv_valid = np.ones(len(arv_smi), dtype=bool)
    for m in models:
        te_valid &= ~np.isnan(sc[m]["test"])
        arv_valid &= ~np.isnan(sc[m]["arv"])
    for m in models:
        sc[m]["test"] = sc[m]["test"][te_valid]
        sc[m]["arv"] = sc[m]["arv"][arv_valid]
    n_arv = int(arv_valid.sum())
    arv_kept = [arvs[i] for i in range(len(arvs)) if arv_valid[i]]
    print(f"\nARV valides: {n_arv}/{len(arvs)} | test valides: {int(te_valid.sum())}")

    # ── Ensemble rank_avg_5 sur pool [arv + test] ──
    pools = [np.concatenate([sc[m]["arv"], sc[m]["test"]]) for m in models]
    ens = np.sum([rankdata(p) for p in pools], axis=0) / len(models)
    n_pool = len(ens)
    n_top = max(1, int(n_pool * 0.10))

    # Position depuis le top (1 = meilleur score)
    order = np.argsort(-ens)
    pos = np.empty(n_pool, dtype=int)
    pos[order] = np.arange(1, n_pool + 1)
    arv_pos = pos[:n_arv]

    # ── Tanimoto max vs actifs training ──
    print("\n▶ Tanimoto max ARV vs actifs training")
    arv_sim = tanimoto_max_to_actives([d["smiles_canonical"] for d in arv_kept],
                                       train_actives)

    # ── Tableau par ARV, trié par rang ──
    rows = []
    for i, d in enumerate(arv_kept):
        rows.append({
            "name": d["name"],
            "class": d.get("class", "?"),
            "approved": d.get("approved", "?"),
            "position": int(arv_pos[i]),
            "percentile": round(100 * arv_pos[i] / n_pool, 2),
            "found": bool(arv_pos[i] <= n_top),
            "max_sim_train_active": round(arv_sim[i], 3) if arv_sim[i] is not None else None,
            "ood": (arv_sim[i] is not None and arv_sim[i] < NOVELTY_THRESHOLD),
        })
    rows.sort(key=lambda r: r["position"])

    print("\n" + "═" * 90)
    print(f"ARV classés par rang (top-10% = top {n_top}/{n_pool})")
    print("═" * 90)
    print(f"{'ARV':<22}{'classe':<10}{'an':>6}{'rang':>7}{'%ile':>7}{'simAct':>8}{'statut':>12}")
    print("─" * 90)
    for r in rows:
        statut = "✓ FOUND" if r["found"] else ("✗ OOD" if r["ood"] else "✗ in-dom")
        sim = f"{r['max_sim_train_active']:.3f}" if r["max_sim_train_active"] is not None else "N/A"
        print(f"{r['name']:<22}{r['class']:<10}{str(r['approved']):>6}{r['position']:>7}"
              f"{r['percentile']:>6.1f}%{sim:>8}{statut:>12}")

    # ── Synthèse des MANQUÉS ──
    missed = [r for r in rows if not r["found"]]
    missed_ood = [r for r in missed if r["ood"]]
    missed_indomain = [r for r in missed if not r["ood"]]

    print("\n" + "═" * 90)
    print("SYNTHÈSE DES ARV MANQUÉS")
    print("═" * 90)
    print(f"Trouvés: {n_arv - len(missed)}/{n_arv} (gate≥25)")
    print(f"Manqués total: {len(missed)}")
    print(f"  dont OOD (sim < {NOVELTY_THRESHOLD}, scaffold inédit): {len(missed_ood)}")
    print(f"  dont in-domain (sim ≥ {NOVELTY_THRESHOLD}, mal classés): {len(missed_indomain)}")

    if missed_ood:
        print(f"\n  OOD (non rattrapables par FP → fix = DONNÉES) :")
        for r in missed_ood:
            print(f"    {r['name']:<20} {r['class']:<8} sim={r['max_sim_train_active']} "
                  f"rang={r['position']} ({r['percentile']}%)")
    if missed_indomain:
        print(f"\n  IN-DOMAIN (marge modèle légitime) :")
        for r in missed_indomain:
            print(f"    {r['name']:<20} {r['class']:<8} sim={r['max_sim_train_active']} "
                  f"rang={r['position']} ({r['percentile']}%)")

    # Verdict
    print("\n" + "═" * 90)
    if len(missed_indomain) == 0:
        print("VERDICT : tous les manqués sont OOD → PLAFOND atteint avec ce training.")
        print("          Le fix légitime = ajouter des actifs de ces classes au training,")
        print("          PAS ajouter des fingerprints (ne rattrapera pas des scaffolds inédits).")
    else:
        print(f"VERDICT : {len(missed_indomain)} manqué(s) in-domain → marge d'amélioration")
        print("          légitime côté modèle (sans toucher au gate).")
    print("═" * 90)

    # Sauvegarde
    save_dir = CHECKPOINT_DIR / "phase3_stacking"
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n_arv": n_arv, "n_found": n_arv - len(missed),
        "n_top": n_top, "n_pool": n_pool,
        "novelty_threshold": NOVELTY_THRESHOLD,
        "arvs": rows,
        "missed_ood": [r["name"] for r in missed_ood],
        "missed_indomain": [r["name"] for r in missed_indomain],
    }
    out_path = save_dir / "diagnose_arv_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {out_path}")


if __name__ == "__main__":
    main()
