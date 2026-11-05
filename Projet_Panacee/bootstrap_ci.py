# -*- coding: utf-8 -*-
"""
Bootstrap IC95% + test de significativité paired — stacking GNN+RF.

Répond à la question : "rank_average > rf_only est-il RÉEL ou du bruit ?"

Méthode (rigoureuse) :
  - Paired bootstrap : à chaque rééchantillon, les MÊMES indices servent pour
    toutes les stratégies → la différence inter-stratégie est mesurée sur le
    même sous-ensemble (contrôle la corrélation entre modèles).
  - B=2000 rééchantillons avec remise du test set.
  - IC95% = percentiles [2.5, 97.5] de la distribution bootstrap.
  - Significativité : IC95% de la DIFFÉRENCE (rank_average - rf_only). Si l'IC
    exclut 0 → amélioration significative au seuil 5%. Si l'IC contient 0 →
    dans le bruit, non prouvé.
  - rank_average recalculé À CHAQUE rééchantillon (rangs dépendent du sous-ensemble).
  - DrugBank : ARV fixes (34), seul le fond test est rééchantillonné.

Prérequis : stacking_scores.npz produit par stack_gnn_rf.py.

Usage :
    python bootstrap_ci.py
    python bootstrap_ci.py --n_boot 5000
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import rankdata

from src.config import CHECKPOINT_DIR
from src.validation.pharma_metrics import full_screening_report

STRATEGIES = ["gnn_only", "rf_only", "rank_average", "lr_meta"]
PHARMA_KEYS = ["roc_auc", "ef_1pct", "ef_5pct", "bedroc_alpha20",
               "hit_rate_top_100", "recall_top_10pct", "prc_auc"]


def strat_test_scores(gnn, rf, meta, idx):
    """Scores par stratégie sur un sous-ensemble test (indices idx)."""
    g, r, m = gnn[idx], rf[idx], meta[idx]
    rank = (rankdata(g) + rankdata(r)) / 2.0  # recalculé sur le sous-ensemble
    return {"gnn_only": g, "rf_only": r, "rank_average": rank, "lr_meta": m}


def pharma_metrics(y, s):
    """EF/BEDROC/hit/AUC/PRC (sans novelty → rapide). None si pas d'actifs."""
    if y.sum() == 0:
        return None
    rep = full_screening_report(y, s)  # smiles=None → saute novelty
    return {k: rep.get(k) for k in PHARMA_KEYS}


def drugbank_recall(arv_scores, test_scores, top_pct=0.10):
    """Recall ARV = nb d'ARV (indices <n_arv) dans le top-X% du pool [ARV+test]."""
    n_arv = len(arv_scores)
    pool = np.concatenate([arv_scores, test_scores])
    n_top = max(1, int(len(pool) * top_pct))
    top_idx = np.argsort(-pool)[:n_top]
    return int(np.sum(top_idx < n_arv))


def strat_drugbank_pool(gnn_arv, rf_arv, meta_arv, gnn_te, rf_te, meta_te, idx):
    """Pour chaque stratégie : (arv_scores, test_scores) sur le fond rééchantillonné."""
    gt, rt, mt = gnn_te[idx], rf_te[idx], meta_te[idx]
    # rank : recalculé sur le pool [ARV + test rééchantillonné]
    g_pool = np.concatenate([gnn_arv, gt])
    r_pool = np.concatenate([rf_arv, rt])
    rank_pool = (rankdata(g_pool) + rankdata(r_pool)) / 2.0
    n_arv = len(gnn_arv)
    return {
        "gnn_only": (gnn_arv, gt),
        "rf_only": (rf_arv, rt),
        "rank_average": (rank_pool[:n_arv], rank_pool[n_arv:]),
        "lr_meta": (meta_arv, mt),
    }


def ci95(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.median(a)),
            float(np.percentile(a, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    npz_path = CHECKPOINT_DIR / "phase3_stacking" / "stacking_scores.npz"
    if not npz_path.exists():
        print(f"❌ {npz_path} absent — lance d'abord stack_gnn_rf.py")
        sys.exit(1)
    d = np.load(npz_path)
    y_te = d["y_te"].astype(float)
    gnn_te, rf_te, meta_te = d["gnn_te"], d["rf_te"], d["meta_te"]
    gnn_arv, rf_arv, meta_arv = d["gnn_arv"], d["rf_arv"], d["meta_arv"]
    n = len(y_te)
    print(f"Test aligné: {n} molécules, {int(y_te.sum())} actifs")
    print(f"ARV: {len(gnn_arv)} | Bootstrap B={args.n_boot}\n")

    # Accumulateurs : metric → strategy → liste bootstrap
    pharma_boot = {k: {s: [] for s in STRATEGIES} for k in PHARMA_KEYS}
    db_boot = {s: [] for s in STRATEGIES}
    # Différences paired vs rank_average
    diff_vs_rf = {k: [] for k in PHARMA_KEYS + ["drugbank"]}
    diff_vs_gnn = {k: [] for k in PHARMA_KEYS + ["drugbank"]}

    n_valid = 0
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=n)  # rééchantillon avec remise
        y_res = y_te[idx]
        if y_res.sum() == 0:
            continue
        scores = strat_test_scores(gnn_te, rf_te, meta_te, idx)
        pools = strat_drugbank_pool(gnn_arv, rf_arv, meta_arv,
                                    gnn_te, rf_te, meta_te, idx)

        m_by_strat = {}
        db_by_strat = {}
        ok = True
        for s in STRATEGIES:
            m = pharma_metrics(y_res, scores[s])
            if m is None or any(m[k] is None for k in PHARMA_KEYS):
                ok = False
                break
            m_by_strat[s] = m
            arv_sc, test_sc = pools[s]
            db_by_strat[s] = drugbank_recall(arv_sc, test_sc)
        if not ok:
            continue

        n_valid += 1
        for s in STRATEGIES:
            for k in PHARMA_KEYS:
                pharma_boot[k][s].append(m_by_strat[s][k])
            db_boot[s].append(db_by_strat[s])
        # Différences paired (même rééchantillon)
        for k in PHARMA_KEYS:
            diff_vs_rf[k].append(m_by_strat["rank_average"][k] - m_by_strat["rf_only"][k])
            diff_vs_gnn[k].append(m_by_strat["rank_average"][k] - m_by_strat["gnn_only"][k])
        diff_vs_rf["drugbank"].append(db_by_strat["rank_average"] - db_by_strat["rf_only"])
        diff_vs_gnn["drugbank"].append(db_by_strat["rank_average"] - db_by_strat["gnn_only"])

    print(f"Rééchantillons valides: {n_valid}/{args.n_boot}\n")

    # ── IC95% par stratégie ──
    print("═" * 78)
    print("IC95% PAR STRATÉGIE  [médiane (bas – haut)]")
    print("═" * 78)
    report = {"n_boot": n_valid, "per_strategy": {}, "differences": {}}
    label = {"roc_auc": "AUC", "ef_1pct": "EF@1%", "ef_5pct": "EF@5%",
             "bedroc_alpha20": "BEDROC", "hit_rate_top_100": "hit@100",
             "recall_top_10pct": "recall@10%", "prc_auc": "PRC-AUC"}
    for k in PHARMA_KEYS:
        print(f"\n{label[k]}:")
        report["per_strategy"][k] = {}
        for s in STRATEGIES:
            lo, med, hi = ci95(pharma_boot[k][s])
            report["per_strategy"][k][s] = {"median": med, "lo": lo, "hi": hi}
            print(f"  {s:<14} {med:7.4f}  ({lo:7.4f} – {hi:7.4f})")
    print("\nDrugBank recall /34:")
    report["per_strategy"]["drugbank"] = {}
    for s in STRATEGIES:
        lo, med, hi = ci95(db_boot[s])
        report["per_strategy"]["drugbank"][s] = {"median": med, "lo": lo, "hi": hi}
        print(f"  {s:<14} {med:7.1f}  ({lo:5.1f} – {hi:5.1f})")

    # ── Test de significativité paired ──
    print("\n" + "═" * 78)
    print("SIGNIFICATIVITÉ — différence paired rank_average − RÉFÉRENCE")
    print("IC95% exclut 0 → SIGNIFICATIF (✓).  IC95% contient 0 → bruit (✗)")
    print("═" * 78)

    def _report_diff(diffs, ref_name):
        print(f"\n▶ rank_average − {ref_name}:")
        out = {}
        for k in PHARMA_KEYS + ["drugbank"]:
            lo, med, hi = ci95(diffs[k])
            sig = "✓ SIGNIFICATIF" if (lo > 0 or hi < 0) else "✗ bruit"
            name = label.get(k, "DrugBank")
            print(f"  Δ{name:<12} {med:+8.4f}  ({lo:+8.4f} – {hi:+8.4f})  {sig}")
            out[k] = {"median": med, "lo": lo, "hi": hi,
                      "significant": bool(lo > 0 or hi < 0)}
        return out

    report["differences"]["vs_rf_only"] = _report_diff(diff_vs_rf, "rf_only")
    report["differences"]["vs_gnn_only"] = _report_diff(diff_vs_gnn, "gnn_only")

    out_path = CHECKPOINT_DIR / "phase3_stacking" / "bootstrap_ci_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {out_path}")


if __name__ == "__main__":
    main()
