# -*- coding: utf-8 -*-
"""
Stacking GNN + RF(ECFP4) — combine les forces mesurées des deux modèles.

Motivation MESURÉE (rapports bruts confirmés) :
  GNN-only : AUC 0.7194 | EF@1% 15.39 | BEDROC 0.2845 | hit@100 0.16 | DrugBank 17/34
  RF+ECFP4 : AUC 0.6524 | EF@1% 17.75 | BEDROC 0.3282 | hit@100 0.23 | DrugBank 20/34
  Hybrid   : AUC 0.6749 | EF@1% 11.84 | BEDROC 0.2809 | hit@100 0.19 | DrugBank 20/34 (ÉCHEC)

Le GNN gagne l'AUC globale ; RF gagne le ranking top-K (métriques pharma).
La fusion interne (hybrid concat) a échoué. On tente la fusion EXTERNE :
stacking des prédictions. Chaque modèle garde sa force, un meta-learner apprend
à les combiner SANS les affaiblir.

Rigueur :
  - Alignement par INDEX (NaN si SMILES invalide), intersection des valides →
    aucun désalignement GNN/RF possible.
  - Meta-learner entraîné UNIQUEMENT sur val, évalué sur test → zéro fuite.
  - RF hyperparams identiques à benchmark_baselines.py → RF-alone doit
    reproduire AUC 0.6524 (sanity check interne).
  - 4 stratégies de combinaison comparées sur le MÊME test aligné :
      1. GNN seul       (référence)
      2. RF seul        (référence)
      3. Rank-average   (non-paramétrique, scale-free)
      4. LR meta-learner (paramétrique, appris sur val)

Usage :
    python stack_gnn_rf.py
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
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR, PHASE3
from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyPredictor
from src.preprocessing.graph_builder import smiles_to_graph
from src.utils.fingerprints import smiles_to_morgan_array
from src.utils.reproducibility import set_seed
from src.utils.safe_load import safe_load_checkpoint
from src.validation.drugbank_antiretroviral import (
    canonicalize_arv_smiles,
    validate_model_recall,
)
from src.validation.pharma_metrics import full_screening_report


# ═══════════════════════════════════════════════════════════════════════════
# Prédicteurs ALIGNÉS par index (NaN si molécule invalide)
# ═══════════════════════════════════════════════════════════════════════════

def gnn_scores_aligned(model, smiles_list, device, batch_size=128):
    """Scores GNN (sigmoid efficacy) alignés sur l'index d'entrée. NaN si invalide."""
    from torch_geometric.data import Batch

    scores = np.full(len(smiles_list), np.nan, dtype=np.float64)
    idx_buf, graph_buf = [], []

    def _flush():
        if not graph_buf:
            return
        batch = Batch.from_data_list(graph_buf).to(device)
        with torch.no_grad():
            logits = model(batch)["efficacy"].cpu().numpy().ravel()
        proba = 1.0 / (1.0 + np.exp(-logits))
        for k, original_idx in enumerate(idx_buf):
            scores[original_idx] = proba[k]
        idx_buf.clear()
        graph_buf.clear()

    for i, s in enumerate(smiles_list):
        try:
            g = smiles_to_graph(s)
        except Exception:
            g = None
        if g is not None:
            graph_buf.append(g)
            idx_buf.append(i)
        if len(graph_buf) >= batch_size:
            _flush()
    _flush()
    return scores


def rf_scores_aligned(rf, smiles_list, radius=2, n_bits=2048):
    """Scores RF (proba classe 1) alignés sur l'index d'entrée. NaN si invalide."""
    scores = np.full(len(smiles_list), np.nan, dtype=np.float64)
    X, mask = smiles_to_morgan_array(smiles_list, radius=radius, n_bits=n_bits)
    if mask.sum() > 0:
        proba = rf.predict_proba(X[mask])[:, 1]
        scores[mask] = proba
    return scores


def load_gnn(ckpt_path, device):
    """Charge le GNN-only (sans morgan_dim)."""
    ckpt = safe_load_checkpoint(ckpt_path, trusted=True)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    use_chirality = cfg.get("use_chirality", PHASE3.get("use_chirality", True))
    encoder = MolecularEncoder(use_chirality=use_chirality)
    model = MultiPropertyPredictor(encoder, hidden_dim=256)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    print(f"  ✓ GNN chargé (use_chirality={use_chirality})")
    return model


# ═══════════════════════════════════════════════════════════════════════════

def load_hiv_data(data_dir):
    with open(os.path.join(data_dir, "data_paths.json")) as f:
        dp = json.load(f)
    tr = pd.read_csv(dp["hiv"]["train"])
    va = pd.read_csv(dp["hiv"]["val"])
    te = pd.read_csv(dp["hiv"]["test"])
    lbl = next(c for c in ("hiv_HIV_active", "HIV_active", "activity")
               if c in te.columns)
    return tr, va, te, lbl


def eval_strategy(name, y_true, scores, smiles, train_actives):
    """full_screening_report + AUC pour une stratégie donnée."""
    rep = full_screening_report(y_true, scores, smiles=smiles,
                                training_actives=train_actives)
    return {
        "auc": float(rep.get("roc_auc", float("nan"))),
        "ef_1pct": float(rep.get("ef_1pct", float("nan"))),
        "ef_5pct": float(rep.get("ef_5pct", float("nan"))),
        "ef_10pct": float(rep.get("ef_10pct", float("nan"))),
        "bedroc_alpha20": float(rep.get("bedroc_alpha20", float("nan"))),
        "hit_rate_top_100": float(rep.get("hit_rate_top_100", float("nan"))),
        "recall_top_10pct": float(rep.get("recall_top_10pct", float("nan"))),
        "prc_auc": float(rep.get("prc_auc", float("nan"))),
        "_full": rep,
    }


def drugbank_gate(arv_scores, arv_smiles, test_scores, test_smiles):
    """Validate_model_recall pour une stratégie (ARV + test combinés)."""
    combined = pd.DataFrame({
        "smiles": list(arv_smiles) + list(test_smiles),
        "hiv_score": np.concatenate([arv_scores, test_scores]),
    })
    g = validate_model_recall(combined, smiles_col="smiles", score_col="hiv_score",
                              top_pct=0.10, min_recall=25)
    return {"recall": g["recall"], "total": g["total"],
            "recall_rate": g["recall_rate"], "verdict": g["verdict"]}


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = str(EXTERNAL_DIR / "phase3")
    tr, va, te, lbl = load_hiv_data(data_dir)
    print(f"HIV train/val/test: {len(tr)}/{len(va)}/{len(te)} | label={lbl}")
    train_actives = tr[tr[lbl] == 1]["smiles"].tolist()

    # ── 1. Entraîner RF (identique benchmark_baselines) ──
    print("\n▶ Entraînement RF+ECFP4 (n_estimators=500, class_weight=balanced)")
    Xtr, mtr = smiles_to_morgan_array(tr["smiles"].tolist())
    ytr = tr[lbl].values
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42, class_weight="balanced",
    )
    rf.fit(Xtr[mtr], ytr[mtr])
    print(f"  RF entraîné sur {int(mtr.sum())} molécules valides")

    # ── 2. Charger GNN-only ──
    gnn_ckpt = str(CHECKPOINT_DIR / "phase3_improved" / "best_val.pth")
    if not os.path.exists(gnn_ckpt):
        print(f"❌ GNN checkpoint absent: {gnn_ckpt}")
        sys.exit(1)
    print(f"\n▶ Chargement GNN: {gnn_ckpt}")
    gnn = load_gnn(gnn_ckpt, device)

    # ── 3. Scores alignés val + test ──
    print("\n▶ Prédictions alignées (val + test)")
    gnn_va = gnn_scores_aligned(gnn, va["smiles"].tolist(), device)
    rf_va = rf_scores_aligned(rf, va["smiles"].tolist())
    gnn_te = gnn_scores_aligned(gnn, te["smiles"].tolist(), device)
    rf_te = rf_scores_aligned(rf, te["smiles"].tolist())

    # Intersection des valides (ni GNN ni RF NaN)
    va_valid = ~np.isnan(gnn_va) & ~np.isnan(rf_va)
    te_valid = ~np.isnan(gnn_te) & ~np.isnan(rf_te)
    print(f"  Val valides: {int(va_valid.sum())}/{len(va)} | "
          f"Test valides: {int(te_valid.sum())}/{len(te)}")

    y_va = va[lbl].values[va_valid]
    y_te = te[lbl].values[te_valid]
    smi_te = te["smiles"].values[te_valid].tolist()
    gnn_va, rf_va = gnn_va[va_valid], rf_va[va_valid]
    gnn_te, rf_te = gnn_te[te_valid], rf_te[te_valid]

    # ── 4. Meta-learner LR (entraîné sur VAL uniquement) ──
    print("\n▶ Meta-learner LR (features: [gnn_score, rf_score], fit sur val)")
    Xva_meta = np.column_stack([gnn_va, rf_va])
    Xte_meta = np.column_stack([gnn_te, rf_te])
    meta = LogisticRegression(max_iter=1000)
    meta.fit(Xva_meta, y_va)
    meta_te = meta.predict_proba(Xte_meta)[:, 1]
    print(f"  Coefs meta: gnn={meta.coef_[0][0]:.3f}, rf={meta.coef_[0][1]:.3f}, "
          f"intercept={meta.intercept_[0]:.3f}")

    # ── 5. Rank-average (non-paramétrique, scale-free) ──
    rank_te = (rankdata(gnn_te) + rankdata(rf_te)) / 2.0

    # ── 6. Scores ARV (DrugBank) pour chaque stratégie ──
    print("\n▶ Scores ARV DrugBank")
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    gnn_arv = gnn_scores_aligned(gnn, arv_smi, device)
    rf_arv = rf_scores_aligned(rf, arv_smi)
    arv_valid = ~np.isnan(gnn_arv) & ~np.isnan(rf_arv)
    arv_smi_v = [s for s, v in zip(arv_smi, arv_valid) if v]
    gnn_arv, rf_arv = gnn_arv[arv_valid], rf_arv[arv_valid]
    print(f"  ARV valides: {int(arv_valid.sum())}/{len(arvs)}")

    meta_arv = meta.predict_proba(np.column_stack([gnn_arv, rf_arv]))[:, 1]
    # Rank-avg ARV : rang au sein du combiné ARV+test pour cohérence du gate
    #   On recalcule rank sur la concat ARV+test (le gate prend le top-10% global).
    gnn_cat = np.concatenate([gnn_arv, gnn_te])
    rf_cat = np.concatenate([rf_arv, rf_te])
    rank_cat = (rankdata(gnn_cat) + rankdata(rf_cat)) / 2.0
    rank_arv_cat, rank_te_cat = rank_cat[:len(gnn_arv)], rank_cat[len(gnn_arv):]

    # ── 7. Évaluation des 4 stratégies ──
    print("\n" + "═" * 60)
    print("ÉVALUATION — 4 stratégies (même test aligné)")
    print("═" * 60)

    strategies = {
        "gnn_only": gnn_te,
        "rf_only": rf_te,
        "rank_average": rank_te,
        "lr_meta": meta_te,
    }
    results = {}
    for name, sc in strategies.items():
        m = eval_strategy(name, y_te, sc, smi_te, train_actives)
        # DrugBank
        if name == "gnn_only":
            db = drugbank_gate(gnn_arv, arv_smi_v, gnn_te, smi_te)
        elif name == "rf_only":
            db = drugbank_gate(rf_arv, arv_smi_v, rf_te, smi_te)
        elif name == "rank_average":
            db = drugbank_gate(rank_arv_cat, arv_smi_v, rank_te_cat, smi_te)
        else:  # lr_meta
            db = drugbank_gate(meta_arv, arv_smi_v, meta_te, smi_te)
        m["drugbank"] = db
        results[name] = m
        print(f"\n[{name}]")
        print(f"  AUC={m['auc']:.4f} EF@1%={m['ef_1pct']:.2f} EF@5%={m['ef_5pct']:.2f} "
              f"BEDROC={m['bedroc_alpha20']:.4f}")
        print(f"  hit@100={m['hit_rate_top_100']:.3f} recall@10%={m['recall_top_10pct']:.3f} "
              f"PRC={m['prc_auc']:.4f}")
        print(f"  DrugBank={db['recall']}/{db['total']} ({db['recall_rate']*100:.1f}%) "
              f"→ {db['verdict']}")

    # ── 8. Tableau comparatif + sauvegarde ──
    print("\n" + "═" * 60)
    print(f"{'Stratégie':<16}{'AUC':>8}{'EF@1%':>8}{'BEDROC':>9}{'hit100':>8}{'DrugBank':>10}")
    print("─" * 60)
    for name, m in results.items():
        db = m["drugbank"]
        print(f"{name:<16}{m['auc']:>8.4f}{m['ef_1pct']:>8.2f}{m['bedroc_alpha20']:>9.4f}"
              f"{m['hit_rate_top_100']:>8.3f}{str(db['recall'])+'/'+str(db['total']):>10}")

    out = {
        "n_test_aligned": int(te_valid.sum()),
        "n_val_aligned": int(va_valid.sum()),
        "n_actives_test": int(y_te.sum()),
        "meta_coef": {"gnn": float(meta.coef_[0][0]), "rf": float(meta.coef_[0][1]),
                      "intercept": float(meta.intercept_[0])},
        "strategies": {k: {kk: vv for kk, vv in v.items() if kk != "_full"}
                       for k, v in results.items()},
    }
    save_dir = CHECKPOINT_DIR / "phase3_stacking"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "stacking_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Rapport: {out_path}")


if __name__ == "__main__":
    main()
