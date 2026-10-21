# -*- coding: utf-8 -*-
"""
Évaluation complète Phase 3 — applique les modules P2 au checkpoint final.

Génère un rapport scientifique rigoureux avec :
  1. Métriques pharma : EF@1%/5%/10%, BEDROC α=20/80, hit rate top-K
  2. Validation DrugBank : recall@top-10% des 35 ARV approuvés (gate scientifique)
  3. Conformal prediction : prediction sets {active, inactive, both, none} + coverage
  4. Tanimoto AD : classification in/out of applicability domain
  5. Novelty score vs training set

Usage :
    python evaluate_phase3.py                          # eval sur test set par défaut
    python evaluate_phase3.py --checkpoint path.pth    # eval sur ckpt spécifique
    python evaluate_phase3.py --save_report report.json

Produit : report.json + report.md (lisible humain)
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

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR, PHASE3
from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyPredictor
from src.preprocessing.graph_builder import smiles_to_graph
from src.utils.safe_load import safe_load_checkpoint
from src.validation.conformal_prediction import (
    ConformalClassifier,
    TanimotoApplicabilityDomain,
    evaluate_conformal_coverage,
)
from src.validation.drugbank_antiretroviral import (
    canonicalize_arv_smiles,
    validate_model_recall,
)
from src.validation.pharma_metrics import full_screening_report


def load_model(checkpoint_path: str, device: torch.device) -> MultiPropertyPredictor:
    """Charge le modèle Phase 3 depuis un checkpoint."""
    ckpt = safe_load_checkpoint(checkpoint_path, trusted=True)
    encoder = MolecularEncoder()
    heads_config = PHASE3["property_heads"]
    model = MultiPropertyPredictor(encoder, heads_config, hidden_dim=256)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model


def predict_batch(model, smiles_list, device, batch_size: int = 128) -> dict:
    """Prédit toutes les propriétés pour une liste de SMILES."""
    from torch_geometric.data import Batch

    all_scores = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bbb_penetration", "clinical_toxicity"]}
    valid_smiles = []

    for i in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[i:i + batch_size]
        graphs = []
        valid_mask = []
        for s in chunk:
            try:
                g = smiles_to_graph(s)
                if g is not None:
                    graphs.append(g)
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            except Exception:
                valid_mask.append(False)

        if not graphs:
            continue

        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            preds = model(batch)

        # Sigmoid sur classif, raw sur régression
        classif = {"toxicity", "efficacy", "bbb_penetration", "clinical_toxicity"}
        for k, v in preds.items():
            arr = v.cpu().numpy()
            if k in classif:
                arr = 1.0 / (1.0 + np.exp(-arr))  # sigmoid
            all_scores[k].append(arr)

        for j, ok in enumerate(valid_mask):
            if ok:
                valid_smiles.append(chunk[j])

    results = {}
    for k, chunks in all_scores.items():
        if chunks:
            results[k] = np.concatenate(chunks, axis=0)
    return results, valid_smiles


def main():
    import argparse

    p = argparse.ArgumentParser(description="Évaluation complète Phase 3 avec P2 modules")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path checkpoint Phase 3 (défaut: checkpoints/phase3/panacee_phase3_complete.pth)")
    p.add_argument("--phase3_data_dir", type=str, default=None)
    p.add_argument("--save_report", type=str, default="phase3_evaluation_report.json")
    p.add_argument("--save_md", type=str, default="phase3_evaluation_report.md")
    p.add_argument("--conformal_alpha", type=float, default=0.1,
                   help="Niveau d'erreur conformelle (0.1 → coverage 90%)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Chargement modèle ──
    ckpt_path = args.checkpoint or str(CHECKPOINT_DIR / "phase3" / PHASE3["checkpoint_name"])
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint introuvable : {ckpt_path}")
        sys.exit(1)
    print(f"\n▶ Chargement modèle : {ckpt_path}")
    model = load_model(ckpt_path, device)

    # ── Chargement data ──
    data_dir = args.phase3_data_dir or str(EXTERNAL_DIR / "phase3")
    paths_file = os.path.join(data_dir, "data_paths.json")
    if not os.path.exists(paths_file):
        print(f"❌ data_paths.json manquant : {paths_file}")
        sys.exit(1)
    with open(paths_file) as f:
        data_paths = json.load(f)

    hiv_train = pd.read_csv(data_paths["hiv"]["train"])
    hiv_val = pd.read_csv(data_paths["hiv"]["val"])
    hiv_test = pd.read_csv(data_paths["hiv"]["test"])
    print(f"  HIV train/val/test : {len(hiv_train)}/{len(hiv_val)}/{len(hiv_test)}")

    report = {"checkpoint": ckpt_path, "device": str(device)}

    # ═══════════════════════════════════════════════════════════════
    # 1. PRÉDICTIONS sur HIV test
    # ═══════════════════════════════════════════════════════════════
    print("\n═══ 1. Prédictions sur HIV test ═══")
    test_scores, test_valid_smiles = predict_batch(
        model, hiv_test["smiles"].tolist(), device,
    )
    hiv_proba_test = test_scores["efficacy"].ravel()
    hiv_true_test = hiv_test[hiv_test["smiles"].isin(test_valid_smiles)]["HIV_active"].values

    # ═══════════════════════════════════════════════════════════════
    # 2. MÉTRIQUES PHARMA (EF, BEDROC, hit rate)
    # ═══════════════════════════════════════════════════════════════
    print("\n═══ 2. Métriques pharma virtual screening ═══")
    train_actives_smi = hiv_train[hiv_train["HIV_active"] == 1]["smiles"].tolist()
    pharma = full_screening_report(
        hiv_true_test, hiv_proba_test,
        smiles=test_valid_smiles,
        training_actives=train_actives_smi,
    )
    report["pharma_metrics_hiv_test"] = pharma
    for k, v in pharma.items():
        if isinstance(v, (int, float, type(None))):
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")

    # ═══════════════════════════════════════════════════════════════
    # 3. VALIDATION DRUGBANK 35 ARV
    # ═══════════════════════════════════════════════════════════════
    print("\n═══ 3. Validation DrugBank 35 ARV ═══")
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    arv_scores_dict, arv_valid = predict_batch(model, arv_smi, device)
    if "efficacy" in arv_scores_dict:
        arv_proba = arv_scores_dict["efficacy"].ravel()
        print(f"  {len(arv_valid)}/{len(arvs)} ARV prédits avec succès")
        print(f"  Score HIV moyen ARV approuvés : {arv_proba.mean():.3f}")
        print(f"  Score HIV min ARV approuvés   : {arv_proba.min():.3f}")
        print(f"  Score HIV max ARV approuvés   : {arv_proba.max():.3f}")

        # Combinaison : ARV + HIV test pour validate_model_recall
        combined_smi = list(arv_valid) + test_valid_smiles
        combined_scores = np.concatenate([arv_proba, hiv_proba_test])
        combined_df = pd.DataFrame({"smiles": combined_smi, "hiv_score": combined_scores})
        drugbank_gate = validate_model_recall(
            combined_df, smiles_col="smiles", score_col="hiv_score",
            top_pct=0.10, min_recall=25,
        )
        report["drugbank_validation"] = drugbank_gate
        print(f"  Verdict : {drugbank_gate['verdict']}")
        print(f"  Recall  : {drugbank_gate['recall']}/{drugbank_gate['total']} "
              f"({drugbank_gate['recall_rate']*100:.1f}%)")
        print(f"  Trouvés : {drugbank_gate['found_arvs'][:10]}")
        print(f"  Manqués : {drugbank_gate['missing_arvs'][:10]}")

    # ═══════════════════════════════════════════════════════════════
    # 4. CONFORMAL PREDICTION (coverage garantie 1-α)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n═══ 4. Conformal prediction (α={args.conformal_alpha}) ═══")
    val_scores, val_valid_smiles = predict_batch(
        model, hiv_val["smiles"].tolist(), device,
    )
    hiv_proba_val = val_scores["efficacy"].ravel()
    hiv_true_val = hiv_val[hiv_val["smiles"].isin(val_valid_smiles)]["HIV_active"].values

    cp = ConformalClassifier(alpha=args.conformal_alpha, mondrian=True)
    cp.fit(hiv_proba_val, hiv_true_val)
    pred_sets = cp.predict_set(hiv_proba_test)
    coverage = evaluate_conformal_coverage(hiv_true_test, pred_sets,
                                            alpha=args.conformal_alpha)
    report["conformal_coverage"] = coverage
    for k, v in coverage.items():
        print(f"  {k}: {v}")

    # Décisions détaillées
    decisions = cp.predict_with_pvalues(hiv_proba_test)
    decision_counts = {}
    for d in decisions:
        decision_counts[d["decision"]] = decision_counts.get(d["decision"], 0) + 1
    report["conformal_decisions"] = decision_counts
    print(f"  Décisions : {decision_counts}")

    # ═══════════════════════════════════════════════════════════════
    # 5. TANIMOTO APPLICABILITY DOMAIN
    # ═══════════════════════════════════════════════════════════════
    print("\n═══ 5. Tanimoto Applicability Domain ═══")
    try:
        ad = TanimotoApplicabilityDomain(high_threshold=0.7, low_threshold=0.4)
        ad.fit(hiv_train["smiles"].tolist())
        ad_results = ad.classify(test_valid_smiles)
        ad_counts = {"in_domain_high": 0, "in_domain_moderate": 0,
                     "out_of_domain": 0, "invalid": 0}
        for r in ad_results:
            ad_counts[r["ad_class"]] = ad_counts.get(r["ad_class"], 0) + 1
        report["applicability_domain"] = {
            "counts": ad_counts,
            "high_threshold": 0.7,
            "low_threshold": 0.4,
            "mean_max_sim": float(np.mean([r["max_sim"] for r in ad_results
                                            if r["max_sim"] is not None])),
        }
        for k, v in ad_counts.items():
            pct = 100 * v / len(ad_results) if ad_results else 0
            print(f"  {k}: {v} ({pct:.1f}%)")
    except Exception as e:
        print(f"  ⚠ Tanimoto AD échoué : {e}")
        report["applicability_domain"] = {"error": str(e)}

    # ═══════════════════════════════════════════════════════════════
    # SAUVEGARDE RAPPORT
    # ═══════════════════════════════════════════════════════════════
    def _json_default(o):
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    report_path = args.save_report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default, ensure_ascii=False)
    print(f"\n✅ Rapport JSON : {report_path}")

    # Markdown lisible
    md = _format_markdown_report(report)
    with open(args.save_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ Rapport Markdown : {args.save_md}")


def _format_markdown_report(report: dict) -> str:
    md = ["# Rapport évaluation Phase 3 — Panacée GNN anti-VIH\n"]
    md.append(f"**Checkpoint** : `{report['checkpoint']}`\n")
    md.append(f"**Device** : {report['device']}\n\n")

    md.append("## 1. Métriques pharma virtual screening (HIV test)\n")
    pharma = report.get("pharma_metrics_hiv_test", {})
    md.append("| Métrique | Valeur |\n|---|---|\n")
    for k in ["n_total", "n_actives", "actives_fraction", "roc_auc", "prc_auc",
              "ef_1pct", "ef_5pct", "ef_10pct", "bedroc_alpha20", "bedroc_alpha80",
              "hit_rate_top_100", "hit_rate_top_500", "hit_rate_top_1000",
              "recall_top_1pct", "recall_top_5pct", "recall_top_10pct"]:
        if k in pharma:
            v = pharma[k]
            md.append(f"| {k} | {v:.4f} |\n" if isinstance(v, float) else f"| {k} | {v} |\n")

    if "novelty" in pharma:
        md.append("\n### Novelty vs training\n")
        for k, v in pharma["novelty"].items():
            md.append(f"- **{k}** : {v}\n")

    md.append("\n## 2. Validation DrugBank 35 ARV approuvés\n")
    db = report.get("drugbank_validation", {})
    md.append(f"- **Verdict** : {db.get('verdict', 'N/A')}\n")
    md.append(f"- **Recall@top-{int(db.get('top_pct', 0.1)*100)}%** : "
              f"{db.get('recall', 0)}/{db.get('total', 0)} "
              f"({db.get('recall_rate', 0)*100:.1f}%)\n")
    md.append(f"- **ARV trouvés** : {', '.join(db.get('found_arvs', []))}\n")
    md.append(f"- **ARV manqués** : {', '.join(db.get('missing_arvs', []))}\n")

    md.append("\n## 3. Conformal prediction (coverage garantie)\n")
    cov = report.get("conformal_coverage", {})
    md.append(f"- **Coverage empirique** : {cov.get('empirical_coverage', 0)*100:.1f}% "
              f"(cible ≥ {cov.get('target_coverage', 0)*100:.1f}%)\n")
    md.append(f"- **Coverage valide** : {cov.get('coverage_valid', False)}\n")
    md.append(f"- **Décisions** : {report.get('conformal_decisions', {})}\n")

    md.append("\n## 4. Tanimoto Applicability Domain\n")
    ad = report.get("applicability_domain", {})
    if "counts" in ad:
        md.append(f"- Similarité moyenne max au train : {ad.get('mean_max_sim', 0):.3f}\n")
        for k, v in ad["counts"].items():
            md.append(f"- **{k}** : {v}\n")

    return "".join(md)


if __name__ == "__main__":
    main()
