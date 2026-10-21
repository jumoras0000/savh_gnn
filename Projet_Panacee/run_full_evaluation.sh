#!/usr/bin/env bash
# Pipeline complet post-training Phase 3 — orchestre toutes les améliorations P2.
#
# Enchaîne :
#   1. Évaluation modèle base (déjà sauvegardé en phase3_eval.json)
#   2. Baselines classiques (RF + LR) pour comparaison honnête
#   3. Analyse 3D (ARV vs HIV actifs/inactifs)
#   4. Fine-tune HIV ciblé via ChEMBL (anti-fuite ARV)
#   5. Évaluation modèle fine-tuné
#   6. Ensemble 5-seed (OPT — long, à activer avec --with_ensemble)
#   7. Rapport consolidé final
#
# Usage :
#   bash run_full_evaluation.sh                  # pipeline complet sans ensemble
#   bash run_full_evaluation.sh --with_ensemble  # inclut ensemble 5-seed (long)

set -e
cd "$(dirname "$0")"

WITH_ENSEMBLE=0
for arg in "$@"; do
    case "$arg" in
        --with_ensemble) WITH_ENSEMBLE=1 ;;
    esac
done

mkdir -p reports
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=~/full_eval_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

echo "═══ Pipeline évaluation complet | $(date) ═══"
echo "Log : $LOG"

# ── 1. Base model eval ───────────────────────────────────────
echo ""
echo "▶ Étape 1/7 : Évaluation modèle base"
python evaluate_phase3.py \
    --save_report reports/phase3_eval_base_${STAMP}.json \
    --save_md reports/phase3_eval_base_${STAMP}.md

# ── 2. Baselines ─────────────────────────────────────────────
echo ""
echo "▶ Étape 2/7 : Baselines RF + LR"
python benchmark_baselines.py \
    --save_report reports/baselines_${STAMP}.json

# ── 3. Analyse 3D ────────────────────────────────────────────
echo ""
echo "▶ Étape 3/7 : Analyse 3D ARV vs HIV"
python analyze_3d_hiv_actives.py \
    --n_conformers 5 --max_per_group 100 \
    --save_report reports/analysis_3d_${STAMP}.json \
    || echo "  ⚠ Analyse 3D a échoué (non-bloquant)"

# ── 4. Fine-tune HIV ciblé ───────────────────────────────────
echo ""
echo "▶ Étape 4/7 : Fine-tune HIV ChEMBL ciblé (anti-fuite ARV)"
# Si ChEMBL pas téléchargé, le télécharger d'abord
if [ ! -f data/external/chembl_hiv/chembl_hiv_merged.csv ]; then
    echo "  Téléchargement ChEMBL HIV targets..."
    python -m src.preprocessing.chembl_hiv_loader
fi
python finetune_hiv_chembl.py \
    --epochs 100 --batch_size 64 --patience 20 \
    --save_path checkpoints/phase3/panacee_phase3_hiv_chembl_${STAMP}.pth

# ── 5. Évaluation modèle fine-tuné ───────────────────────────
echo ""
echo "▶ Étape 5/7 : Évaluation modèle fine-tuné"
python evaluate_phase3.py \
    --checkpoint checkpoints/phase3/panacee_phase3_hiv_chembl_${STAMP}.pth \
    --save_report reports/phase3_eval_hiv_chembl_${STAMP}.json \
    --save_md reports/phase3_eval_hiv_chembl_${STAMP}.md

# ── 6. Ensemble 5-seed (optionnel) ──────────────────────────
if [ "$WITH_ENSEMBLE" -eq 1 ]; then
    echo ""
    echo "▶ Étape 6/7 : Ensemble 5-seed (long ~5-7h)"
    python run_phase3_ensemble.py --n_seeds 5 --epochs 500 --batch_size 128
else
    echo ""
    echo "▶ Étape 6/7 : Ensemble 5-seed SKIPÉ (relancer avec --with_ensemble)"
fi

# ── 7. Rapport consolidé ────────────────────────────────────
echo ""
echo "▶ Étape 7/7 : Rapport consolidé"
python -c "
import json, os
from pathlib import Path

stamp = '${STAMP}'
reports_dir = Path('reports')
files = {
    'base_eval': reports_dir / f'phase3_eval_base_{stamp}.json',
    'baselines': reports_dir / f'baselines_{stamp}.json',
    'analysis_3d': reports_dir / f'analysis_3d_{stamp}.json',
    'hiv_chembl_eval': reports_dir / f'phase3_eval_hiv_chembl_{stamp}.json',
}

summary = {'timestamp': stamp, 'reports': {}}
for name, p in files.items():
    if p.exists():
        with open(p) as f:
            summary['reports'][name] = json.load(f)
    else:
        summary['reports'][name] = {'status': 'missing'}

# Comparaison AUC + DrugBank gate
md = ['# Rapport consolidé — Panacée GNN anti-VIH\n', f'Timestamp : {stamp}\n\n']
md.append('## Comparaison AUC HIV test (scaffold split global)\n\n')
md.append('| Modèle | ROC-AUC | EF@1% | BEDROC α=20 | DrugBank recall |\n')
md.append('|---|---|---|---|---|\n')

def _get(d, *keys):
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if d else None

base = summary['reports'].get('base_eval', {})
hiv_ft = summary['reports'].get('hiv_chembl_eval', {})
baselines = summary['reports'].get('baselines', {})

for name, src in [('Panacée Phase 3 (base)', base),
                   ('Panacée Phase 3 + HIV ChEMBL fine-tune', hiv_ft)]:
    pm = src.get('pharma_metrics_hiv_test', {}) if isinstance(src, dict) else {}
    db = src.get('drugbank_validation', {}) if isinstance(src, dict) else {}
    md.append(f'| {name} | {pm.get(\"roc_auc\", \"N/A\"):.4f} | '
              f'{pm.get(\"ef_1pct\", 0):.2f} | '
              f'{pm.get(\"bedroc_alpha20\", 0):.4f} | '
              f'{db.get(\"recall\", 0)}/{db.get(\"total\", 0)} '
              f'({db.get(\"recall_rate\", 0)*100:.1f}%) |\n'
              if pm else f'| {name} | - | - | - | - |\n')

for name, bl in baselines.items():
    rpt = bl.get('report', {})
    md.append(f'| {bl.get(\"model\", name)} | '
              f'{rpt.get(\"roc_auc\", 0):.4f} | '
              f'{rpt.get(\"ef_1pct\", 0):.2f} | '
              f'{rpt.get(\"bedroc_alpha20\", 0):.4f} | - |\n')

out = reports_dir / f'summary_{stamp}.md'
out.write_text(''.join(md), encoding='utf-8')
print(f'  ✅ Rapport consolidé : {out}')

with open(reports_dir / f'summary_{stamp}.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
print(f'  ✅ JSON complet : {reports_dir}/summary_{stamp}.json')
"

echo ""
echo "═══ ✅ Pipeline complet terminé | $(date) ═══"
echo "Rapports : reports/*_${STAMP}.*"
echo "Log : $LOG"
