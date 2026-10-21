#!/usr/bin/env bash
# Setup serveur ESIEA — install deps + lance pipeline complet Phase 2→3
# Usage : bash setup_server.sh [--skip-install] [--phase3-only]

set -e

SKIP_INSTALL=0
PHASE3_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-install) SKIP_INSTALL=1 ;;
        --phase3-only)  PHASE3_ONLY=1 ;;
    esac
done

cd "$(dirname "$0")"

echo "═══ Setup Panacée serveur ═══"

if [ "$SKIP_INSTALL" -eq 0 ]; then
    echo ""
    echo "▶ Installation des dépendances..."
    pip install -q -r requirements.txt
    python -c "
import importlib
mods = ['pandas','numpy','scipy','sklearn','torch','torch_geometric','rdkit',
        'tqdm','psutil','requests','httpx','matplotlib','xgboost','lightgbm',
        'statsmodels','pydantic','pyarrow']
ok, fail = [], []
for m in mods:
    try: importlib.import_module(m); ok.append(m)
    except Exception as e: fail.append((m, str(e)[:60]))
print(f'  ✓ {len(ok)}/{len(mods)} modules OK')
if fail:
    print(f'  ✗ {len(fail)} manquants: {fail}')
    exit(1)
"
fi

echo ""
echo "▶ Vérification GPU..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  ✓ GPU: {gpu} ({vram:.1f} GB)')
else:
    print('  ⚠ Pas de GPU — training CPU (très lent)')
"

if [ "$PHASE3_ONLY" -eq 0 ]; then
    echo ""
    echo "▶ Étape 1 : scaffold split global Phase 3..."
    python prepare_phase3_data.py

    echo ""
    echo "▶ Étape 2 : fine-tune Phase 2 (sur train Phase 3 uniquement)..."
    python run_phase2.py --phase3_data_dir data/external/phase3 --epochs 800 --batch_size 128
fi

echo ""
echo "▶ Étape 3 : training Phase 3 complet..."
python run_phase3.py --epochs 3000 --batch_size 128 --skip_checks

echo ""
echo "═══ ✅ Pipeline complet terminé ═══"
echo "Checkpoints:"
echo "  Phase 1: checkpoints/phase1/sovereign_encoder_v1.pth"
echo "  Phase 2: checkpoints/phase2/best_toxicity_model.pth"
echo "  Phase 3: checkpoints/phase3/panacee_phase3_complete.pth"
