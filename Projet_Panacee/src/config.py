"""
Configuration centralisée pour Projet Panacée.
Tous les hyperparamètres, chemins et constantes au même endroit.
"""
import os
import sys
from pathlib import Path

import torch

# ══════════════════════════════════════════════════════════════════════
# CHEMINS
# ══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
PLOT_DIR = PROJECT_ROOT / "plots"

# Créer les dossiers nécessaires
for d in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombre de workers du DataLoader. Sur Windows : 0 (le spawn re-sérialise tout).
# Sinon : tous les cœurs CPU disponibles, pour nourrir le GPU sans famine.
# Surchargé par la variable d'environnement PANACEE_NUM_WORKERS (utile sur Kaggle).
_DEFAULT_WORKERS = 0 if sys.platform == "win32" else (os.cpu_count() or 4)
NUM_WORKERS = int(os.environ.get("PANACEE_NUM_WORKERS", _DEFAULT_WORKERS))
PIN_MEMORY = torch.cuda.is_available()

# Nombre de batchs préchargés par worker (recouvre calcul GPU et préparation CPU).
PREFETCH_FACTOR = int(os.environ.get("PANACEE_PREFETCH_FACTOR", 4))


def loader_kwargs(num_workers: int | None = None) -> dict:
    """Arguments DataLoader optimisés pour saturer GPU + CPU (anti-famine).

    - workers persistants : évite de relancer les processus à chaque epoch
    - prefetch : prépare les prochains batchs pendant que le GPU calcule
    """
    nw = NUM_WORKERS if num_workers is None else num_workers
    kw = {"num_workers": nw, "pin_memory": PIN_MEMORY}
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = PREFETCH_FACTOR
    return kw

# ══════════════════════════════════════════════════════════════════════
# FEATURES MOLÉCULAIRES
# ══════════════════════════════════════════════════════════════════════
# Atomes autorisés (mapping numéro atomique → index)
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C,N,O,F,P,S,Cl,Br,I
MAX_ATOMIC_NUM = 118

# Dimensions des features
ATOM_FEATURE_DIM = 9   # nombre de features par atome
BOND_FEATURE_DIM = 6   # nombre de features par liaison

# Features atomiques encodées :
#   0: Numéro atomique (normalisé /118)
#   1: Degré (normalisé /6)
#   2: Charge formelle (normalisée)
#   3: Hybridation (one-hot réduit, normalisé /6)
#   4: Aromatique (0/1)
#   5: Électrons radicaux (normalisé)
#   6: Valence implicite (normalisée /6)
#   7: Nombre total H (normalisé /8)
#   8: Dans un cycle (0/1)

# ══════════════════════════════════════════════════════════════════════
# MODÈLE - ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════
HIDDEN_DIM = 256
NUM_GNN_LAYERS = 6
OUTPUT_DIM = 256
DROPOUT = 0.2                # augmenté de 0.1 à 0.2 pour meilleure régularisation

# Type de convolution : "attention" (GATv2 edge-aware, recommandé) ou "mpnn" (Gilmer)
CONV_TYPE = "attention"
ATTENTION_HEADS = 4         # nombre de têtes d'attention (doit diviser HIDDEN_DIM)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1 — PRÉ-ENTRAÎNEMENT MGM
# ══════════════════════════════════════════════════════════════════════
PHASE1 = {
    "mask_prob": 0.15,       # probabilité de masquer un atome
    "epochs": 100,
    "batch_size": 64,        # augmenté de 32 → 64 pour convergence plus stable
    "lr": 5e-4,              # learning rate pré-entraînement
    "lr_min": 1e-6,
    "weight_decay": 1e-5,
    "warmup_epochs": 5,      # warmup linéaire
    "patience": 15,          # early stopping
    "grad_clip": 1.0,
    "val_split": 0.1,
    "checkpoint_name": "sovereign_encoder_v1.pth",
}

# ══════════════════════════════════════════════════════════════════════
# PHASE 2 — FINE-TUNING TOXICITÉ
# ══════════════════════════════════════════════════════════════════════
PHASE2 = {
    # Tox21 task names (NR = Nuclear Receptor, SR = Stress Response)
    "tox21_tasks": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ],
    "num_tasks_tox21": 12,
    "num_tasks_sider": 27,

    # Entraînement
    "epochs": 80,
    "batch_size": 64,
    "lr_encoder": 1e-4,       # LR pour l'encodeur pré-entraîné (plus bas)
    "lr_head": 1e-3,          # LR pour la tête de classification (plus haut)
    "lr_min": 1e-6,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "patience": 20,           # early stopping
    "grad_clip": 1.0,

    # Classification
    "threshold_default": 0.5,
    "use_optimal_threshold": True,
    "freeze_encoder_epochs": 10,  # geler encodeur les N premières epochs
}

# ══════════════════════════════════════════════════════════════════════
# PHASE 3 — ANALYSE MULTI-PROPRIÉTÉS & IA RAISONNEMENT
# ══════════════════════════════════════════════════════════════════════
PHASE3 = {
    # Propriétés prédites (multi-tâches étendues)
    "property_heads": {
        "toxicity": 12,       # 12 tâches Tox21
        "efficacy": 1,        # Score d'efficacité prédit
        "solubility": 1,      # Solubilité aqueuse
        "lipophilicity": 1,   # LogP
        "bioavailability": 1, # Biodisponibilité orale
        "metabolic_stability": 1,  # Stabilité métabolique
    },
    "total_tasks": 17,

    # Entraînement
    "epochs": 100,
    "batch_size": 32,
    "lr_encoder": 5e-5,       # LR très bas pour encodeur pré-entraîné Phase 2
    "lr_heads": 5e-4,         # LR pour les nouvelles têtes
    "lr_reasoner": 1e-4,      # LR pour le module IA raisonnement
    "lr_min": 1e-7,
    "weight_decay": 1e-4,
    "warmup_epochs": 8,
    "patience": 25,
    "grad_clip": 1.0,
    "freeze_encoder_epochs": 5,

    # Module IA Raisonnement
    "reasoner_hidden_dim": 512,
    "reasoner_num_heads": 8,    # Multi-head attention
    "reasoner_num_layers": 4,   # Transformer layers
    "reasoner_dropout": 0.15,
    "max_molecules_combo": 5,   # Max molécules par combinaison

    # Analyse combinatoire
    "dose_levels": [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],  # mg/kg
    "synergy_threshold": 0.7,   # Seuil synergie
    "confidence_min": 0.6,      # Confiance minimale

    # Checkpoint
    "checkpoint_name": "panacee_phase3_complete.pth",
}

# ══════════════════════════════════════════════════════════════════════
# NOMS DES COLONNES
# ══════════════════════════════════════════════════════════════════════
SMILES_COLUMN_CANDIDATES = [
    "smiles", "SMILES", "canonical_smiles", "ids", "mol", "molecule",
]
