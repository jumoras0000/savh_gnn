"""
config.py – Configuration centralisée · Projet Panacée
=======================================================
Adapté Google Colab (T4 / A100) et exécution locale.
Tous les hyperparamètres, chemins et constantes au même endroit.
"""
import os
import sys
import torch

# ══════════════════════════════════════════════════════════════════════
# 1. DÉTECTION ENVIRONNEMENT
# ══════════════════════════════════════════════════════════════════════
IN_COLAB: bool = "google.colab" in sys.modules or os.path.exists("/content")

# ══════════════════════════════════════════════════════════════════════
# 2. RÉPERTOIRES  (plats, créés automatiquement)
# ══════════════════════════════════════════════════════════════════════
BASE_DIR       = "/content/panacee" if IN_COLAB else "."
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
DATA_DIR       = os.path.join(BASE_DIR, "data")
PLOT_DIR       = os.path.join(BASE_DIR, "plots")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
EXTERNAL_DIR   = os.path.join(DATA_DIR, "external")

for _d in [CHECKPOINT_DIR, LOG_DIR, DATA_DIR, PLOT_DIR, RESULTS_DIR, EXTERNAL_DIR]:
    os.makedirs(_d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 3. DEVICE & PARALLÉLISME
# ══════════════════════════════════════════════════════════════════════
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2 if IN_COLAB else 0       # Colab : 2 CPU cores disponibles
PIN_MEMORY  = torch.cuda.is_available()

# ══════════════════════════════════════════════════════════════════════
# 4. FEATURES MOLÉCULAIRES
# ══════════════════════════════════════════════════════════════════════
# 9 features atomiques (normalisées [0,1]) :
#   atomic_num/118, degree/6, formal_charge, hybridization/6,
#   is_aromatic, radical_electrons, implicit_valence/6, total_H/8, in_ring
ATOM_FEATURE_DIM = 9

# 6 features de liaison (binaires) :
#   single, double, triple, aromatic, conjugated, in_ring
BOND_FEATURE_DIM = 6

# ══════════════════════════════════════════════════════════════════════
# 5. ARCHITECTURE GNN
# ══════════════════════════════════════════════════════════════════════
# Architectures disponibles :
#   'mpnn'  – Message Passing NN (Gilmer 2017), baseline solide
#   'attfp' – Attentive FP (Xiong 2020, JACS) ← recommandé pour Tox21
#   'gin'   – Graph Isomorphism Network (Xu 2019), très expressif
#   'gps'   – GPS Transformer (Rampásek 2022), état de l'art
#   'pna'   – Principal Neighbourhood Aggregation (Corso 2020)
GNN_ARCHITECTURE = "attfp"

HIDDEN_DIM  = 256
NUM_LAYERS  = 6
OUTPUT_DIM  = 256
DROPOUT     = 0.20

# ══════════════════════════════════════════════════════════════════════
# 6. RÉGULARISATION ANTI-SURAPPRENTISSAGE
# ══════════════════════════════════════════════════════════════════════
REGULARIZATION = dict(
    dropout         = 0.20,     # dropout par couche GNN
    weight_decay    = 1e-4,     # L2 regularisation AdamW
    gradient_clip   = 1.0,      # gradient clipping (norme max)
    label_smoothing = 0.05,     # lissage des labels (classification)
    stochastic_depth= 0.10,     # drop-path rate (GPS uniquement)
    patience        = 20,       # early stopping
    min_delta       = 1e-4,     # seuil d'amélioration min
)

# ══════════════════════════════════════════════════════════════════════
# 7. PHASE 1 — PRÉ-ENTRAÎNEMENT MGM
# ══════════════════════════════════════════════════════════════════════
PHASE1 = dict(
    mask_prob        = 0.15,   # probabilité de masquer un atome (BERT-like)
    epochs           = 100,
    batch_size       = 64,
    lr               = 5e-4,
    lr_min           = 1e-6,
    weight_decay     = 1e-5,
    warmup_epochs    = 5,      # warmup linéaire
    patience         = 15,     # early stopping sur val_loss
    grad_clip        = 1.0,
    val_split        = 0.10,
    checkpoint_name  = "phase1_encoder.pth",
)

# ══════════════════════════════════════════════════════════════════════
# 8. PHASE 2 — FINE-TUNING TOXICITÉ
# ══════════════════════════════════════════════════════════════════════
PHASE2 = dict(
    tox21_tasks = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ],
    num_tasks_tox21         = 12,
    epochs                  = 80,
    batch_size              = 64,
    lr_encoder              = 1e-4,    # encodeur pré-entraîné : LR bas
    lr_head                 = 1e-3,    # tête de classification : LR haut
    lr_min                  = 1e-6,
    weight_decay            = 1e-4,
    warmup_epochs           = 5,
    patience                = 20,
    grad_clip               = 1.0,
    freeze_encoder_epochs   = 10,      # dégel progressif de l'encodeur
    checkpoint_name         = "phase2_toxicity.pth",
)

# ══════════════════════════════════════════════════════════════════════
# 9. PHASE 3 — MULTI-PROPRIÉTÉS & RAISONNEMENT
# ══════════════════════════════════════════════════════════════════════
PHASE3 = dict(
    property_heads = dict(
        toxicity            = 12,  # Tox21 (classification)
        efficacy            = 1,   # Score d'efficacité (classification)
        solubility          = 1,   # LogS (régression)
        lipophilicity       = 1,   # LogP (régression)
        bioavailability     = 1,   # BBBP (classification)
        metabolic_stability = 1,   # ClinTox (classification)
    ),
    total_tasks             = 17,
    epochs                  = 100,
    batch_size              = 32,
    lr_encoder              = 5e-5,  # très bas pour encodeur affiné
    lr_heads                = 5e-4,
    lr_min                  = 1e-7,
    weight_decay            = 1e-4,
    warmup_epochs           = 8,
    patience                = 25,
    grad_clip               = 1.0,
    freeze_encoder_epochs   = 5,
    reasoner_hidden_dim     = 512,
    reasoner_num_heads      = 8,
    reasoner_num_layers     = 4,
    reasoner_dropout        = 0.15,
    max_molecules_combo     = 5,
    synergy_threshold       = 0.70,
    checkpoint_name         = "phase3_complete.pth",
)

# ══════════════════════════════════════════════════════════════════════
# 10. DÉTECTION COLONNE SMILES
# ══════════════════════════════════════════════════════════════════════
SMILES_COLUMN_CANDIDATES = [
    "smiles", "SMILES", "canonical_smiles", "ids", "mol", "molecule",
]
