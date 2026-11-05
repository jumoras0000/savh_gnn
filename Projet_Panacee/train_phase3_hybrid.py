# -*- coding: utf-8 -*-
"""
Phase 3 HYBRID — GNN + Morgan ECFP4 fingerprints.

Concatène les features apprises (GNN embedding 256d) avec Morgan FP (2048d)
avant la couche partagée → la fusion apprend à combiner les deux signaux.

Motivation mesurée (pas supposée) :
  - RF+ECFP4 bat le GNN sur EF@1% (17.8 vs 15.4), BEDROC (0.328 vs 0.284),
    hit rate top-100 (23% vs 16%), et DrugBank recall (20/34 vs 17/34).
  - Le GNN gagne en AUC globale (0.719 vs 0.652).
  → Le modèle hybrid vise le meilleur des deux : AUC du GNN + précision
    top-K des fingerprints.

Référence : Yang et al. 2019 (Chemprop), Jiang et al. 2021.

Inclut :
  - Resume + checkpoint atomique (robustesse SIGHUP)
  - Calibration Platt en fin de run (fix ambiguïté conformal)
  - Éval pharma complète (EF, BEDROC, DrugBank)

Usage :
    python train_phase3_hybrid.py --epochs 200 --resume
    python train_phase3_hybrid.py --eval_only --checkpoint path.pth
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

from src.config import CHECKPOINT_DIR, EXTERNAL_DIR, PHASE3
from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyPredictor
from src.preprocessing.graph_builder import smiles_to_graph
from src.training.improvements import (
    BCEWithLabelSmoothing,
    CosineRestartScheduler,
    FocalBCELoss,
    make_weighted_sampler,
)
from src.utils.fingerprints import smiles_to_morgan_array
from src.utils.reproducibility import set_seed
from src.utils.safe_load import safe_load_checkpoint

MORGAN_DIM = 2048
MORGAN_RADIUS = 2


# ═══════════════════════════════════════════════════════════════════════════
# Dataset : graphe + Morgan FP pré-calculé
# ═══════════════════════════════════════════════════════════════════════════

class HybridSmilesDataset(torch.utils.data.Dataset):
    """Dataset qui fournit graphe PyG + Morgan FP pour chaque molécule."""

    def __init__(self, smiles_list, labels, morgan_fps, train_mode: bool = True):
        """
        Args:
            smiles_list: list[str]
            labels: list[float]
            morgan_fps: np.ndarray [N, 2048] uint8 pré-calculé
            train_mode: True → pas de cache (shuffle sampler)
        """
        self.smiles = list(smiles_list)
        self.labels = list(labels)
        self.morgan_fps = morgan_fps  # [N, 2048] np.uint8
        self.train_mode = train_mode
        self._cache = {}

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        lbl = self.labels[idx]

        if smi in self._cache:
            g = self._cache[smi].clone()
        else:
            try:
                g = smiles_to_graph(smi)
            except Exception:
                g = None
            if not self.train_mode and g is not None:
                self._cache[smi] = g.clone()

        if g is None:
            g = smiles_to_graph("C")  # fallback minimal

        g.y = torch.tensor([float(lbl)], dtype=torch.float32)
        # Attacher le Morgan FP au graphe → sera collated par Batch.from_data_list.
        # IMPORTANT : shape [1, 2048] (pas [2048]) car PyG concatène sur dim 0.
        # [1, 2048] × B → cat dim 0 → [B, 2048] ✅
        # [2048] × B → cat dim 0 → [B*2048] ❌ (1D, crash au torch.cat avec mol_emb)
        g.morgan_fp = torch.tensor(
            self.morgan_fps[idx], dtype=torch.float32,
        ).unsqueeze(0)  # [1, 2048]
        return g


def _collate(batch):
    return Batch.from_data_list([b for b in batch if b is not None])


def _atomic_save(obj, path):
    """Sauvegarde atomique (.tmp → rename) pour résister aux kills."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════════
# Chargement modèle avec gestion shape mismatch pour shared_layer
# ═══════════════════════════════════════════════════════════════════════════

def load_hybrid_model(ckpt_path, device, morgan_dim=MORGAN_DIM):
    """
    Charge un checkpoint GNN-only (shared_layer input=256) dans un modèle
    hybrid (shared_layer input=256+2048).

    Gestion du mismatch :
      - shared_layer.0.weight : ckpt [256, 256] → nouveau [256, 256+2048]
        → copie les 256 premières colonnes, initialise les 2048 suivantes
          avec Kaiming (random small → Morgan commence en "bruit", le training
          apprend à l'utiliser).
      - shared_layer.0.bias : même taille [256] → copie directe.
      - Tous les autres poids : strict=False tolère les clés identiques.
    """
    ckpt = safe_load_checkpoint(ckpt_path, trusted=True)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    use_chirality = cfg.get("use_chirality", PHASE3.get("use_chirality", True))

    encoder = MolecularEncoder(use_chirality=use_chirality)
    model = MultiPropertyPredictor(
        encoder, hidden_dim=256, morgan_dim=morgan_dim,
    ).to(device)

    # Charger en gérant le shape mismatch de shared_layer
    state = ckpt["model_state_dict"]
    model_state = model.state_dict()

    loaded, skipped = 0, 0
    for k, v in state.items():
        if k not in model_state:
            skipped += 1
            continue
        if model_state[k].shape == v.shape:
            model_state[k].copy_(v)
            loaded += 1
        elif k == "shared_layer.0.weight":
            # ckpt: [256, 256], nouveau: [256, 256+2048]
            # Copier la partie GNN, Kaiming init la partie Morgan
            old_in = v.shape[1]  # 256
            model_state[k][:, :old_in].copy_(v)
            nn.init.kaiming_normal_(model_state[k][:, old_in:], nonlinearity="linear")
            print(f"  shared_layer.0.weight: [{v.shape}] → [{model_state[k].shape}] "
                  f"(GNN copié, Morgan Kaiming init)")
            loaded += 1
        else:
            print(f"  ⚠ shape mismatch ignoré : {k} ckpt={v.shape} model={model_state[k].shape}")
            skipped += 1

    model.load_state_dict(model_state, strict=False)
    print(f"  ✓ Chargé {loaded} params, {skipped} ignorés, chirality={use_chirality}")
    model.to(device).eval()
    return model, use_chirality


# ═══════════════════════════════════════════════════════════════════════════

def load_hiv_data(data_dir: str) -> dict:
    paths_file = os.path.join(data_dir, "data_paths.json")
    with open(paths_file) as f:
        dp = json.load(f)
    hiv_tr = pd.read_csv(dp["hiv"]["train"])
    hiv_va = pd.read_csv(dp["hiv"]["val"])
    hiv_te = pd.read_csv(dp["hiv"]["test"])
    lbl = next(c for c in ("hiv_HIV_active", "HIV_active", "activity")
                if c in hiv_te.columns)
    return {"train": hiv_tr, "val": hiv_va, "test": hiv_te, "label_col": lbl}


def precompute_morgan(smiles_list, radius=MORGAN_RADIUS, n_bits=MORGAN_DIM):
    """Pré-calcule Morgan FP pour toute la liste. Molécules invalides → vecteur nul."""
    print(f"  Pré-calcul Morgan FP ({len(smiles_list)} mol, radius={radius}, bits={n_bits})...")
    X, valid = smiles_to_morgan_array(smiles_list, radius=radius, n_bits=n_bits)
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"  ⚠ {n_invalid} molécules invalides → Morgan vecteur nul")
    return X  # [N, n_bits] uint8


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 3 Hybrid GNN+Morgan training")
    p.add_argument("--base_checkpoint", type=str, default=None,
                   help="Checkpoint warm-start (défaut: best_val Phase 3 improved)")
    p.add_argument("--phase3_data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200,
                   help="200 epochs suffisent (shared_layer converge vite)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="LR plus bas (GNN pré-entraîné, seulement fusion à apprendre)")
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smooth", type=float, default=0.05)
    p.add_argument("--oversample_ratio", type=float, default=3.0)
    p.add_argument("--save_dir", type=str,
                   default=str(CHECKPOINT_DIR / "phase3_hybrid"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--eval_only", action="store_true",
                   help="Évaluation seule sur checkpoint existant")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint spécifique pour --eval_only")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # ── Data ──
    data_dir = args.phase3_data_dir or str(EXTERNAL_DIR / "phase3")
    data = load_hiv_data(data_dir)
    hiv_tr, hiv_va, hiv_te = data["train"], data["val"], data["test"]
    lbl = data["label_col"]
    print(f"HIV train/val/test: {len(hiv_tr)}/{len(hiv_va)}/{len(hiv_te)}")
    print(f"Actifs train: {int((hiv_tr[lbl] == 1).sum())}/{len(hiv_tr)} "
          f"({100*(hiv_tr[lbl]==1).sum()/len(hiv_tr):.2f}%)")

    # ── Pré-calcul Morgan FP ──
    print("\n▶ Pré-calcul Morgan FP (ECFP4, 2048-bit)")
    morgan_train = precompute_morgan(hiv_tr["smiles"].tolist())
    morgan_val = precompute_morgan(hiv_va["smiles"].tolist())
    morgan_test = precompute_morgan(hiv_te["smiles"].tolist())

    # ── Modèle hybrid ──
    ckpt_path = args.checkpoint or args.base_checkpoint or str(
        CHECKPOINT_DIR / "phase3_improved" / "best_val.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint absent: {ckpt_path}")
        sys.exit(1)
    print(f"\n▶ Chargement checkpoint: {ckpt_path}")
    model, use_chirality = load_hybrid_model(ckpt_path, device)

    if args.eval_only:
        print("\n▶ Mode évaluation seule")
        _full_eval(model, hiv_te, lbl, morgan_test, device, save_dir,
                   hiv_va, morgan_val, hiv_tr)
        return

    # ── Datasets ──
    train_ds = HybridSmilesDataset(
        hiv_tr["smiles"].tolist(), hiv_tr[lbl].tolist(), morgan_train, train_mode=True,
    )
    val_ds = HybridSmilesDataset(
        hiv_va["smiles"].tolist(), hiv_va[lbl].tolist(), morgan_val, train_mode=False,
    )

    sampler = make_weighted_sampler(hiv_tr[lbl].tolist(),
                                     oversample_ratio=args.oversample_ratio)
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=sampler, collate_fn=_collate, num_workers=0)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=_collate)

    # ── Loss ──
    focal = FocalBCELoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    smooth = BCEWithLabelSmoothing(epsilon=args.label_smooth)

    # ── Optimizer : fine-tune shared_layer (fusion) + efficacy_head + convs.4/5 ──
    for n, p_ in model.named_parameters():
        p_.requires_grad = ("efficacy_head" in n or "shared_layer" in n or
                            "convs.4" in n or "convs.5" in n or "projection" in n)
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_train = sum(p_.numel() for p_ in trainable)
    n_total = sum(p_.numel() for p_ in model.parameters())
    print(f"\n▶ Params: {n_train:,} trainables / {n_total:,} total ({100*n_train/n_total:.1f}%)")
    print(f"  Focal(α={args.focal_alpha}, γ={args.focal_gamma}) + LabelSmoothing(ε={args.label_smooth})")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineRestartScheduler(
        optimizer, cycle_epochs=args.epochs, n_cycles=1,
        lr_max=args.lr, lr_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    from sklearn.metrics import roc_auc_score

    best_val_auc = -1.0
    start_epoch = 0

    # ── Resume ──
    resume_path = save_dir / "resume_state.pth"
    if args.resume and resume_path.exists():
        print(f"\n▶ Reprise depuis {resume_path}")
        rs = safe_load_checkpoint(resume_path, map_location=str(device), trusted=True)
        model.load_state_dict(rs["model_state_dict"], strict=False)
        optimizer.load_state_dict(rs["optimizer_state_dict"])
        scaler.load_state_dict(rs["scaler_state_dict"])
        start_epoch = int(rs["epoch"]) + 1
        best_val_auc = float(rs["best_val_auc"])
        try:
            torch.set_rng_state(rs["torch_rng_state"].cpu().to(torch.uint8))
            np.random.set_state(rs["numpy_rng_state"])
            if torch.cuda.is_available() and rs.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(rs["cuda_rng_state"].cpu().to(torch.uint8))
        except Exception as e:
            print(f"  ⚠ RNG non restauré ({e})")
        print(f"  → reprise epoch {start_epoch}/{args.epochs} | best_val={best_val_auc:.4f}")
    elif args.resume:
        print(f"\n▶ --resume mais {resume_path} absent → départ de zéro")

    # ── Training ──
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        model.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train e{epoch+1}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                preds = model(batch)
                logits = preds["efficacy"].squeeze(-1)
                y = batch.y.squeeze(-1)
                loss = 0.6 * focal(logits, y) + 0.4 * smooth(logits, y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch.y.size(0)
            n += batch.y.size(0)

        # ── Eval val ──
        model.eval()
        val_logits, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch)
                val_logits.append(preds["efficacy"].squeeze(-1).cpu().numpy())
                val_true.append(batch.y.squeeze(-1).cpu().numpy())
        val_logits = np.concatenate(val_logits)
        val_true = np.concatenate(val_true)
        val_proba = 1.0 / (1.0 + np.exp(-val_logits))
        try:
            val_auc = roc_auc_score(val_true, val_proba)
        except ValueError:
            val_auc = 0.5

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:4d}/{args.epochs} | t_loss={total_loss/max(1,n):.4f} | "
              f"val_auc={val_auc:.4f} | lr={lr:.2e}")

        # ── Best val ──
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            _atomic_save({
                "model_state_dict": {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                "config": {"use_chirality": use_chirality,
                           "best_val_auc": val_auc, "epoch": epoch,
                           "morgan_dim": MORGAN_DIM, "morgan_radius": MORGAN_RADIUS},
            }, save_dir / "best_val.pth")

        # ── Resume checkpoint ──
        if ((epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1):
            _atomic_save({
                "model_state_dict": {k: v.cpu().clone()
                                     for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "cuda_rng_state": (torch.cuda.get_rng_state()
                                   if torch.cuda.is_available() else None),
            }, resume_path)

    duration = time.time() - t0
    print(f"\n✅ Training terminé en {duration/60:.1f}min")
    print(f"✅ Best val AUC: {best_val_auc:.4f}")

    # ── Recharger best_val pour évaluation ──
    print("\n▶ Rechargement best_val.pth pour évaluation finale")
    best_ckpt = safe_load_checkpoint(save_dir / "best_val.pth", trusted=True)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=False)
    model.eval()

    _full_eval(model, hiv_te, lbl, morgan_test, device, save_dir,
               hiv_va, morgan_val, hiv_tr)


# ═══════════════════════════════════════════════════════════════════════════
# Évaluation complète : AUC + pharma + DrugBank + calibration Platt
# ═══════════════════════════════════════════════════════════════════════════

def _full_eval(model, hiv_test, lbl_col, morgan_test, device, save_dir,
               hiv_val=None, morgan_val=None, hiv_train=None):
    """Évaluation complète du modèle hybrid."""
    from sklearn.metrics import roc_auc_score
    from src.validation.pharma_metrics import full_screening_report

    model.eval()

    # ── Prédictions test ──
    print("\n═══ Prédictions HIV test ═══")
    scores, y_true, valid_smiles = [], [], []
    test_smiles = hiv_test["smiles"].tolist()
    test_labels = hiv_test[lbl_col].tolist()

    for i in tqdm(range(len(test_smiles)), desc="Test eval"):
        smi = test_smiles[i]
        try:
            g = smiles_to_graph(smi)
            if g is None:
                continue
            g.morgan_fp = torch.tensor(morgan_test[i], dtype=torch.float32).unsqueeze(0)
            batch = Batch.from_data_list([g]).to(device)
            with torch.no_grad():
                logit = model(batch)["efficacy"].cpu().numpy().ravel()[0]
            scores.append(1.0 / (1.0 + np.exp(-logit)))
            y_true.append(int(test_labels[i]))
            valid_smiles.append(smi)
        except Exception:
            continue

    scores = np.array(scores)
    y_true = np.array(y_true)

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = None
    print(f"  AUC test : {auc:.4f}" if auc else "  AUC indéfini")

    # ── Métriques pharma ──
    print("\n═══ Métriques pharma ═══")
    train_actives = (hiv_train[hiv_train[lbl_col] == 1]["smiles"].tolist()
                     if hiv_train is not None else None)
    pharma = full_screening_report(y_true, scores,
                                    smiles=valid_smiles,
                                    training_actives=train_actives)
    for k, v in pharma.items():
        if isinstance(v, (int, float, type(None))):
            print(f"  {k}: {v}")

    # ── DrugBank ──
    print("\n═══ DrugBank ARV ═══")
    try:
        from src.validation.drugbank_antiretroviral import (
            canonicalize_arv_smiles, validate_model_recall,
        )
        arvs = canonicalize_arv_smiles()
        arv_smi = [d["smiles_canonical"] for d in arvs]
        arv_morgan, arv_mask = smiles_to_morgan_array(arv_smi)

        arv_scores = []
        arv_valid_smi = []
        for j, d in enumerate(arvs):
            try:
                g = smiles_to_graph(d["smiles_canonical"])
                if g is None or not arv_mask[j]:
                    continue
                g.morgan_fp = torch.tensor(arv_morgan[j], dtype=torch.float32).unsqueeze(0)
                batch = Batch.from_data_list([g]).to(device)
                with torch.no_grad():
                    logit = model(batch)["efficacy"].cpu().numpy().ravel()[0]
                arv_scores.append(1.0 / (1.0 + np.exp(-logit)))
                arv_valid_smi.append(d["smiles_canonical"])
            except Exception:
                continue

        arv_scores = np.array(arv_scores)
        print(f"  {len(arv_valid_smi)}/{len(arvs)} ARV prédits")
        print(f"  Score moyen ARV: {arv_scores.mean():.3f}")

        combined = pd.DataFrame({
            "smiles": list(arv_valid_smi) + valid_smiles,
            "hiv_score": np.concatenate([arv_scores, scores]),
        })
        gate = validate_model_recall(combined, top_pct=0.10, min_recall=25)
        print(f"  Verdict: {gate['verdict']}")
        print(f"  Recall: {gate['recall']}/{gate['total']} ({gate['recall_rate']*100:.1f}%)")
    except Exception as e:
        gate = {"error": str(e)}
        print(f"  ⚠ DrugBank échoué: {e}")

    # ── Calibration Platt (si val disponible) ──
    platt_info = {}
    if hiv_val is not None and morgan_val is not None:
        print("\n═══ Calibration Platt (sur val) ═══")
        try:
            from sklearn.linear_model import LogisticRegression as LR

            # Prédictions val
            val_scores, val_true = [], []
            val_smiles = hiv_val["smiles"].tolist()
            val_labels = hiv_val[lbl_col].tolist()
            for i in range(len(val_smiles)):
                try:
                    g = smiles_to_graph(val_smiles[i])
                    if g is None:
                        continue
                    g.morgan_fp = torch.tensor(morgan_val[i], dtype=torch.float32).unsqueeze(0)
                    batch = Batch.from_data_list([g]).to(device)
                    with torch.no_grad():
                        logit = model(batch)["efficacy"].cpu().numpy().ravel()[0]
                    val_scores.append(1.0 / (1.0 + np.exp(-logit)))
                    val_true.append(int(val_labels[i]))
                except Exception:
                    continue

            val_scores = np.array(val_scores).reshape(-1, 1)
            val_true = np.array(val_true)

            # Platt scaling = LR sur les scores bruts
            platt = LR(solver="lbfgs", max_iter=1000)
            platt.fit(val_scores, val_true)

            # Appliquer au test
            scores_calibrated = platt.predict_proba(scores.reshape(-1, 1))[:, 1]
            auc_cal = roc_auc_score(y_true, scores_calibrated)
            print(f"  AUC après Platt: {auc_cal:.4f} (avant: {auc:.4f})")

            # Re-pharma calibré
            pharma_cal = full_screening_report(y_true, scores_calibrated,
                                                smiles=valid_smiles,
                                                training_actives=train_actives)
            print(f"  EF@1% calibré: {pharma_cal.get('ef_1pct', 'N/A')}")
            print(f"  BEDROC calibré: {pharma_cal.get('bedroc_alpha20', 'N/A')}")

            platt_info = {
                "auc_calibrated": float(auc_cal),
                "platt_coef": float(platt.coef_[0][0]),
                "platt_intercept": float(platt.intercept_[0]),
                "pharma_calibrated": pharma_cal,
            }
        except Exception as e:
            print(f"  ⚠ Calibration Platt échouée: {e}")
            platt_info = {"error": str(e)}

    # ── Sauvegarde rapport ──
    report = {
        "model": "GNN_hybrid_Morgan2048",
        "auc_test": float(auc) if auc else None,
        "pharma_test": pharma,
        "drugbank": gate if isinstance(gate, dict) else {"error": str(gate)},
        "platt_calibration": platt_info,
        "n_test": len(y_true),
        "n_actives": int(y_true.sum()),
    }
    out = save_dir / "hybrid_eval_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f"\n✅ Rapport: {out}")


if __name__ == "__main__":
    main()
