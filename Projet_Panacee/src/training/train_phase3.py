"""
Phase 3 - Entraînement multi-propriétés + IA raisonnement.

Pipeline complet :
  1. Charger l'encodeur pré-entraîné Phase 2 (best_toxicity_model.pth)
  2. Construire le MultiPropertyPredictor (N têtes)
  3. Entraîner sur les datasets fusionnés (Tox21, ESOL, Lipo, BBBP, ClinTox, HIV)
  4. Entraîner le MolecularReasoner pour l'analyse combinatoire
  5. Sauvegarder le modèle complet : panacee_phase3_complete.pth
"""
import sys
import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyPredictor, MultiPropertyLoss
from src.models.reasoner import MolecularReasoner, ReasonerLoss
from src.preprocessing.multi_property_loader import (
    MultiPropertyDataset, collate_multi_property,
    download_all_phase3_data, merge_datasets,
)
from src.config import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
    HIDDEN_DIM, NUM_GNN_LAYERS, OUTPUT_DIM, DROPOUT,
    CONV_TYPE, ATTENTION_HEADS,
    PHASE3, CHECKPOINT_DIR, LOG_DIR, EXTERNAL_DIR,
)
from src.utils.gpu_manager import get_gpu_manager
from src.utils.error_handler import (
    setup_logging, HealthMonitor, emergency_save,
)


# ======================================================================
# Scheduler warmup + cosine
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-7):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            alpha = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            alpha = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.lr_min + (blr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ======================================================================
# Métriques
# ======================================================================

def compute_phase3_metrics(predictions, targets):
    """
    Calcule les métriques pour toutes les propriétés.

    Returns:
        dict {property: {metric: value}}
    """
    metrics = {}

    for prop_name in predictions:
        if prop_name not in targets:
            continue

        pred = predictions[prop_name].detach().cpu()
        tgt = targets[prop_name].detach().cpu().numpy()
        valid = ~np.isnan(tgt)

        if valid.sum() == 0:
            continue

        prop_metrics = {}

        if prop_name in ("toxicity", "bioavailability", "metabolic_stability"):
            # Classification : AUC + F1
            probs = torch.sigmoid(pred).numpy()
            aucs = []
            f1s = []
            for t in range(tgt.shape[1]):
                mask = valid[:, t]
                if mask.sum() < 2:
                    continue
                y = tgt[mask, t]
                p = probs[mask, t]
                if len(np.unique(y)) > 1:
                    aucs.append(roc_auc_score(y, p))
                f1s.append(f1_score(y, (p > 0.5).astype(int), zero_division=0))

            prop_metrics["roc_auc"] = float(np.mean(aucs)) if aucs else 0.0
            prop_metrics["f1"] = float(np.mean(f1s)) if f1s else 0.0

        elif prop_name in ("efficacy",):
            # Classification binaire
            probs = torch.sigmoid(pred).numpy()
            y = tgt[valid].flatten()
            p = probs[valid[:, 0] if valid.ndim > 1 else valid].flatten()
            if len(y) > 0 and len(np.unique(y)) > 1:
                prop_metrics["roc_auc"] = float(roc_auc_score(y, p))
                prop_metrics["f1"] = float(f1_score(y, (p > 0.5).astype(int), zero_division=0))
            else:
                prop_metrics["roc_auc"] = 0.0
                prop_metrics["f1"] = 0.0

        else:
            # Régression : RMSE + R²
            pred_np = pred.numpy()
            pred_valid = pred_np[valid]
            tgt_valid = tgt[valid]
            rmse = float(np.sqrt(np.mean((pred_valid - tgt_valid) ** 2)))
            ss_res = np.sum((tgt_valid - pred_valid) ** 2)
            ss_tot = np.sum((tgt_valid - tgt_valid.mean()) ** 2)
            r2 = float(1 - ss_res / max(ss_tot, 1e-8))
            prop_metrics["rmse"] = rmse
            prop_metrics["r2"] = r2

        metrics[prop_name] = prop_metrics

    return metrics


# ======================================================================
# Boucles train / eval
# ======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch):
    model.train()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bioavailability", "metabolic_stability"]}
    all_targets = {k: [] for k in all_preds}

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch_data, labels in pbar:
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()
        predictions = model(batch_data)

        loss, loss_details = criterion(predictions, labels_device)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_data.num_graphs
        n += batch_data.num_graphs

        for k in predictions:
            if k in labels_device:
                all_preds[k].append(predictions[k].detach())
                all_targets[k].append(labels_device[k].detach())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Agréger les prédictions
    merged_preds = {}
    merged_targets = {}
    for k in all_preds:
        if all_preds[k]:
            merged_preds[k] = torch.cat(all_preds[k])
            merged_targets[k] = torch.cat(all_targets[k])

    metrics = compute_phase3_metrics(merged_preds, merged_targets)
    metrics["_loss"] = total_loss / max(n, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds = {k: [] for k in ["toxicity", "efficacy", "solubility",
                                   "lipophilicity", "bioavailability", "metabolic_stability"]}
    all_targets = {k: [] for k in all_preds}

    for batch_data, labels in loader:
        batch_data = batch_data.to(device)
        labels_device = {k: v.to(device) for k, v in labels.items()}

        predictions = model(batch_data)
        loss, _ = criterion(predictions, labels_device)

        if not torch.isnan(loss):
            total_loss += loss.item() * batch_data.num_graphs
            n += batch_data.num_graphs

        for k in predictions:
            if k in labels_device:
                all_preds[k].append(predictions[k].detach())
                all_targets[k].append(labels_device[k].detach())

    merged_preds = {}
    merged_targets = {}
    for k in all_preds:
        if all_preds[k]:
            merged_preds[k] = torch.cat(all_preds[k])
            merged_targets[k] = torch.cat(all_targets[k])

    metrics = compute_phase3_metrics(merged_preds, merged_targets)
    metrics["_loss"] = total_loss / max(n, 1)
    return metrics, merged_preds, merged_targets


# ======================================================================
# Fonction principale
# ======================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 3 - Multi-propriétés + IA Raisonnement")
    p.add_argument("--download", action="store_true", help="Télécharge tous les datasets")
    p.add_argument("--data_dir", type=str, default=None, help="Dossier des données Phase 3")
    p.add_argument("--pretrained_model", type=str, default=None,
                   help="Checkpoint Phase 2 (best_toxicity_model.pth)")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase3"))
    p.add_argument("--epochs", type=int, default=PHASE3["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE3["batch_size"])
    p.add_argument("--patience", type=int, default=PHASE3["patience"])
    p.add_argument("--device", type=str, default=str(DEVICE))
    args = p.parse_args()

    # Initialisation GPU et logging
    gpu = get_gpu_manager(force_cpu=(args.device == "cpu"))
    device = gpu.device
    logger = setup_logging(name="panacee.phase3")
    health = HealthMonitor(check_interval=50)

    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = LOG_DIR / "phase3"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Étape 0 : Vérifications ──────────────────────────────────────
    print("=" * 80)
    print("  PHASE 3 - MULTI-PROPRIÉTÉS + IA RAISONNEMENT")
    print("=" * 80)
    gpu.print_summary()

    # Chercher le modèle Phase 2
    pretrained = args.pretrained_model or str(CHECKPOINT_DIR / "phase2" / "best_toxicity_model.pth")
    if not os.path.exists(pretrained):
        print(f"❌ Modèle Phase 2 introuvable : {pretrained}")
        print("   Lancez d'abord Phase 2 : python run_phase2.py --download")
        sys.exit(1)
    print(f"✓ Modèle Phase 2 : {pretrained}")

    # ── Étape 1 : Télécharger / charger les données ──────────────────
    data_dir = args.data_dir or str(EXTERNAL_DIR / "phase3")

    if args.download:
        print("\n📥 Téléchargement de tous les datasets...")
        data_paths = download_all_phase3_data(data_dir)

        # Sauvegarder les chemins pour utilisation future
        paths_file = os.path.join(data_dir, "data_paths.json")
        serializable_paths = {}
        for k, v in data_paths.items():
            if v is not None:
                serializable_paths[k] = v
        with open(paths_file, "w") as f:
            json.dump(serializable_paths, f, indent=2)
        print(f"✓ Chemins sauvegardés : {paths_file}")
    else:
        paths_file = os.path.join(data_dir, "data_paths.json")
        if not os.path.exists(paths_file):
            print("❌ Pas de données. Utilisez --download pour télécharger")
            sys.exit(1)
        with open(paths_file, "r") as f:
            data_paths = json.load(f)
        print(f"✓ Données chargées depuis {paths_file}")

    # ── Étape 2 : Fusionner les datasets ─────────────────────────────
    print("\n🔀 Fusion des datasets...")
    try:
        train_df = merge_datasets(data_paths, "train")
        val_df = merge_datasets(data_paths, "val")
    except Exception as e:
        print(f"❌ Erreur fusion : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Étape 3 : Créer les datasets PyTorch ─────────────────────────
    print("\n📊 Construction des datasets...")
    try:
        train_ds = MultiPropertyDataset(train_df)
        val_ds = MultiPropertyDataset(val_df)
    except Exception as e:
        print(f"❌ Erreur dataset : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"  Train : {len(train_ds)} molécules")
    print(f"  Val   : {len(val_ds)} molécules")

    # Ajuster batch_size selon la VRAM disponible
    batch_size = gpu.optimize_batch_size(args.batch_size, model_size_mb=50)
    print(f"  Batch size : {batch_size} (demandé: {args.batch_size})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_multi_property,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=collate_multi_property,
    )

    # ── Étape 4 : Construire le modèle ───────────────────────────────
    print("\n🏗️ Construction du modèle Phase 3...")

    # Encodeur de base
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM, dropout=DROPOUT,
        conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS,
    )

    # Charger les poids Phase 2
    ckpt = torch.load(pretrained, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        if encoder_sd:
            encoder.load_state_dict(encoder_sd, strict=False)
            print("  ✓ Encodeur Phase 2 chargé")
        else:
            print("  ⚠ Pas de poids encoder dans le checkpoint Phase 2")
    else:
        print("  ⚠ Format checkpoint non reconnu")

    # Modèle multi-propriétés
    model = MultiPropertyPredictor(
        encoder=encoder, hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT, freeze_encoder=True,
    ).to(device)

    # Raisonneur IA (sera entraîné après le multi-propriétés)
    reasoner = MolecularReasoner(
        mol_dim=OUTPUT_DIM,
        hidden_dim=PHASE3["reasoner_hidden_dim"],
        num_heads=PHASE3["reasoner_num_heads"],
        num_layers=PHASE3["reasoner_num_layers"],
        max_molecules=PHASE3["max_molecules_combo"],
        num_dose_levels=len(PHASE3["dose_levels"]),
        dropout=PHASE3["reasoner_dropout"],
    ).to(device)

    n_params_model = sum(pp.numel() for pp in model.parameters())
    n_params_reasoner = sum(pp.numel() for pp in reasoner.parameters())
    n_train_model = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  MultiPropertyPredictor : {n_params_model:,} params ({n_train_model:,} entraînables)")
    print(f"  MolecularReasoner      : {n_params_reasoner:,} params")
    print(f"  Total                  : {n_params_model + n_params_reasoner:,} params")

    # ── Étape 5 : Loss et Optimiseur ─────────────────────────────────
    tox_pos_weight = train_ds.get_pos_weight("toxicity")
    if tox_pos_weight is not None:
        tox_pos_weight = tox_pos_weight.to(device)
    criterion = MultiPropertyLoss(tox_pos_weight=tox_pos_weight).to(device)

    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": PHASE3["lr_encoder"]},
        {"params": model.shared_layer.parameters(), "lr": PHASE3["lr_heads"]},
        {"params": model.toxicity_head.parameters(), "lr": PHASE3["lr_heads"]},
        {"params": model.efficacy_head.parameters(), "lr": PHASE3["lr_heads"]},
        {"params": model.solubility_head.parameters(), "lr": PHASE3["lr_heads"]},
        {"params": model.lipophilicity_head.parameters(), "lr": PHASE3["lr_heads"]},
        {"params": model.bioavailability_head.parameters(), "lr": PHASE3["lr_heads"]},
        {"params": model.metabolic_stability_head.parameters(), "lr": PHASE3["lr_heads"]},
    ], weight_decay=PHASE3["weight_decay"])

    scheduler = WarmupCosineScheduler(
        optimizer, PHASE3["warmup_epochs"], args.epochs, PHASE3["lr_min"],
    )

    # ── Étape 6 : Boucle d'entraînement ──────────────────────────────
    best_score = 0.0
    no_improve = 0
    history = []
    best_path = Path(args.save_dir) / PHASE3["checkpoint_name"]
    latest_path = Path(args.save_dir) / "checkpoint_latest.pth"
    freeze_epochs = PHASE3["freeze_encoder_epochs"]

    print(f"\n{'='*80}")
    print("▶️ ENTRAÎNEMENT MULTI-PROPRIÉTÉS")
    print(f"  {args.epochs} epochs, freeze_encoder={freeze_epochs}, patience={args.patience}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Dégel progressif de l'encodeur
        if epoch <= freeze_epochs:
            model._freeze_encoder()
        else:
            model.gradual_unfreeze(epoch, freeze_epochs)

        scheduler.step(epoch - 1)
        lrs = scheduler.get_last_lr()

        # Entraînement
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            PHASE3["grad_clip"], epoch,
        )

        # Validation
        val_m, _, _ = evaluate(model, val_loader, criterion, device)

        # Score composite (moyenne pondérée des métriques)
        score = 0.0
        n_scores = 0
        for prop, prop_m in val_m.items():
            if prop.startswith("_"):
                continue
            if isinstance(prop_m, dict):
                if "roc_auc" in prop_m:
                    score += prop_m["roc_auc"]
                    n_scores += 1
                elif "r2" in prop_m:
                    score += max(prop_m["r2"], 0)
                    n_scores += 1
        composite_score = score / max(n_scores, 1)

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))

        # Affichage
        train_loss = train_m.get("_loss", 0)
        val_loss = val_m.get("_loss", 0)

        prop_summary = []
        for prop in ["toxicity", "efficacy", "solubility", "lipophilicity",
                      "bioavailability", "metabolic_stability"]:
            if prop in val_m and isinstance(val_m[prop], dict):
                m = val_m[prop]
                if "roc_auc" in m:
                    prop_summary.append(f"{prop[:4]}={m['roc_auc']:.3f}")
                elif "r2" in m:
                    prop_summary.append(f"{prop[:4]}={m['r2']:.3f}")

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"t_loss={train_loss:.4f} v_loss={val_loss:.4f} | "
            f"score={composite_score:.4f} | "
            f"{' '.join(prop_summary)} | "
            f"lr={lrs[0]:.1e} | ETA {eta}"
        )

        # Historique
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "composite_score": composite_score,
            "lr": lrs[0],
        }
        for prop in val_m:
            if not prop.startswith("_") and isinstance(val_m[prop], dict):
                history_entry[f"val_{prop}"] = val_m[prop]
        history.append(history_entry)

        # Checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": {k: v for k, v in val_m.items() if not k.startswith("_")},
            "composite_score": composite_score,
            "best_score": best_score,
            "history": history,
            "property_dims": train_ds.get_property_dims(),
            "config": {
                "hidden_dim": HIDDEN_DIM,
                "num_gnn_layers": NUM_GNN_LAYERS,
                "output_dim": OUTPUT_DIM,
                "dropout": DROPOUT,
                "atom_feature_dim": ATOM_FEATURE_DIM,
                "bond_feature_dim": BOND_FEATURE_DIM,
                "conv_type": CONV_TYPE,
                "attention_heads": ATTENTION_HEADS,
            },
        }
        torch.save(ckpt_data, latest_path)

        if composite_score > best_score:
            best_score = composite_score
            no_improve = 0
            torch.save(ckpt_data, best_path)
            print(f"  → Nouveau meilleur score={best_score:.4f} sauvegardé !")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping (patience={args.patience})")
                break

        # Monitoring de santé
        if not health.step(val_loss):
            print("  ALERTE SANTÉ : trop de NaN, sauvegarde d'urgence...")
            emergency_save(
                model, optimizer, epoch,
                str(Path(args.save_dir) / "emergency_checkpoint.pth"),
            )
            break

        gpu.clear_memory()

    # ── Étape 7 : Entraîner le Raisonneur IA ─────────────────────────
    print(f"\n{'='*80}")
    print("🧠 ENTRAÎNEMENT DU MODULE IA DE RAISONNEMENT")
    print(f"{'='*80}\n")

    # Charger le meilleur modèle multi-propriétés
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Pré-calculer les embeddings + une "qualité pharmacologique" par molécule.
    # qualité = (1 - toxicité moyenne prédite) pondérée par l'efficacité prédite.
    # → sert de signal d'auto-supervision plus pertinent que la pure diversité.
    print("  Calcul des embeddings + qualité prédite...")
    all_embeddings = []
    all_quality = []
    with torch.no_grad():
        for batch_data, _ in tqdm(train_loader, desc="  Embeddings"):
            batch_data = batch_data.to(device)
            emb = model.encode(batch_data)
            preds = model(batch_data)

            tox = torch.sigmoid(preds["toxicity"]).mean(dim=1)        # [B] tox moyenne
            quality = 1.0 - tox                                        # moins toxique = mieux
            if "efficacy" in preds:
                eff = torch.sigmoid(preds["efficacy"]).view(-1)        # [B]
                quality = quality * (0.5 + 0.5 * eff)                  # pondère par l'efficacité

            all_embeddings.append(emb.cpu())
            all_quality.append(quality.cpu())
    all_embeddings = torch.cat(all_embeddings)
    all_quality = torch.cat(all_quality).clamp(0.0, 1.0)
    print(f"  {all_embeddings.shape[0]} embeddings calculés, dim={all_embeddings.shape[1]}")

    # Entraîner le raisonneur sur des combinaisons aléatoires
    reasoner_optimizer = AdamW(reasoner.parameters(), lr=PHASE3["lr_reasoner"])
    reasoner_criterion = ReasonerLoss()

    max_combo = PHASE3["max_molecules_combo"]
    n_combos_per_epoch = min(len(all_embeddings) // max_combo, 500)

    print(f"  {n_combos_per_epoch} combinaisons par epoch, max {max_combo} molécules")

    reasoner_epochs = 30
    for r_epoch in range(1, reasoner_epochs + 1):
        reasoner.train()
        total_r_loss = 0.0

        for _ in range(n_combos_per_epoch):
            # Échantillonner N molécules aléatoires
            n_mols = torch.randint(2, max_combo + 1, (1,)).item()
            indices = torch.randperm(len(all_embeddings))[:n_mols]
            combo_emb = all_embeddings[indices].unsqueeze(0).to(device)  # [1, N, D]

            # Padding si nécessaire
            if n_mols < max_combo:
                pad = torch.zeros(1, max_combo - n_mols, combo_emb.shape[-1], device=device)
                combo_emb = torch.cat([combo_emb, pad], dim=1)
                mask = torch.zeros(1, max_combo, dtype=torch.bool, device=device)
                mask[0, n_mols:] = True
            else:
                mask = None

            reasoner_optimizer.zero_grad()
            output = reasoner(combo_emb, mask)

            # Auto-supervision (proxy de synergie, en attendant des données
            # réelles type DrugComb) : combine la QUALITÉ pharmacologique prédite
            # des molécules (faible toxicité / bonne efficacité) et leur DIVERSITÉ
            # structurelle. Une bonne combinaison = molécules de qualité + complémentaires.
            quality = all_quality[indices].mean().item()
            diversity = torch.pdist(combo_emb[0, :n_mols]).mean().item()
            success = 0.7 * quality + 0.3 * min(diversity / 5.0, 1.0)
            success_target = torch.tensor([[success]], device=device)

            targets = {"success_labels": success_target}
            loss, _ = reasoner_criterion(output, targets)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
                reasoner_optimizer.step()
                total_r_loss += loss.item()

        avg_loss = total_r_loss / max(n_combos_per_epoch, 1)
        print(f"  Reasoner Epoch {r_epoch:2d}/{reasoner_epochs} | loss={avg_loss:.4f}")

    # ── Étape 8 : Sauvegarder le modèle complet ──────────────────────
    print(f"\n{'='*80}")
    print("💾 SAUVEGARDE DU MODÈLE COMPLET")
    print(f"{'='*80}")

    # Recharger le meilleur checkpoint et ajouter le raisonneur
    best_ckpt["reasoner_state_dict"] = reasoner.state_dict()
    best_ckpt["reasoner_config"] = {
        "hidden_dim": PHASE3["reasoner_hidden_dim"],
        "num_heads": PHASE3["reasoner_num_heads"],
        "num_layers": PHASE3["reasoner_num_layers"],
        "max_molecules": PHASE3["max_molecules_combo"],
        "num_dose_levels": len(PHASE3["dose_levels"]),
        "dose_levels": PHASE3["dose_levels"],
        "dropout": PHASE3["reasoner_dropout"],
    }
    torch.save(best_ckpt, best_path)

    total_time = time.time() - t0
    print(f"\n✅ Phase 3 terminée en {timedelta(seconds=int(total_time))}")
    print(f"   Meilleur score composite = {best_score:.4f}")
    print(f"   Modèle sauvegardé : {best_path}")

    # Sauvegarder l'historique
    log_path = log_dir / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"   Log : {log_path}")


if __name__ == "__main__":
    main()
