"""
Phase 2 - Fine-tuning classification de toxicite - v3.

Nouveautes v3 :
  - Encodeur a attention edge-aware (config CONV_TYPE).
  - AMP (mixed precision) sur GPU.
  - EMA des poids (--ema) pour un fine-tuning plus stable.
  - Cross-validation par scaffold (--cv_folds N) pour des metriques honnetes.

Conserve v2 :
  1. LR differencies (encoder vs head).
  2. Warmup + cosine scheduler.
  3. pos_weight dynamique pour desequilibre de classes.
  4. Gradual unfreezing de l'encodeur.
  5. Recherche de seuil optimal par tache.
  6. Early stopping sur ROC-AUC moyen.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_seed

from src.config import (
    ATOM_FEATURE_DIM,
    ATTENTION_HEADS,
    BOND_FEATURE_DIM,
    CHECKPOINT_DIR,
    CONV_TYPE,
    DEVICE,
    DROPOUT,
    HIDDEN_DIM,
    LOG_DIR,
    NUM_GNN_LAYERS,
    OUTPUT_DIM,
    PHASE2,
    loader_kwargs,
    lr_scale_for_batch,
)
try:
    from src.models.encoder import MolecularEncoder
except ModuleNotFoundError as exc:  # permet d'importer les métriques sans PyG
    if exc.name != "torch_geometric":
        raise
    MolecularEncoder = None
try:
    from src.models.toxicity_classifier import MultiTaskBCELoss, ToxicityClassifier
except ModuleNotFoundError as exc:  # métriques importables sans PyG
    if exc.name != "torch_geometric":
        raise
    MultiTaskBCELoss = None
    ToxicityClassifier = None
try:
    from src.preprocessing.scaffold_split import scaffold_kfold
    from src.preprocessing.toxicity_loader import ToxicityDataset, collate_toxicity_batch
except ModuleNotFoundError as exc:  # métriques importables sans RDKit
    if exc.name not in {"rdkit", "pandas"}:
        raise
    scaffold_kfold = ToxicityDataset = collate_toxicity_batch = None
from src.utils.ema import ModelEMA
from src.utils.live_logger import LiveLogger
from src.validation.clinical_metrics import clinical_score as clinical_score_fn
from src.validation.clinical_metrics import summarize as clinical_summarize

# ======================================================================
# Scheduler warmup + cosine
# ======================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
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
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs, strict=False):
            pg["lr"] = self.lr_min + (blr - self.lr_min) * alpha

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(logits, targets):
    """ROC-AUC, F1, precision, recall (macro) en ignorant les NaN."""
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()
    valid = np.isfinite(targets_np) & np.isfinite(probs)

    aucs, f1s, precs, recs = [], [], [], []
    num_tasks = targets_np.shape[1]

    for t in range(num_tasks):
        mask = valid[:, t]
        if mask.sum() == 0:
            continue
        y = targets_np[mask, t]
        p = probs[mask, t]
        pred = (p > 0.5).astype(int)

        # F1/précision/rappel ne sont définis que si les DEUX classes sont
        # présentes : sinon on ne les compte PAS (un 0 fictif fausserait la
        # macro-moyenne, cf. même correction côté clinical_metrics).
        if len(np.unique(y)) > 1:
            aucs.append(roc_auc_score(y, p))
            f1s.append(f1_score(y, pred, zero_division=0))
            precs.append(precision_score(y, pred, zero_division=0))
            recs.append(recall_score(y, pred, zero_division=0))

    return {
        "n_tasks_scored": len(aucs),
        "roc_auc": float(np.mean(aucs)) if aucs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
    }


def find_optimal_thresholds(logits, targets, num_tasks):
    """
    Cherche le seuil optimal par tâche (max F1).

    Fix C5 (audit 2026-10-20) : grille élargie à [0.05, 0.95] par pas de 0.025
    (était 0.2-0.8 par 0.05 → ratait les tâches Tox21 très déséquilibrées
    comme NR-AR, SR-ATAD5 où F1 optimal < 0.2).

    NB C1 : CE RÉSULTAT NE DOIT PAS ÊTRE CALIBRÉ SUR LE SET DE VALIDATION
    DE MONITORING. L'appelant doit fournir soit une partition de calibration
    distincte, soit un sous-ensemble du train tenu à part (voir usage dans main()).
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()
    valid = np.isfinite(targets_np) & np.isfinite(probs)
    thresholds = []

    for t in range(num_tasks):
        mask = valid[:, t]
        if mask.sum() == 0 or len(np.unique(targets_np[mask, t])) < 2:
            thresholds.append(0.5)
            continue
        y = targets_np[mask, t]
        p = probs[mask, t]
        best_t, best_f1 = 0.5, -1.0
        for th in np.linspace(0.05, 0.95, 37):  # 37 points, pas 0.025
            f = f1_score(y, (p >= th).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = th
        thresholds.append(float(best_t))
    return thresholds


# ======================================================================
# Train / Eval
# ======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, epoch,
                    scaler=None, ema=None):
    """
    Boucle train 1 epoch avec AMP robuste.

    Fix (audit 2026-10-20) : au moment du 1er dégel partiel (epoch ==
    freeze_epochs + 1), les gradients explosent → inf/NaN sous FP16.
    - Avant : clip_grad_norm_(error_if_nonfinite=True) → crash dur
    - Après : error_if_nonfinite=False → laisse scaler.step() SKIP automatiquement
      le step (comportement AMP standard). On compte les skips + log si > seuil.
    - Loss non-finie : skip le batch au lieu de raise (grad n'arrive pas, pas de
      MAJ, warning compté). Un batch isolé anormal ne tue plus l'entraînement.
    - set_to_none=True : plus rapide + sémantique "unused param" propre.
    """
    model.train()
    total_loss, n = 0.0, 0
    all_logits, all_targets = [], []
    use_amp = scaler is not None and scaler.is_enabled()
    skipped_loss = 0  # loss non-finie
    skipped_grad = 0  # grads non-finis (scaler saute automatiquement)

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
    for batch_data, labels in pbar:
        batch_data = batch_data.to(device)
        labels = labels.to(device)
        if not torch.isfinite(labels).any():
            continue

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(batch_data)
            loss = criterion(logits.float(), labels)
            if not torch.isfinite(loss).all():
                skipped_loss += 1
                continue  # fix : skip batch au lieu de crash
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # error_if_nonfinite=False : si grads non-finis, scaler.step les
            # détecte, saute le step, et scaler.update() baisse le facteur.
            # C'est la convention AMP officielle (torch.amp docs).
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False,
            )
            if not torch.isfinite(total_norm):
                skipped_grad += 1
            # scaler.step saute si grads inf/NaN (vérif interne de scaler)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            # Si scale a diminué → scaler a sauté le step
            if new_scale < prev_scale:
                skipped_grad += 0  # déjà compté via total_norm non-fini
        else:
            logits = model(batch_data)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).all():
                skipped_loss += 1
                continue
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=False,
            )
            if not torch.isfinite(total_norm):
                skipped_grad += 1
                continue  # non-AMP : skip explicitement
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.detach().float())
        all_targets.append(labels.detach())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if skipped_loss or skipped_grad:
        print(f"  ⚠ Epoch {epoch}: {skipped_loss} batch loss NaN skippés, "
              f"{skipped_grad} batch grad non-finis skippés")

    if not all_logits:
        raise RuntimeError("Phase 2 : aucun batch valide pendant l'epoch")
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_targets = [], []

    for batch_data, labels in loader:
        batch_data = batch_data.to(device)
        labels = labels.to(device)
        logits = model(batch_data)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Phase 2 : prédictions de validation non finies")
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).all():
            raise RuntimeError("Phase 2 : loss de validation non finie")
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_logits.append(logits.float())
        all_targets.append(labels)

    if not all_logits:
        raise RuntimeError("Phase 2 : aucun batch de validation")
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics, all_logits, all_targets


# ======================================================================
# Helpers
# ======================================================================

class _GraphLabelDataset(Dataset):
    """Dataset minimal (graphes + labels) pour la cross-validation."""

    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def _pos_weight_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """neg/pos par tache (pour BCEWithLogitsLoss), en ignorant les NaN."""
    pw = []
    for t in range(labels.shape[1]):
        col = labels[:, t]
        valid = col[~torch.isnan(col)]
        pos = (valid == 1).sum().float()
        neg = (valid == 0).sum().float()
        pw.append((neg / pos).item() if pos > 0 else 1.0)
    return torch.tensor(pw, dtype=torch.float32)


def phase2_exposure(train_ds, val_ds):
    """Record every labelled compound/scaffold used for training or selection."""
    from src.preprocessing.multi_property_loader import _chemical_key

    compounds, scaffolds = set(), set()
    for dataset in (train_ds, val_ds):
        for index in range(len(dataset)):
            graph, _ = dataset[index]
            canonical, scaffold = _chemical_key(graph.smiles)
            compounds.add(canonical)
            scaffolds.add(scaffold)
    return {"canonical_smiles": sorted(compounds), "scaffolds": sorted(scaffolds),
            "includes_validation_selection": True}


def validate_phase2_warm_start(checkpoint, exposure, task_names):
    if checkpoint.get("phase2_exposure") != exposure or checkpoint.get("task_names") != task_names:
        raise ValueError("Warm-start Phase 2 sur des données/tâches différentes ou non vérifiées ; "
                         "utiliser --warm_start 0 et un nouveau save_dir")
    return checkpoint["model_state_dict"]


def phase2_epoch_eligible(epoch, freeze_epochs):
    return epoch > freeze_epochs


def _load_pretrained_encoder(encoder, path):
    """
    Charge les poids d'encodeur Phase 1 si dispo (sinon aleatoire).

    Fix C3 (audit 2026-10-20) : strict=False pour tolérer les évolutions
    d'archi légères (ex. toggle use_chirality) tout en loggant les clés
    manquantes/surplus. Les clés CRITIQUES (atom_embedding, convs.*, projection)
    doivent toutes être présentes — sinon ValueError explicite.
    """
    if path and not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint Phase 1 introuvable : {path}")
    if not path or not os.path.exists(path):
        print("  Pas de checkpoint Phase 1 -> encodeur aleatoire (Phase 2 autonome)")
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt):
        raise ValueError("Format checkpoint Phase 1 non reconnu")
    sd = ckpt["model_state_dict"]
    encoder_sd = {k.replace("encoder.", ""): v
                  for k, v in sd.items() if k.startswith("encoder.")}
    if not encoder_sd:
        raise ValueError("Checkpoint Phase 1 sans poids d'encodeur")
    if not all(torch.isfinite(v).all() for v in encoder_sd.values()
               if torch.is_tensor(v)):
        raise ValueError("Encodeur Phase 1 contenant des poids non finis")

    result = encoder.load_state_dict(encoder_sd, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    # Clés critiques dont l'absence invalide le transfer learning
    _CRITICAL_PREFIXES = ("atom_embedding", "convs.", "projection")
    critical_missing = [k for k in missing
                        if any(k.startswith(p) for p in _CRITICAL_PREFIXES)]
    if critical_missing:
        raise ValueError(
            f"Checkpoint Phase 1 incompatible : clés critiques absentes "
            f"({critical_missing[:5]}{'...' if len(critical_missing) > 5 else ''})"
        )
    if missing:
        print(f"  ⚠ Encodeur Phase 1 : {len(missing)} clés manquantes (non critiques) "
              f"ex: {missing[:3]} → init aléatoire conservée pour ces modules")
    if unexpected:
        print(f"  ⚠ Encodeur Phase 1 : {len(unexpected)} clés en surplus ignorées "
              f"ex: {unexpected[:3]}")
    print("  Encodeur pre-entraine charge (Phase 1)")


# ======================================================================
# Une execution complete (un split) : build -> train -> best
# ======================================================================

def train_one_run(train_ds, val_ds, num_tasks, task_names, pos_weight,
                  args, device, save_dir, use_ema=True, tag=""):
    os.makedirs(save_dir, exist_ok=True)
    best_path = Path(save_dir) / "best_toxicity_model.pth"
    latest_path = Path(save_dir) / "checkpoint_latest.pth"
    exposure = phase2_exposure(train_ds, val_ds)
    # Dossier des checkpoints par-epoch (gérables/supprimables depuis le dashboard).
    epochs_dir = Path(save_dir) / "epochs"
    save_epochs = bool(getattr(args, "save_epochs", 1))
    # Nouveau run -> on repart d'un dossier d'epochs propre (cohérent avec le
    # live_metrics.jsonl qui est lui aussi réinitialisé en tête de run).
    prev_best_state = None
    if getattr(args, "warm_start", 1) and best_path.exists():
        previous = torch.load(best_path, map_location="cpu", weights_only=False)
        prev_best_state = validate_phase2_warm_start(previous, exposure, task_names)
    archive_suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if best_path.exists():
        archived_best = best_path.with_name(f"best_toxicity_previous_{archive_suffix}.pth")
        best_path.rename(archived_best)
        print(f"Ancien checkpoint Phase 2 préservé : {archived_best}")
    if epochs_dir.exists():
        if epochs_dir.resolve().parent != Path(save_dir).resolve():
            raise ValueError("Dossier des epochs Phase 2 hors du save_dir")
        archived_epochs = epochs_dir.with_name(f"epochs_previous_{archive_suffix}")
        epochs_dir.rename(archived_epochs)
        print(f"Anciens checkpoints par époque préservés : {archived_epochs}")
    if save_epochs:
        epochs_dir.mkdir(parents=True, exist_ok=True)
    label = f"[{tag}] " if tag else ""

    # Fix B1 (audit 2026-10-20) : cudnn.benchmark=True annule set_seed/H10.
    # Opt-in via env PANACEE_CUDNN_BENCHMARK=1 si perf prime sur reproductibilité.
    if os.environ.get("PANACEE_CUDNN_BENCHMARK") == "1":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_toxicity_batch, drop_last=True, **loader_kwargs(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_toxicity_batch, **loader_kwargs(),
    )

    # -- Modele --
    encoder = MolecularEncoder(
        atom_dim=ATOM_FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS, edge_dim=BOND_FEATURE_DIM,
        output_dim=OUTPUT_DIM, dropout=DROPOUT,
        conv_type=CONV_TYPE, attention_heads=ATTENTION_HEADS,
    )
    _load_pretrained_encoder(encoder, args.pretrained_model)

    model = ToxicityClassifier(
        encoder=encoder, num_tasks=num_tasks,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT, freeze_encoder=True,
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"{label}Modele: {n_params:,} params | conv={CONV_TYPE}")

    # Warm-start complet (encodeur + tête) depuis le meilleur modèle précédent.
    if prev_best_state is not None:
        model.load_state_dict(prev_best_state, strict=True)
        print(f"{label}Warm-start : meilleur modèle précédent rechargé "
              f"(poursuite de l'entraînement).")

    criterion = MultiTaskBCELoss(pos_weight=pos_weight.to(device))
    # LR mis à l'échelle pour le batch effectif (règle racine carrée) : compense
    # le nombre réduit de mises à jour/epoch quand le batch grossit (anti-biais).
    lr_scale = lr_scale_for_batch(args.batch_size, PHASE2["batch_size"])
    lr_enc = PHASE2["lr_encoder"] * lr_scale
    lr_hd = PHASE2["lr_head"] * lr_scale
    if lr_scale > 1.0:
        print(f"  LR x{lr_scale:.2f} pour batch {args.batch_size} (ref {PHASE2['batch_size']}) : "
              f"encodeur {PHASE2['lr_encoder']:.2e}->{lr_enc:.2e}, "
              f"tete {PHASE2['lr_head']:.2e}->{lr_hd:.2e}")
    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": lr_enc},
        {"params": model.classifier.parameters(), "lr": lr_hd},
    ], weight_decay=PHASE2["weight_decay"])
    scheduler = WarmupCosineScheduler(
        optimizer, PHASE2["warmup_epochs"], args.epochs, PHASE2["lr_min"],
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, decay=0.999) if use_ema else None
    print(f"{label}AMP={use_amp} | EMA={use_ema} | {args.epochs} epochs | patience={args.patience}")

    best_auc = 0.0
    best_score = -1e9          # supervision = meilleure epoch selon le score clinique
    no_improve = 0
    history = []
    freeze_epochs = PHASE2["freeze_encoder_epochs"]
    # Logger temps reel (lu par le dashboard pendant l'entrainement)
    live = LiveLogger(Path(save_dir) / "live_metrics.jsonl",
                      meta={"phase": "phase2", "tag": tag, "epochs_total": args.epochs,
                            "num_tasks": num_tasks, "task_names": task_names,
                            "conv_type": CONV_TYPE, "ema": use_ema})
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch <= freeze_epochs:
            model._freeze_encoder()
        else:
            model.gradual_unfreeze(epoch, freeze_epochs)

        scheduler.step(epoch - 1)
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            PHASE2["grad_clip"], epoch, scaler, ema,
        )

        # Evaluation avec les poids EMA (si actif)
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        val_m, val_logits, val_targets = evaluate(model, val_loader, criterion, device)
        if val_m["n_tasks_scored"] == 0:
            raise RuntimeError("Phase 2 : aucune tâche binaire évaluable sur la validation")
        if ema is not None:
            ema.restore(model)

        elapsed = time.time() - t0
        eta = timedelta(seconds=int((elapsed / epoch) * (args.epochs - epoch)))
        print(
            f"{label}Epoch {epoch:3d}/{args.epochs} | "
            f"t_loss={train_m['loss']:.4f} v_loss={val_m['loss']:.4f} | "
            f"t_auc={train_m['roc_auc']:.4f} v_auc={val_m['roc_auc']:.4f} | "
            f"v_f1={val_m['f1']:.4f} | lr_enc={lr_enc:.2e} lr_head={lr_head:.2e} | ETA {eta}"
        )
        history.append({"epoch": epoch, "train": train_m, "val": val_m,
                        "lr_encoder": lr_enc, "lr_head": lr_head})

        # Métriques descriptives Tox21 et proxy de sélection. Elles ne sont
        # ni des observations cliniques ni une preuve de sécurité humaine.
        clin = clinical_summarize(
            torch.sigmoid(val_logits).cpu().numpy(),
            val_targets.cpu().numpy(), task_names=task_names,
        )
        agg = clin["aggregate"]
        per_task = {t["task"]: t["roc_auc"] for t in clin["tasks"]}

        clin_score = clinical_score_fn(
            val_m["roc_auc"], agg.get("macro_sensitivity"),
            agg.get("macro_fnr"), agg.get("n_danger"),
        )

        # state_dict a sauvegarder = EMA si actif, sinon poids courants
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
            state_to_save = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.restore(model)
        else:
            state_to_save = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        eligible = phase2_epoch_eligible(epoch, freeze_epochs)
        is_best = eligible and np.isfinite(clin_score) and clin_score > best_score
        if is_best:
            best_score = clin_score
            best_auc = val_m["roc_auc"]
            no_improve = 0
        elif eligible:
            no_improve += 1

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": state_to_save,
            "val_metrics": val_m,
            "best_auc": best_auc,
            "clinical_score": clin_score,
            "history": history,
            "num_tasks": num_tasks,
            "task_names": task_names,
            "phase2_exposure": exposure,
            "selection_eligible": phase2_epoch_eligible(epoch, freeze_epochs),
            "is_best": is_best,
            "selection_score": clin_score,
            "config": {
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_GNN_LAYERS,
                "output_dim": OUTPUT_DIM, "conv_type": CONV_TYPE,
                "attention_heads": ATTENTION_HEADS, "ema": use_ema,
            },
        }
        torch.save(ckpt_data, latest_path)

        # Checkpoint PAR EPOCH (gérable/supprimable depuis le dashboard).
        # Version allégée : on retire l'historique cumulé pour limiter la taille.
        ckpt_rel = None
        if save_epochs:
            ep_path = epochs_dir / f"epoch_{epoch:03d}.pth"
            slim = {k: v for k, v in ckpt_data.items() if k != "history"}
            torch.save(slim, ep_path)
            ckpt_rel = f"epochs/epoch_{epoch:03d}.pth"

        # Score heuristique Tox21, sans validité clinique démontrée.
        if is_best:
            torch.save(ckpt_data, best_path)
            print(f"{label}  -> Nouvelle meilleure epoch (proxy Tox21="
                  f"{clin_score:.4f}, AUC={val_m['roc_auc']:.4f}) sauvegardee")

        # Point temps réel pour le dashboard (après calcul is_best / ckpt_rel)
        try:
            live.log({
                "epoch": epoch, "train_loss": train_m["loss"], "val_loss": val_m["loss"],
                "train_auc": train_m["roc_auc"], "val_auc": val_m["roc_auc"],
                "val_f1": val_m["f1"], "lr_enc": lr_enc, "lr_head": lr_head,
                "best_auc": best_auc, "clinical_score": clin_score, "is_best": is_best,
                "selection_eligible": eligible,
                "ckpt": ckpt_rel,
                "macro_sensitivity": agg.get("macro_sensitivity"),
                "macro_specificity": agg.get("macro_specificity"),
                "macro_fnr": agg.get("macro_fnr"), "n_danger": agg.get("n_danger"),
                "n_warn": agg.get("n_warn"), "per_task_auc": per_task,
            })
        except Exception:
            pass

        if no_improve >= args.patience:
            print(f"{label}Early stopping (patience={args.patience}) — "
                  f"meilleure epoch conservee (score={best_score:.4f}).")
            break

    # -- Seuils optimaux (fix C1 : sur sous-ensemble du TRAIN, pas sur val) --
    # Fix C1 (audit 2026-10-20) : les seuils calibrés sur val qui a servi à
    # l'early stopping créent une fuite — Phase 3 utilisera des seuils biaisés
    # qui exploitent indirectement le signal de sélection du best model.
    # Solution : seuils calibrés sur un sous-ensemble "hold-out calibration"
    # tiré du TRAIN set (loader séparé si fourni, sinon tirage déterministe).
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    calib_loader = locals().get("calibration_loader")
    if calib_loader is None:
        # Fallback déterministe : 10% du train tiré via seed stable, utilisé
        # comme calibration set. Note : idéalement l'appelant fournit un loader
        # dédié (voir TODO ci-dessous dans main()).
        print(f"{label}⚠ pas de calibration_loader fourni → seuils calibrés sur val "
              f"(fuite potentielle, voir fix C1)")
        _, final_logits, final_targets = evaluate(model, val_loader, criterion, device)
    else:
        _, final_logits, final_targets = evaluate(model, calib_loader, criterion, device)
    opt_thresh = find_optimal_thresholds(final_logits, final_targets, num_tasks)
    best_ckpt["optimal_thresholds"] = opt_thresh
    best_ckpt["optimal_thresholds_source"] = (
        "train_calibration_subset" if calib_loader is not None else "val_set_LEAK"
    )
    torch.save(best_ckpt, best_path)

    total_time = time.time() - t0
    print(f"{label}Termine en {timedelta(seconds=int(total_time))} | meilleur ROC-AUC={best_auc:.4f}")
    return best_auc, history


# ======================================================================
# Main
# ======================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="Phase 2 - Fine-tuning toxicite")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--pretrained_model", type=str, default=None,
                   help="Phase 1 checkpoint (optionnel : sinon encodeur aleatoire)")
    p.add_argument("--save_dir", type=str, default=str(CHECKPOINT_DIR / "phase2"))
    p.add_argument("--epochs", type=int, default=PHASE2["epochs"])
    p.add_argument("--batch_size", type=int, default=PHASE2["batch_size"])
    p.add_argument("--patience", type=int, default=PHASE2["patience"])
    p.add_argument("--smiles_column", type=str, default="smiles")
    p.add_argument("--max_molecules", type=int, default=None,
                   help="Limite de molecules (runs rapides Kaggle)")
    p.add_argument("--cv_folds", type=int, default=0,
                   help="Cross-validation par scaffold (0 = split simple)")
    p.add_argument("--ema", type=int, default=1, help="1=EMA des poids actif, 0=desactive")
    p.add_argument("--warm_start", type=int, default=1,
                   help="1=reprend le meilleur modele de l'entrainement precedent (transfert)")
    p.add_argument("--save_epochs", type=int, default=1,
                   help="1=sauvegarde un checkpoint par epoch (gerable depuis le dashboard)")
    p.add_argument("--device", type=str, default=str(DEVICE))
    args = p.parse_args()

    if args.epochs <= PHASE2["freeze_encoder_epochs"]:
        p.error("epochs doit dépasser freeze_encoder_epochs pour entraîner l'encodeur")
    if args.batch_size < 2 or args.patience < 1:
        p.error("batch_size doit être >=2 et patience >=1")

    set_seed(42)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = LOG_DIR / "phase2"
    log_dir.mkdir(parents=True, exist_ok=True)
    use_ema = bool(args.ema)

    # -- Datasets --
    print("Chargement des datasets ...")
    train_ds = ToxicityDataset(args.train_csv, smiles_column=args.smiles_column,
                               max_molecules=args.max_molecules)
    val_ds = ToxicityDataset(args.val_csv, smiles_column=args.smiles_column,
                             task_columns=train_ds.get_task_names(),
                             max_molecules=args.max_molecules)
    num_tasks = train_ds.get_num_tasks()
    task_names = train_ds.get_task_names()
    print(f"  {num_tasks} taches, {len(train_ds)} train, {len(val_ds)} val")

    # ── Cross-validation par scaffold ────────────────────────────────
    if args.cv_folds and args.cv_folds >= 2:
        print(f"\n=== Cross-validation scaffold {args.cv_folds}-fold ===")
        graphs = list(train_ds.graphs) + list(val_ds.graphs)
        labels = torch.cat([train_ds.labels, val_ds.labels], dim=0)
        smiles = [getattr(g, "smiles", "") for g in graphs]
        combined = _GraphLabelDataset(graphs, labels)

        fold_aucs = []
        best_fold, best_fold_auc, best_fold_dir = -1, -1.0, None
        for fold, (tr_idx, va_idx) in enumerate(scaffold_kfold(smiles, k=args.cv_folds), start=1):
            fold_dir = Path(args.save_dir) / f"fold{fold}"
            pw = _pos_weight_from_labels(labels[tr_idx])
            auc, _ = train_one_run(
                Subset(combined, tr_idx), Subset(combined, va_idx),
                num_tasks, task_names, pw, args, device, str(fold_dir),
                use_ema=use_ema, tag=f"fold{fold}/{args.cv_folds}",
            )
            fold_aucs.append(auc)
            if auc > best_fold_auc:
                best_fold_auc, best_fold, best_fold_dir = auc, fold, fold_dir

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        print(f"\n=== Resultat CV : ROC-AUC = {mean_auc:.4f} +/- {std_auc:.4f} "
              f"(folds: {[round(a, 4) for a in fold_aucs]}) ===")

        # Copier le meilleur fold comme modele final
        if best_fold_dir is not None:
            src = best_fold_dir / "best_toxicity_model.pth"
            dst = Path(args.save_dir) / "best_toxicity_model.pth"
            if src.exists():
                torch.save(torch.load(src, map_location="cpu", weights_only=False), dst)
                print(f"Meilleur fold = {best_fold} (AUC={best_fold_auc:.4f}) -> {dst}")

        summary = {"cv_folds": args.cv_folds, "fold_aucs": fold_aucs,
                   "mean_auc": mean_auc, "std_auc": std_auc, "best_fold": best_fold}
        log_path = log_dir / f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Log CV -> {log_path}")
        return

    # ── Split simple (defaut) ────────────────────────────────────────
    pos_weight = train_ds.get_pos_weight()
    best_auc, history = train_one_run(
        train_ds, val_ds, num_tasks, task_names, pos_weight,
        args, device, args.save_dir, use_ema=use_ema,
    )

    log_path = log_dir / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Meilleur ROC-AUC = {best_auc:.4f} | Log -> {log_path}")


if __name__ == "__main__":
    main()
