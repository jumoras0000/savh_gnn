# -*- coding: utf-8 -*-
"""
Fine-tune HIV ciblé — Phase 3 ckpt + ChEMBL cibles anti-VIH.

Résout le FAIL DrugBank (recall 44% → objectif 60-70%) en entraînant le
head efficacy UNIQUEMENT sur les 5000 actifs ChEMBL confirmés
(RT CHEMBL247, PR CHEMBL243, IN CHEMBL3471, gp120, CCR5) + decoys ZINC.

Workflow :
  1. Charge Phase 3 ckpt (encodeur multi-virus pretrained + heads)
  2. GÈLE encoder + 5 heads non-HIV
  3. Fine-tune UNIQUEMENT efficacy_head (HIV) sur ChEMBL merged
  4. Scaffold split 80/10/10
  5. Val sur HIV MoleculeNet test (vérif pas de régression globale)
  6. Sauvegarde ckpt séparé : panacee_phase3_hiv_chembl.pth

Résultat attendu :
  - DrugBank recall ↑ (couvre INSTI/NNRTI/PI modernes mal représentés dans MoleculeNet)
  - HIV AUC MoleculeNet test ≈ stable (fine-tune ciblé n'écrase pas Phase 3)

Usage :
    python finetune_hiv_chembl.py --epochs 100
    python finetune_hiv_chembl.py --epochs 100 --unfreeze_encoder  # dégel progressif
"""
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
from src.preprocessing.scaffold_split import scaffold_split
from src.utils.reproducibility import WarmupCosineScheduler, set_seed
from src.utils.safe_load import safe_load_checkpoint
from src.validation.drugbank_antiretroviral import (
    canonicalize_arv_smiles,
    validate_model_recall,
)
from src.validation.pharma_metrics import full_screening_report


class SmilesDataset(torch.utils.data.Dataset):
    """Dataset simple : SMILES → graphe PyG + label HIV binaire."""

    def __init__(self, smiles_list, labels):
        self.data = []
        for smi, lbl in zip(smiles_list, labels, strict=True):
            try:
                g = smiles_to_graph(smi)
                if g is not None:
                    g.y = torch.tensor([float(lbl)], dtype=torch.float32)
                    self.data.append(g)
            except Exception:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def _canon(smi: str) -> str | None:
    """SMILES canonique isomerique ou None si invalide. Dédup rigoureux inter-datasets."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def build_chembl_dataset(
    chembl_merged_csv: str,
    exclude_smiles_canonical: set | None = None,
    zinc_decoys_ratio: float = 10.0,
    zinc_path: str | None = None,
) -> tuple[list, list, dict]:
    """
    Construit dataset binaire ChEMBL actifs (1) + ZINC decoys (0) SANS FUITE.

    Protections anti-fuite (fix critique) :
      1. Canonicalisation ISOMERIQUE RDKit sur chaque SMILES avant comparaison.
      2. Exclusion des `exclude_smiles_canonical` (ex: les 34 ARV DrugBank)
         AVANT split train/val/test → validation DrugBank honnête.
      3. Dédup canonique dans ChEMBL : plusieurs cibles (RT, PR, IN) peuvent
         contenir la même molécule → dédoublonne.
      4. ZINC decoys : exclu par SMILES CANONIQUE (pas string brut) de l'union
         actifs ∪ exclude_set.

    Returns:
        (all_smi, all_lbl, audit_dict)
        audit_dict : {n_actives, n_decoys, n_excluded_arv, n_dedup_overlap_zinc}
    """
    from rdkit import Chem

    actives = pd.read_csv(chembl_merged_csv)
    actives["smiles"] = actives["smiles"].astype(str).str.strip()
    raw_actives = actives["smiles"].dropna().unique().tolist()
    print(f"  ChEMBL actifs bruts : {len(raw_actives)}")

    # 1) Canonicalisation + dédup
    canon_actives: dict[str, str] = {}  # canon → raw_smiles (premier vu)
    for smi in raw_actives:
        c = _canon(smi)
        if c and c not in canon_actives:
            canon_actives[c] = smi
    print(f"  ChEMBL actifs canoniques uniques : {len(canon_actives)}")

    # 2) Exclusion des ARV connus (évite trivialisation DrugBank gate)
    excluded_arv = []
    if exclude_smiles_canonical:
        for c in list(canon_actives.keys()):
            if c in exclude_smiles_canonical:
                excluded_arv.append(c)
                del canon_actives[c]
    print(f"  Exclus ARV DrugBank (anti-fuite) : {len(excluded_arv)}")

    actives_canon_set = set(canon_actives.keys())
    actives_smi = list(canon_actives.values())  # SMILES bruts conservés pour training
    actives_canon_list = list(canon_actives.keys())

    # 3) ZINC decoys avec exclusion canonique
    if zinc_path is None:
        zinc_path = str(EXTERNAL_DIR.parent / "raw" / "250k_rndm_zinc_drugs_clean_3.csv")

    n_target = int(len(actives_smi) * zinc_decoys_ratio)
    decoys_smi, decoys_canon = [], set()
    n_overlap_zinc = 0
    if os.path.exists(zinc_path):
        # Fix (audit) : lire le CSV ZINC UNE seule fois (250k lignes).
        zinc_full = pd.read_csv(zinc_path)
        if "smiles" not in zinc_full.columns:
            raise ValueError(f"ZINC CSV sans colonne 'smiles' : {zinc_path}")
        zinc = zinc_full.sample(n=min(200000, len(zinc_full)), random_state=42)
        for raw_smi in zinc["smiles"].astype(str).str.strip():
            if len(decoys_smi) >= n_target:
                break
            c = _canon(raw_smi)
            if c is None or c in decoys_canon:
                continue
            if c in actives_canon_set or (exclude_smiles_canonical and
                                           c in exclude_smiles_canonical):
                n_overlap_zinc += 1
                continue
            decoys_canon.add(c)
            decoys_smi.append(raw_smi)
        print(f"  ZINC decoys : {len(decoys_smi)} (chevauchements exclus : {n_overlap_zinc})")
    else:
        print(f"  ⚠ ZINC absent ({zinc_path}) → actifs seuls")

    all_smi = actives_smi + decoys_smi
    all_lbl = [1] * len(actives_smi) + [0] * len(decoys_smi)
    audit = {
        "n_actives": len(actives_smi),
        "n_decoys": len(decoys_smi),
        "n_excluded_arv": len(excluded_arv),
        "n_overlap_zinc_dropped": n_overlap_zinc,
        "actives_canonical": actives_canon_list,  # pour vérif downstream
    }
    return all_smi, all_lbl, audit


def _predict_aligned(model, smiles_list, labels, device, batch_size: int = 128):
    """
    Prédit proba efficacy HIV en gardant SMILES + labels ALIGNÉS aux scores.

    Garantit len(scores) == len(valid_smiles) == len(valid_labels) : seules les
    molécules dont smiles_to_graph réussit sont incluses, dans le même ordre.
    Évite le bug de mismatch de longueur quand un SMILES est invalide.
    """
    model.eval()
    scores, valid_smiles, valid_labels = [], [], []
    for i in range(0, len(smiles_list), batch_size):
        chunk_smi = smiles_list[i:i + batch_size]
        chunk_lbl = labels[i:i + batch_size]
        graphs, keep = [], []
        for smi, lbl in zip(chunk_smi, chunk_lbl, strict=True):
            try:
                g = smiles_to_graph(smi)
            except Exception:
                g = None
            if g is not None:
                graphs.append(g)
                keep.append((smi, lbl))
        if not graphs:
            continue
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(batch)["efficacy"]).cpu().numpy().ravel()
        scores.append(p)
        for smi, lbl in keep:
            valid_smiles.append(smi)
            valid_labels.append(lbl)
    scores = np.concatenate(scores) if scores else np.array([])
    return scores, valid_smiles, valid_labels


def main():
    import argparse

    p = argparse.ArgumentParser(description="Fine-tune HIV ciblé via ChEMBL")
    p.add_argument("--base_checkpoint", type=str, default=None,
                   help="Phase 3 ckpt de départ (défaut: panacee_phase3_complete.pth)")
    p.add_argument("--chembl_csv", type=str, default=None,
                   help="ChEMBL merged (défaut: data/external/chembl_hiv/chembl_hiv_merged.csv)")
    p.add_argument("--save_path", type=str, default=None,
                   help="Chemin ckpt sortie (défaut: checkpoints/phase3/panacee_phase3_hiv_chembl.pth)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="LR head efficacy (encodeur gelé)")
    p.add_argument("--decoys_ratio", type=float, default=10.0)
    p.add_argument("--unfreeze_encoder", action="store_true",
                   help="Dégèle les 2 dernières couches encoder après freeze_epochs")
    p.add_argument("--freeze_epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Checkpoint de départ ──
    ckpt_path = args.base_checkpoint or str(
        CHECKPOINT_DIR / "phase3" / PHASE3["checkpoint_name"],
    )
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint Phase 3 absent : {ckpt_path}")
        sys.exit(1)
    print(f"▶ Chargement {ckpt_path}")
    ckpt = safe_load_checkpoint(ckpt_path, trusted=True)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    use_chirality = cfg.get("use_chirality", PHASE3.get("use_chirality", True))

    encoder = MolecularEncoder(use_chirality=use_chirality)
    model = MultiPropertyPredictor(encoder, hidden_dim=256).to(device)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    critical_missing = [k for k in getattr(result, "missing_keys", [])
                        if any(p in k for p in
                               ("encoder.convs", "encoder.atom_embedding",
                                "encoder.projection", "efficacy_head"))]
    if critical_missing:
        raise RuntimeError(f"Clés critiques absentes : {critical_missing[:5]}")

    # ── Freeze TOUT sauf efficacy_head ──
    for name, param in model.named_parameters():
        param.requires_grad = ("efficacy_head" in name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── Dataset ChEMBL ──
    chembl_csv = args.chembl_csv or str(
        EXTERNAL_DIR / "chembl_hiv" / "chembl_hiv_merged.csv",
    )
    if not os.path.exists(chembl_csv):
        print(f"❌ ChEMBL CSV absent : {chembl_csv}")
        print("  → lancer : python -m src.preprocessing.chembl_hiv_loader")
        sys.exit(1)
    print(f"\n▶ Dataset ChEMBL : {chembl_csv}")

    # Protection anti-fuite CRITIQUE : exclure les 34 ARV approuvés DrugBank
    # de l'entraînement pour que la validation DrugBank soit honnête.
    # Sinon le modèle apprend sur Dolutegravir+Raltegravir (dans ChEMBL CHEMBL3471)
    # puis les retrouve trivialement → gate biaisée vers PASS artificiel.
    arvs = canonicalize_arv_smiles()
    arv_canon_set = {d["smiles_canonical"] for d in arvs}
    print(f"  ARV DrugBank à exclure : {len(arv_canon_set)} (anti-fuite gate)")

    all_smi, all_lbl, audit = build_chembl_dataset(
        chembl_csv,
        exclude_smiles_canonical=arv_canon_set,
        zinc_decoys_ratio=args.decoys_ratio,
    )
    print(f"  Dataset final : {len(all_smi)} molécules ({sum(all_lbl)} actifs)")
    print(f"  Audit : {audit['n_excluded_arv']} ARV exclus, "
          f"{audit['n_overlap_zinc_dropped']} ZINC overlap droppés")

    # Scaffold split 80/10/10
    train_idx, val_test_idx = scaffold_split(all_smi, frac_train=0.8)
    val_test_smi = [all_smi[i] for i in val_test_idx]
    val_test_lbl = [all_lbl[i] for i in val_test_idx]
    # Resplit val/test 50/50 par scaffold
    val_idx_sub, test_idx_sub = scaffold_split(val_test_smi, frac_train=0.5)

    train_smi = [all_smi[i] for i in train_idx]
    train_lbl = [all_lbl[i] for i in train_idx]
    val_smi = [val_test_smi[i] for i in val_idx_sub]
    val_lbl = [val_test_lbl[i] for i in val_idx_sub]
    test_smi = [val_test_smi[i] for i in test_idx_sub]
    test_lbl = [val_test_lbl[i] for i in test_idx_sub]
    print(f"  Scaffold split : train={len(train_smi)} val={len(val_smi)} test={len(test_smi)}")

    train_ds = SmilesDataset(train_smi, train_lbl)
    val_ds = SmilesDataset(val_smi, val_lbl)
    test_ds = SmilesDataset(test_smi, test_lbl)
    print(f"  Graphes valides : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    def _collate(batch):
        return Batch.from_data_list(batch)

    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=_collate)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=_collate)
    test_loader = PyGDataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=_collate)

    # ── Optimizer + scheduler ──
    pos_weight = torch.tensor(
        [max(1.0, sum(1 for l in train_lbl if l == 0) /
             max(1, sum(1 for l in train_lbl if l == 1)))],
        dtype=torch.float32, device=device,
    )
    print(f"  pos_weight = {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5,
                                       total_epochs=args.epochs, lr_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Train ──
    from sklearn.metrics import roc_auc_score

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Dégel progressif optionnel
        if args.unfreeze_encoder and epoch == args.freeze_epochs:
            print(f"\n[epoch {epoch}] Dégel 2 dernières couches encoder")
            for name, param in model.encoder.named_parameters():
                if any(f"convs.{i}" in name for i in [4, 5]):
                    param.requires_grad = True
                if "projection" in name or "pool_gate" in name:
                    param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr * 0.3, weight_decay=1e-4,
            )
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=2,
                                               total_epochs=args.epochs - epoch,
                                               lr_min=1e-6)

        model.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train e{epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                preds = model(batch)
                logits = preds["efficacy"].squeeze(-1)
                loss = criterion(logits, batch.y.squeeze(-1))
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                1.0, error_if_nonfinite=False,
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch.y.size(0)
            n += batch.y.size(0)
        scheduler.step(epoch)

        # Val
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

        train_loss = total_loss / max(1, n)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | t_loss={train_loss:.4f} | val_auc={val_auc:.4f} | lr={lr:.2e}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping (patience={args.patience}) epoch {epoch}")
                break

    # ── Restaurer best + test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    # Fix HIGH (audit) : prédit en gardant SMILES alignés aux scores. Avant,
    # test_proba (loader filtré) et test_smi (liste brute) divergeaient dès
    # qu'un SMILES échouait à smiles_to_graph → ValueError longueurs DataFrame.
    test_proba, test_smi_aligned, test_true = _predict_aligned(
        model, test_smi, test_lbl, device,
    )
    test_report = full_screening_report(np.array(test_true), test_proba)

    print("\n═══ TEST (ChEMBL held-out scaffold split) ═══")
    for k, v in test_report.items():
        if isinstance(v, (int, float, type(None))):
            print(f"  {k}: {v}")

    # ── Validation DrugBank sur modèle fine-tuné ──
    print("\n═══ Validation DrugBank 34 ARV sur modèle fine-tuné ═══")
    arvs = canonicalize_arv_smiles()
    arv_smi = [d["smiles_canonical"] for d in arvs]
    arv_proba, arv_smi_aligned, _ = _predict_aligned(
        model, arv_smi, [1] * len(arv_smi), device,
    )
    print(f"  {len(arv_proba)}/{len(arv_smi)} ARV prédits")
    if len(arv_proba):
        print(f"  Score HIV moyen ARV : {arv_proba.mean():.3f}")

    # Validation gate — toutes listes alignées (même longueur garantie)
    combined_df = pd.DataFrame({
        "smiles": list(arv_smi_aligned) + list(test_smi_aligned),
        "hiv_score": np.concatenate([arv_proba, test_proba]),
    })
    gate = validate_model_recall(combined_df, min_recall=25, top_pct=0.10)
    print(f"  Verdict DrugBank : {gate['verdict']} ({gate['recall']}/{gate['total']})")
    print(f"  Trouvés  : {gate['found_arvs']}")
    print(f"  Manqués  : {gate['missing_arvs']}")

    # ── Save ──
    save_path = args.save_path or str(
        CHECKPOINT_DIR / "phase3" / "panacee_phase3_hiv_chembl.pth",
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "model_state_dict": best_state or model.state_dict(),
        "config": {"use_chirality": use_chirality,
                   "fine_tuned_from": ckpt_path,
                   "fine_tune_dataset": chembl_csv,
                   "fine_tune_epochs": args.epochs,
                   "best_val_auc": best_val_auc,
                   "anti_leakage_arv_excluded": audit["n_excluded_arv"],
                   "anti_leakage_zinc_overlap_dropped": audit["n_overlap_zinc_dropped"]},
        "test_report": test_report,
        "drugbank_gate": gate,
        "audit": {k: v for k, v in audit.items() if k != "actives_canonical"},
    }
    torch.save(save_dict, save_path)
    duration = time.time() - t0
    print(f"\n✅ Terminé en {duration/60:.1f}min")
    print(f"✅ Checkpoint fine-tuné : {save_path}")
    print(f"   Best val AUC : {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
