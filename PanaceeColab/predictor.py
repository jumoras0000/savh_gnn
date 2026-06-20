"""
predictor.py – Inférence et analyse combinatoire
=================================================
Ce module charge le modèle entraîné et permet :
  • Prédiction de propriétés pour n'importe quelle molécule (SMILES)
  • Analyse combinatoire de plusieurs molécules
  • Identification des meilleures combinaisons (score + synergie)
  • Génération de rapport texte/HTML

Usage :
    predictor = PanaceePredictor.load("phase3_checkpoint.pth")
    results = predictor.predict("CC(=O)Nc1ccc(O)cc1")  # Paracétamol
    report  = predictor.generate_report(["CC(=O)Nc1ccc(O)cc1", "c1ccccc1"])
"""
import os
import json
import logging
from typing import List, Optional, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from config import DEVICE, CHECKPOINT_DIR, HIDDEN_DIM, DROPOUT
from graph_builder import smiles_to_graph
from gnn_models import build_encoder
from prediction_heads import MultiPropertyPredictor, MolecularReasoner
from torch_geometric.data import Batch

logger = logging.getLogger("panacee.predictor")


# ══════════════════════════════════════════════════════════════════════
#  NOMS LISIBLES
# ══════════════════════════════════════════════════════════════════════

PROPERTY_LABELS = {
    "toxicity"          : "Toxicité (12 cibles Tox21)",
    "efficacy"          : "Efficacité thérapeutique",
    "solubility"        : "Solubilité aqueuse (logS)",
    "lipophilicity"     : "Lipophilicité (logD)",
    "bioavailability"   : "Biodisponibilité orale",
    "metabolic_stability": "Stabilité métabolique",
}

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
    "SR-HSE", "SR-MMP", "SR-p53",
]


# ══════════════════════════════════════════════════════════════════════
#  PRÉDICATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class PanaceePredictor:
    """
    Interface d'inférence pour le modèle Panacee complet.

    Usage :
        p = PanaceePredictor.load("checkpoints/phase3/panacee_phase3.pth")
        result = p.predict("CC(=O)Nc1ccc(O)cc1")
        print(result)
    """

    def __init__(
        self,
        model: MultiPropertyPredictor,
        reasoner: Optional[MolecularReasoner],
        arch: str = "attfp",
        device    = None,
    ):
        self.model    = model.to(device or DEVICE)
        self.reasoner = reasoner.to(device or DEVICE) if reasoner is not None else None
        self.device   = device or DEVICE
        self.arch     = arch
        self.model.eval()
        if self.reasoner:
            self.reasoner.eval()

    # ────────────────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        arch: str = "attfp",
        device    = None,
    ) -> "PanaceePredictor":
        """
        Charge le modèle complet depuis un checkpoint Phase 3.
        """
        device = device or DEVICE
        logger.info(f"Chargement du checkpoint : {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_path}\n"
                f"Avez-vous lancé l'entraînement Phase 3 ?"
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        encoder = build_encoder(arch=arch)
        model   = MultiPropertyPredictor(encoder, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        model.load_state_dict(ckpt["model_state"], strict=False)

        reasoner = None
        if "reasoner_state" in ckpt:
            from config import PHASE3
            cfg = ckpt.get("config", PHASE3)
            reasoner = MolecularReasoner(
                mol_emb_dim=HIDDEN_DIM,
                d_model=cfg.get("reasoner_hidden_dim", 512),
                num_heads=cfg.get("reasoner_num_heads", 8),
                num_layers=cfg.get("reasoner_num_layers", 4),
                dropout=cfg.get("reasoner_dropout", 0.1),
            )
            reasoner.load_state_dict(ckpt["reasoner_state"], strict=False)

        logger.info(
            f"Modèle chargé (epoch {ckpt.get('epoch', '?')}, "
            f"val_loss={ckpt.get('best_val_loss', '?'):.5f})"
        )
        return cls(model, reasoner, arch=arch, device=device)

    # ────────────────────────────────────────────────────────────────
    def _smiles_to_batch(self, smiles_list: List[str]):
        """Convertit une liste de SMILES en batch PyG."""
        graphs = []
        valid_smiles = []
        for smi in smiles_list:
            g = smiles_to_graph(smi)
            if g is not None:
                graphs.append(g)
                valid_smiles.append(smi)
            else:
                logger.warning(f"SMILES invalide ignoré : {smi}")

        if not graphs:
            return None, []

        batch = Batch.from_data_list(graphs).to(self.device)
        return batch, valid_smiles

    # ────────────────────────────────────────────────────────────────
    def predict(self, smiles: str) -> dict:
        """
        Prédit toutes les propriétés moléculaires pour un SMILES.

        Returns:
            dict avec une clé par propriété + score global.
        """
        batch, valid = self._smiles_to_batch([smiles])
        if batch is None:
            return {"error": f"SMILES invalide : {smiles}"}

        with torch.no_grad():
            preds = self.model(batch)

        result = {"smiles": smiles}

        # Toxicité : 12 tâches → probabilités
        tox_logits = preds["toxicity"]          # [1, 12]
        tox_probs  = torch.sigmoid(tox_logits).squeeze(0).cpu().numpy()
        result["toxicity"] = {
            task: round(float(p), 4)
            for task, p in zip(TOX21_TASKS, tox_probs)
        }
        result["toxicity_max"]  = round(float(tox_probs.max()), 4)
        result["toxicity_mean"] = round(float(tox_probs.mean()), 4)

        # Efficacité (probabilité 0-1)
        eff = torch.sigmoid(preds["efficacy"]).item()
        result["efficacy"] = round(eff, 4)

        # Solubilité & lipophilicité (valeurs continues, pas de sigmoid)
        result["solubility"]    = round(float(preds["solubility"].squeeze().item()), 4)
        result["lipophilicity"] = round(float(preds["lipophilicity"].squeeze().item()), 4)

        # Biodisponibilité & stabilité métabolique
        result["bioavailability"]    = round(torch.sigmoid(preds["bioavailability"]).item(), 4)
        result["metabolic_stability"]= round(torch.sigmoid(preds["metabolic_stability"]).item(), 4)

        # Score global : combinaison pondérée
        result["global_score"] = round(
            0.35 * (1 - result["toxicity_max"]) +
            0.25 * result["efficacy"] +
            0.15 * result["bioavailability"] +
            0.15 * result["metabolic_stability"] +
            0.10 * max(0.0, 1.0 + result["solubility"] / 6.0),   # normalise logS
            4
        )

        return result

    # ────────────────────────────────────────────────────────────────
    def predict_batch(self, smiles_list: List[str]) -> List[dict]:
        """Prédit les propriétés pour une liste de SMILES."""
        return [self.predict(smi) for smi in smiles_list]

    # ────────────────────────────────────────────────────────────────
    def analyze_combination(self, smiles_list: List[str]) -> dict:
        """
        Analyse synergique d'une combinaison de molécules.
        Utilise le MolecularReasoner si disponible.

        Returns:
            dict {combo_score, synergy_matrix, individual_scores, ...}
        """
        if self.reasoner is None:
            # Fallback : moyenne des scores individuels
            individual = self.predict_batch(smiles_list)
            avg_score  = float(np.mean([r["global_score"] for r in individual
                                         if "global_score" in r]))
            return {
                "combo_score" : round(avg_score, 4),
                "confidence"  : 0.5,
                "individual"  : individual,
                "note"        : "MolecularReasoner non disponible — score moyen utilisé",
            }

        batch, valid = self._smiles_to_batch(smiles_list)
        if batch is None:
            return {"error": "Aucun SMILES valide"}

        with torch.no_grad():
            embeddings = self.model.encode(batch)   # [N, hidden_dim]
            mol_emb    = embeddings.unsqueeze(0)     # [1, N, hidden_dim]
            out        = self.reasoner(mol_emb)

        combo_score    = float(out["combo_score"].squeeze().cpu())
        confidence     = float(out["confidence"].squeeze().cpu())
        synergy_matrix = out["synergy_matrix"].squeeze(0).cpu().numpy().tolist()
        doses          = out["doses"].squeeze(0).cpu().numpy().tolist()

        individual = self.predict_batch(valid)

        return {
            "smiles_list"   : valid,
            "combo_score"   : round(combo_score, 4),
            "confidence"    : round(confidence, 4),
            "synergy_matrix": synergy_matrix,
            "optimal_doses" : doses,
            "individual"    : individual,
        }

    # ────────────────────────────────────────────────────────────────
    def find_best_combinations(
        self,
        smiles_list: List[str],
        top_k: int = 10,
        max_combo_size: int = 3,
    ) -> List[dict]:
        """
        Explore toutes les combinaisons de taille ≤ max_combo_size
        et retourne les top_k meilleures.

        Avertissement : O(n^k) – utiliser avec modération.
        """
        from itertools import combinations
        results = []
        n = len(smiles_list)

        for size in range(2, min(max_combo_size + 1, n + 1)):
            for combo in combinations(range(n), size):
                smi_combo = [smiles_list[i] for i in combo]
                res = self.analyze_combination(smi_combo)
                if "error" not in res:
                    results.append({
                        "indices"    : list(combo),
                        "smiles"     : smi_combo,
                        "combo_score": res["combo_score"],
                        "confidence" : res.get("confidence", 0.5),
                    })

        results.sort(key=lambda x: x["combo_score"], reverse=True)
        return results[:top_k]

    # ────────────────────────────────────────────────────────────────
    def generate_report(
        self,
        smiles_list: List[str],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Génère un rapport texte complet pour une liste de molécules.
        Retourne le rapport en chaîne de caractères et optionnellement
        le sauvegarde dans un fichier.
        """
        lines = [
            "=" * 70,
            "   RAPPORT PANACEE – ANALYSE MOLÉCULAIRE GNN",
            f"   Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}",
            "=" * 70,
            "",
        ]

        for i, smi in enumerate(smiles_list, 1):
            res = self.predict(smi)
            lines.append(f"Molécule {i} : {smi}")
            lines.append("-" * 50)

            if "error" in res:
                lines.append(f"  ✗ Erreur : {res['error']}")
            else:
                lines.append(f"  Score global         : {res['global_score']:.4f}")
                lines.append(f"  Toxicité max         : {res['toxicity_max']:.4f}")
                lines.append(f"  Efficacité           : {res['efficacy']:.4f}")
                lines.append(f"  Solubilité (logS)    : {res['solubility']:.4f}")
                lines.append(f"  Lipophilicité (logD) : {res['lipophilicity']:.4f}")
                lines.append(f"  Biodisponibilité     : {res['bioavailability']:.4f}")
                lines.append(f"  Stabilité métabolique: {res['metabolic_stability']:.4f}")

                # Tâches de toxicité
                lines.append("")
                lines.append("  Profil toxicité :")
                for task, prob in res["toxicity"].items():
                    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                    flag = " ⚠" if prob > 0.5 else ""
                    lines.append(f"    {task:12s}: {prob:.3f} |{bar}|{flag}")

            lines.append("")

        # Section combinaisons (si ≥ 2 molécules)
        if len(smiles_list) >= 2:
            lines.append("─" * 70)
            lines.append("ANALYSE COMBINATOIRE")
            lines.append("─" * 70)
            combo_res = self.analyze_combination(smiles_list)
            if "error" not in combo_res:
                lines.append(f"  Score combinaison : {combo_res['combo_score']:.4f}")
                lines.append(f"  Confiance         : {combo_res['confidence']:.4f}")

        report = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Rapport sauvegardé : {output_path}")

        return report
