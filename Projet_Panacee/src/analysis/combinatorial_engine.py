"""
Moteur d'Analyse Combinatoire Moléculaire – Phase 3.

Ce module utilise le modèle Phase 3 entraîné pour :
  1. Charger des molécules candidates (SMILES)
  2. Prédire toutes leurs propriétés (toxicité, efficacité, solubilité, etc.)
  3. Analyser les combinaisons possibles via le MolecularReasoner
  4. Calculer les scores de synergie entre molécules
  5. Prédire les doses optimales
  6. Générer un rapport complet avec pourcentages de réussite
  7. Raisonnement avancé (MCTS, Bayesian, Pareto, chaîne de pensée)
  8. Vérification via base de connaissances médicales et APIs web

Usage :
  python -m src.analysis.combinatorial_engine --smiles_file molecules.csv --checkpoint phase3.pth
  python -m src.analysis.combinatorial_engine --smiles "CCO,CC(=O)O" --checkpoint phase3.pth
"""
import json
import logging
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ATOM_FEATURE_DIM,
    ATTENTION_HEADS,
    BOND_FEATURE_DIM,
    CHECKPOINT_DIR,
    CONV_TYPE,
    DEVICE,
    DROPOUT,
    HIDDEN_DIM,
    NUM_GNN_LAYERS,
    OUTPUT_DIM,
    PHASE3,
)
from src.models.encoder import MolecularEncoder
from src.models.multi_property_head import MultiPropertyPredictor
from src.models.reasoner import MolecularReasoner
from src.preprocessing.graph_builder import smiles_to_graph
from src.utils.gpu_manager import get_gpu_manager

logger = logging.getLogger("panacee.analysis")

try:
    from torch_geometric.data import Batch
except ImportError as e:
    raise ImportError("torch_geometric requis : pip install torch-geometric") from e


# ══════════════════════════════════════════════════════════════════════
# Classe principale
# ══════════════════════════════════════════════════════════════════════

class PanaceeAnalyzer:
    """
    Moteur d'analyse IA pour la découverte de médicaments.

    Charge le modèle Phase 3 et fournit :
      - predict_properties()  → propriétés d'une molécule
      - analyze_combination() → synergie entre molécules
      - find_best_combos()    → meilleures combinaisons
      - generate_report()     → rapport complet
    """

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        """
        Args:
            checkpoint_path: chemin vers panacee_phase3_complete.pth
            device: 'cuda' ou 'cpu'
        """
        self.device = torch.device(device or str(DEVICE))

        # Chemin par défaut
        if checkpoint_path is None:
            checkpoint_path = str(CHECKPOINT_DIR / "phase3" / PHASE3["checkpoint_name"])

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_path}\n"
                f"Lancez d'abord : python run_phase3.py --download"
            )

        print("🔬 Chargement du modèle Panacée Phase 3...")
        print(f"   Checkpoint : {checkpoint_path}")
        print(f"   Device : {self.device}")

        # Charger le checkpoint (sécurisé : weights_only par défaut, cf. safe_load)
        from src.utils.safe_load import safe_load_checkpoint
        self.ckpt = safe_load_checkpoint(checkpoint_path)

        # Reconstruire l'encodeur
        config = self.ckpt.get("config", {})
        self.encoder = MolecularEncoder(
            atom_dim=config.get("atom_feature_dim", ATOM_FEATURE_DIM),
            hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
            num_layers=config.get("num_gnn_layers", NUM_GNN_LAYERS),
            edge_dim=config.get("bond_feature_dim", BOND_FEATURE_DIM),
            output_dim=config.get("output_dim", OUTPUT_DIM),
            dropout=config.get("dropout", DROPOUT),
            conv_type=config.get("conv_type", CONV_TYPE),
            attention_heads=config.get("attention_heads", ATTENTION_HEADS),
        )

        # Reconstruire le modèle multi-propriétés
        self.model = MultiPropertyPredictor(
            encoder=self.encoder,
            hidden_dim=config.get("hidden_dim", HIDDEN_DIM),
            dropout=config.get("dropout", DROPOUT),
        )

        # Charger les poids
        self.model.load_state_dict(self.ckpt["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Reconstruire le Raisonneur IA
        r_config = self.ckpt.get("reasoner_config", {})
        self.reasoner = MolecularReasoner(
            mol_dim=config.get("output_dim", OUTPUT_DIM),
            hidden_dim=r_config.get("hidden_dim", PHASE3["reasoner_hidden_dim"]),
            num_heads=r_config.get("num_heads", PHASE3["reasoner_num_heads"]),
            num_layers=r_config.get("num_layers", PHASE3["reasoner_num_layers"]),
            max_molecules=r_config.get("max_molecules", PHASE3["max_molecules_combo"]),
            num_dose_levels=r_config.get("num_dose_levels", len(PHASE3["dose_levels"])),
            dropout=r_config.get("dropout", PHASE3["reasoner_dropout"]),
        )

        if "reasoner_state_dict" in self.ckpt:
            self.reasoner.load_state_dict(self.ckpt["reasoner_state_dict"])
        self.reasoner.to(self.device)
        self.reasoner.eval()

        self.dose_levels = r_config.get("dose_levels", PHASE3["dose_levels"])
        self.max_molecules = r_config.get("max_molecules", PHASE3["max_molecules_combo"])

        print("   ✓ Modèle chargé avec succès")
        print("   ✓ Raisonneur IA prêt")

    # ──────────────────────────────────────────────────────────────────
    # Prédiction de propriétés
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_properties(self, smiles: str) -> Optional[Dict]:
        """
        Prédit toutes les propriétés d'une molécule.

        Args:
            smiles: chaîne SMILES de la molécule

        Returns:
            dict avec :
              - smiles: str
              - toxicity: dict {task: {prob, label}}
              - efficacy: float (probabilité d'activité)
              - solubility: float (LogS prédit)
              - lipophilicity: float (LogP prédit)
              - bioavailability: float (probabilité)
              - metabolic_stability: float (probabilité)
              - safety_score: float [0,1]
              - drug_likeness: float [0,1]
        """
        graph = smiles_to_graph(smiles)
        if graph is None:
            print(f"  ⚠ SMILES invalide : {smiles}")
            return None

        batch = Batch.from_data_list([graph]).to(self.device)
        predictions = self.model(batch)

        result = {"smiles": smiles}

        # ── Toxicité (12 tâches Tox21) ──
        tox_probs = torch.sigmoid(predictions["toxicity"]).cpu().numpy()[0]
        tox21_tasks = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ]
        result["toxicity"] = {}
        for i, task_name in enumerate(tox21_tasks):
            if i < len(tox_probs):
                prob = float(tox_probs[i])
                result["toxicity"][task_name] = {
                    "probabilite": round(prob * 100, 1),
                    "toxique": prob > 0.5,
                }

        # Score de sécurité moyen (inverse de la toxicité moyenne)
        avg_tox = float(tox_probs.mean())
        result["safety_score"] = round((1 - avg_tox) * 100, 1)

        # ── Efficacité ──
        eff_prob = torch.sigmoid(predictions["efficacy"]).cpu().item()
        result["efficacy"] = {
            "probabilite_activite": round(eff_prob * 100, 1),
            "actif": eff_prob > 0.5,
        }

        # ── Solubilité ──
        sol_val = predictions["solubility"].cpu().item()
        result["solubility"] = {
            "log_s": round(sol_val, 3),
            "interpretation": self._interpret_solubility(sol_val),
        }

        # ── Lipophilicité ──
        lipo_val = predictions["lipophilicity"].cpu().item()
        result["lipophilicity"] = {
            "log_p": round(lipo_val, 3),
            "interpretation": self._interpret_lipophilicity(lipo_val),
        }

        # ── Biodisponibilité ──
        bio_prob = torch.sigmoid(predictions["bioavailability"]).cpu().item()
        result["bioavailability"] = {
            "probabilite": round(bio_prob * 100, 1),
            "biodisponible": bio_prob > 0.5,
        }

        # ── Stabilité métabolique ──
        stab_prob = torch.sigmoid(predictions["metabolic_stability"]).cpu().item()
        result["metabolic_stability"] = {
            "probabilite": round(stab_prob * 100, 1),
            "stable": stab_prob > 0.5,
        }

        # ── Score Drug-Likeness composite ──
        drug_score = self._compute_drug_likeness(result)
        result["drug_likeness"] = drug_score

        return result

    @torch.no_grad()
    def _get_embedding(self, smiles: str) -> Optional[torch.Tensor]:
        """Retourne l'embedding d'une molécule."""
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None
        batch = Batch.from_data_list([graph]).to(self.device)
        return self.model.encode(batch)  # [1, D]

    # ──────────────────────────────────────────────────────────────────
    # Analyse combinatoire
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def analyze_combination(self, smiles_list: List[str]) -> Optional[Dict]:
        """
        Analyse une combinaison de molécules.

        Args:
            smiles_list: liste de SMILES (2 à max_molecules)

        Returns:
            dict avec :
              - molecules: list de résultats individuels
              - synergy_matrix: matrice de synergie
              - dose_recommendations: doses par molécule
              - confidence: confiance du modèle
              - success_score: score de réussite prédit
              - combined_safety: sécurité combinée
              - interpretation: analyse textuelle
        """
        if len(smiles_list) < 2:
            print("  ⚠ Il faut au moins 2 molécules pour une analyse combinatoire")
            return None

        if len(smiles_list) > self.max_molecules:
            print(f"  ⚠ Maximum {self.max_molecules} molécules par combinaison")
            smiles_list = smiles_list[:self.max_molecules]

        # Prédire les propriétés individuelles
        individual_results = []
        embeddings = []
        valid_smiles = []

        for smi in smiles_list:
            props = self.predict_properties(smi)
            emb = self._get_embedding(smi)
            if props is not None and emb is not None:
                individual_results.append(props)
                embeddings.append(emb)
                valid_smiles.append(smi)

        if len(embeddings) < 2:
            print("  ⚠ Pas assez de molécules valides pour l'analyse")
            return None

        N = len(embeddings)

        # Construire le tenseur d'embeddings [1, N, D]
        emb_tensor = torch.cat(embeddings, dim=0).unsqueeze(0)  # [1, N, D]

        # Padding si nécessaire
        if self.max_molecules > N:
            pad = torch.zeros(1, self.max_molecules - N, emb_tensor.shape[-1],
                              device=self.device)
            emb_padded = torch.cat([emb_tensor, pad], dim=1)
            mask = torch.zeros(1, self.max_molecules, dtype=torch.bool, device=self.device)
            mask[0, N:] = True
        else:
            emb_padded = emb_tensor
            mask = None

        # Passer dans le Raisonneur IA
        reasoner_output = self.reasoner(emb_padded, mask)

        # Extraire les résultats
        synergy_matrix = reasoner_output["synergy_matrix"][0, :N, :N].cpu().numpy()
        dose_dists = reasoner_output["dose_distributions"][0, :N].cpu().numpy()
        confidence = reasoner_output["confidence"][0].cpu().item()
        success_score = reasoner_output["success_score"][0].cpu().item()

        # Construire les recommandations de dose
        dose_recommendations = []
        for i in range(N):
            best_dose_idx = int(np.argmax(dose_dists[i]))
            best_dose = self.dose_levels[best_dose_idx]
            dose_dist = {
                f"{self.dose_levels[j]} mg/kg": round(float(dose_dists[i, j]) * 100, 1)
                for j in range(len(self.dose_levels))
            }
            dose_recommendations.append({
                "smiles": valid_smiles[i],
                "dose_optimale_mg_kg": best_dose,
                "distribution": dose_dist,
            })

        # Sécurité combinée
        safety_scores = [r["safety_score"] for r in individual_results]
        combined_safety = float(np.mean(safety_scores))

        # Synergie moyenne
        synergy_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                syn_score = float(synergy_matrix[i, j])
                synergy_pairs.append({
                    "molecule_1": valid_smiles[i],
                    "molecule_2": valid_smiles[j],
                    "synergie": round(syn_score * 100, 1),
                    "type": "Synergique" if syn_score > PHASE3["synergy_threshold"] else
                            "Additive" if syn_score > 0.4 else "Antagoniste",
                })

        # Interprétation IA
        interpretation = self._generate_interpretation(
            individual_results, synergy_pairs, dose_recommendations,
            confidence, success_score, combined_safety,
        )

        return {
            "molecules": individual_results,
            "synergy_pairs": synergy_pairs,
            "synergy_matrix": synergy_matrix.tolist(),
            "dose_recommendations": dose_recommendations,
            "confidence": round(confidence * 100, 1),
            "success_score": round(success_score * 100, 1),
            "combined_safety": round(combined_safety, 1),
            "interpretation": interpretation,
        }

    # ──────────────────────────────────────────────────────────────────
    # Recherche des meilleures combinaisons
    # ──────────────────────────────────────────────────────────────────

    def find_best_combos(
        self, smiles_list: List[str],
        combo_size: int = 2,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Teste toutes les combinaisons possibles et retourne les meilleures.

        Args:
            smiles_list: liste de SMILES candidats
            combo_size: taille des combinaisons (2, 3, etc.)
            top_k: nombre de meilleures à retourner

        Returns:
            list triée des meilleures combinaisons
        """
        print(f"\n🔍 Analyse combinatoire : {len(smiles_list)} molécules, "
              f"combinaisons de {combo_size}")

        valid_mols = []
        for smi in smiles_list:
            if smiles_to_graph(smi) is not None:
                valid_mols.append(smi)
            else:
                print(f"  ⚠ SMILES invalide ignoré : {smi}")

        if len(valid_mols) < combo_size:
            print(f"  ⚠ Pas assez de molécules valides ({len(valid_mols)} < {combo_size})")
            return []

        all_combos = list(combinations(valid_mols, combo_size))
        print(f"  {len(all_combos)} combinaisons à tester...")

        results = []
        for i, combo in enumerate(all_combos):
            try:
                analysis = self.analyze_combination(list(combo))
                if analysis is not None:
                    results.append({
                        "rank": 0,
                        "molecules": list(combo),
                        "success_score": analysis["success_score"],
                        "confidence": analysis["confidence"],
                        "combined_safety": analysis["combined_safety"],
                        "synergy_pairs": analysis["synergy_pairs"],
                        "dose_recommendations": analysis["dose_recommendations"],
                    })
            except Exception as e:
                print(f"  ⚠ Erreur combo {combo}: {e}")

            if (i + 1) % 10 == 0:
                print(f"  ... {i+1}/{len(all_combos)} analysées")

        # Trier par score de réussite décroissant
        results.sort(key=lambda x: x["success_score"], reverse=True)

        # Assigner les rangs
        for rank, r in enumerate(results[:top_k], 1):
            r["rank"] = rank

        return results[:top_k]

    # ──────────────────────────────────────────────────────────────────
    # Génération de rapport
    # ──────────────────────────────────────────────────────────────────

    def generate_report(
        self,
        smiles_list: List[str],
        combo_size: int = 2,
        top_k: int = 5,
        output_file: str | None = None,
    ) -> str:
        """
        Génère un rapport complet d'analyse.

        Args:
            smiles_list: SMILES des molécules candidates
            combo_size: taille des combinaisons
            top_k: nombre de meilleures combinaisons
            output_file: fichier de sortie (JSON ou None pour stdout)

        Returns:
            rapport en format texte
        """
        print("=" * 80)
        print("🧪 RAPPORT D'ANALYSE PANACÉE – PHASE 3")
        print(f"   Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Molécules : {len(smiles_list)}")
        print("=" * 80)

        # 1. Analyse individuelle
        print("\n" + "─" * 40)
        print("📋 PROPRIÉTÉS INDIVIDUELLES")
        print("─" * 40)

        individual_results = []
        for smi in smiles_list:
            result = self.predict_properties(smi)
            if result is not None:
                individual_results.append(result)
                self._print_molecule_summary(result)

        # 2. Meilleures combinaisons
        print("\n" + "─" * 40)
        print(f"🏆 TOP {top_k} COMBINAISONS (taille={combo_size})")
        print("─" * 40)

        best_combos = self.find_best_combos(smiles_list, combo_size, top_k)

        report_lines = []
        for combo_result in best_combos:
            self._print_combo_summary(combo_result)
            report_lines.append(combo_result)

        # 3. Sauvegarder en JSON
        full_report = {
            "date": datetime.now().isoformat(),
            "num_molecules": len(smiles_list),
            "individual_results": individual_results,
            "best_combinations": report_lines,
            "parameters": {
                "combo_size": combo_size,
                "top_k": top_k,
                "model_checkpoint": str(CHECKPOINT_DIR / "phase3" / PHASE3["checkpoint_name"]),
            },
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📄 Rapport sauvegardé : {output_file}")

        return json.dumps(full_report, indent=2, ensure_ascii=False, default=str)

    # ──────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _interpret_solubility(log_s: float) -> str:
        if log_s > -1:
            return "Très soluble"
        elif log_s > -3:
            return "Soluble"
        elif log_s > -5:
            return "Modérément soluble"
        elif log_s > -7:
            return "Peu soluble"
        else:
            return "Insoluble"

    @staticmethod
    def _interpret_lipophilicity(log_p: float) -> str:
        if log_p < 0:
            return "Hydrophile"
        elif log_p < 2:
            return "Modérément lipophile"
        elif log_p < 5:
            return "Lipophile (bon pour absorption orale)"
        else:
            return "Très lipophile (risque d'accumulation)"

    @staticmethod
    def _compute_drug_likeness(result: Dict) -> Dict:
        """
        Calcule un score de drug-likeness composite.
        Basé sur des règles pharmaceutiques adaptées.
        """
        score = 0.0
        max_score = 0.0
        details = {}

        # Sécurité (poids 3)
        safety = result.get("safety_score", 50)
        safety_score = min(safety / 100.0, 1.0)
        score += 3.0 * safety_score
        max_score += 3.0
        details["securite"] = f"{safety}%"

        # Efficacité (poids 3)
        eff = result.get("efficacy", {}).get("probabilite_activite", 0)
        eff_score = min(eff / 100.0, 1.0)
        score += 3.0 * eff_score
        max_score += 3.0
        details["efficacite"] = f"{eff}%"

        # Biodisponibilité (poids 2)
        bio = result.get("bioavailability", {}).get("probabilite", 0)
        bio_score = min(bio / 100.0, 1.0)
        score += 2.0 * bio_score
        max_score += 2.0
        details["biodisponibilite"] = f"{bio}%"

        # Stabilité métabolique (poids 1)
        stab = result.get("metabolic_stability", {}).get("probabilite", 0)
        stab_score = min(stab / 100.0, 1.0)
        score += 1.0 * stab_score
        max_score += 1.0
        details["stabilite"] = f"{stab}%"

        # Solubilité (poids 1)
        sol_interp = result.get("solubility", {}).get("interpretation", "")
        sol_map = {"Très soluble": 1.0, "Soluble": 0.8, "Modérément soluble": 0.5,
                   "Peu soluble": 0.2, "Insoluble": 0.0}
        sol_score = sol_map.get(sol_interp, 0.5)
        score += 1.0 * sol_score
        max_score += 1.0
        details["solubilite"] = sol_interp

        final_score = round((score / max_score) * 100, 1) if max_score > 0 else 0
        details["score_global"] = f"{final_score}%"

        return details

    def _generate_interpretation(
        self, molecules, synergy_pairs, doses, confidence, success, safety
    ) -> str:
        """Génère une interprétation textuelle des résultats."""
        lines = []
        lines.append("=== ANALYSE IA PANACÉE ===")
        lines.append(f"Confiance du modèle : {round(confidence * 100, 1)}%")
        lines.append(f"Score de réussite prédit : {round(success * 100, 1)}%")
        lines.append(f"Sécurité combinée : {round(safety, 1)}%")
        lines.append("")

        # Analyse de chaque paire
        for pair in synergy_pairs:
            syn = pair["synergie"]
            typ = pair["type"]
            lines.append(f"  {pair['molecule_1'][:30]}... + {pair['molecule_2'][:30]}...")
            lines.append(f"    Synergie : {syn}% ({typ})")

        lines.append("")

        # Recommandations
        if success * 100 > 70:
            lines.append("✅ RECOMMANDATION : Combinaison PROMETTEUSE")
            lines.append("   Score de réussite élevé. À valider en laboratoire.")
        elif success * 100 > 40:
            lines.append("⚠ RECOMMANDATION : Combinaison MODÉRÉE")
            lines.append("   Résultats mitigés. Optimisation nécessaire.")
        else:
            lines.append("❌ RECOMMANDATION : Combinaison FAIBLE")
            lines.append("   Score de réussite bas. Explorer d'autres candidats.")

        # Alertes
        if safety < 50:
            lines.append("⚠ ALERTE SÉCURITÉ : Score de sécurité combiné bas !")

        return "\n".join(lines)

    def _print_molecule_summary(self, result: Dict):
        """Affiche un résumé d'une molécule."""
        smi = result["smiles"]
        print(f"\n  📌 {smi[:60]}{'...' if len(smi) > 60 else ''}")
        print(f"     Sécurité : {result['safety_score']}%")
        print(f"     Efficacité : {result['efficacy']['probabilite_activite']}%"
              f" ({'Actif' if result['efficacy']['actif'] else 'Inactif'})")
        print(f"     Solubilité : {result['solubility']['interpretation']}"
              f" (LogS={result['solubility']['log_s']})")
        print(f"     LogP : {result['lipophilicity']['log_p']}"
              f" ({result['lipophilicity']['interpretation']})")
        print(f"     Biodisponibilité : {result['bioavailability']['probabilite']}%")
        print(f"     Stabilité métab. : {result['metabolic_stability']['probabilite']}%")
        dl = result["drug_likeness"]
        print(f"     Drug-Likeness : {dl['score_global']}")

        # Alertes toxicité
        tox_alerts = [
            name for name, data in result["toxicity"].items()
            if data["toxique"]
        ]
        if tox_alerts:
            print(f"     ⚠ Toxicité détectée : {', '.join(tox_alerts)}")
        else:
            print("     ✅ Pas de toxicité détectée")

    def _print_combo_summary(self, combo: Dict):
        """Affiche un résumé d'une combinaison."""
        print(f"\n  🏅 Rang #{combo['rank']}")
        print(f"     Molécules : {len(combo['molecules'])}")
        for smi in combo["molecules"]:
            print(f"       - {smi[:50]}{'...' if len(smi) > 50 else ''}")
        print(f"     Score réussite : {combo['success_score']}%")
        print(f"     Confiance : {combo['confidence']}%")
        print(f"     Sécurité combinée : {combo['combined_safety']}%")

        for pair in combo.get("synergy_pairs", []):
            print(f"     Synergie : {pair['synergie']}% ({pair['type']})")

    # ──────────────────────────────────────────────────────────────────
    # Analyse avancée (MCTS, Pareto, connaissances, web)
    # ──────────────────────────────────────────────────────────────────

    def advanced_analysis(
        self,
        smiles_list: List[str],
        combo_size: int = 2,
        top_k: int = 5,
        use_mcts: bool = True,
        use_pareto: bool = True,
        use_knowledge: bool = True,
        use_web: bool = False,
        mcts_iterations: int = 300,
        indication: str = "",
        output_file: str | None = None,
    ) -> Dict:
        """
        Analyse avancée combinant tous les algorithmes.

        Args:
            smiles_list: SMILES des molécules candidates
            combo_size: taille des combinaisons
            top_k: nombre de meilleures combinaisons
            use_mcts: utiliser MCTS pour exploration
            use_pareto: utiliser optimisation multi-objectif
            use_knowledge: utiliser la base de connaissances médicales
            use_web: interroger PubChem/ChEMBL/PubMed
            mcts_iterations: nombre d'itérations MCTS
            indication: indication thérapeutique visée
            output_file: fichier de sortie JSON

        Returns:
            Rapport complet avec raisonnement avancé
        """
        from src.utils.error_handler import setup_logging
        setup_logging()

        gpu = get_gpu_manager()
        gpu.print_summary()

        print("=" * 80)
        print("  ANALYSE AVANCÉE PANACÉE – RAISONNEMENT IA MULTI-NIVEAU")
        print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Molécules : {len(smiles_list)} | Combinaisons de {combo_size}")
        print(f"  MCTS: {'Oui' if use_mcts else 'Non'} | "
              f"Pareto: {'Oui' if use_pareto else 'Non'} | "
              f"Web: {'Oui' if use_web else 'Non'}")
        print("=" * 80)

        report = {
            "date": datetime.now().isoformat(),
            "config": {
                "n_molecules": len(smiles_list),
                "combo_size": combo_size,
                "use_mcts": use_mcts,
                "use_pareto": use_pareto,
                "use_knowledge": use_knowledge,
                "use_web": use_web,
                "indication": indication,
            },
        }

        # ── 1. Propriétés individuelles + connaissances ──
        print("\n── ÉTAPE 1 : Analyse individuelle ──")
        individual = []
        valid_smiles = []
        embeddings = []

        for smi in smiles_list:
            props = self.predict_properties(smi)
            emb = self._get_embedding(smi)
            if props and emb is not None:
                # Enrichir avec connaissances médicales
                if use_knowledge:
                    try:
                        from src.knowledge.medical_rules import comprehensive_evaluation
                        med_eval = comprehensive_evaluation(smi)
                        if med_eval.get("valid"):
                            props["medical_evaluation"] = {
                                "lipinski_score": med_eval["lipinski_score"],
                                "drug_likeness_expert": round(med_eval["drug_likeness"], 3),
                                "bioavailability_estimated": round(med_eval["estimated_bioavailability_pct"], 1),
                                "structural_alerts": med_eval["structural_alerts"],
                                "overall_expert_score": round(med_eval["overall_score"], 3),
                            }
                    except Exception as e:
                        logger.debug(f"Connaissances médicales ignorées: {e}")

                individual.append(props)
                valid_smiles.append(smi)
                embeddings.append(emb)
                self._print_molecule_summary(props)

        report["individual_results"] = individual

        if len(valid_smiles) < combo_size:
            print(f"  Pas assez de molécules valides ({len(valid_smiles)})")
            return report

        # ── 2. Recherche des meilleures combinaisons ──
        print("\n── ÉTAPE 2 : Analyse combinatoire ──")

        if use_mcts and len(valid_smiles) > 6:
            # Pour les grands ensembles, utiliser MCTS
            print(f"  MCTS activé ({mcts_iterations} itérations)...")
            try:
                from src.models.advanced_reasoner import MCTSCombinationSearch

                def score_combo(indices):
                    combo_smi = [valid_smiles[i] for i in indices]
                    result = self.analyze_combination(combo_smi)
                    if result is None:
                        return 0.0
                    return result["success_score"] / 100.0

                mcts = MCTSCombinationSearch(
                    score_function=score_combo,
                    n_molecules=len(valid_smiles),
                    combo_size=combo_size,
                )
                best_indices, best_score = mcts.search(n_iterations=mcts_iterations)
                best_combo = [valid_smiles[i] for i in best_indices]
                print(f"  MCTS meilleure combinaison: score={best_score:.3f}")
                report["mcts_best"] = {
                    "molecules": best_combo,
                    "score": best_score,
                }
            except Exception as e:
                logger.warning(f"MCTS échoué: {e}")
                print(f"  MCTS indisponible: {e}")

        # Analyse combinatoire classique
        best_combos = self.find_best_combos(valid_smiles, combo_size, top_k)
        report["best_combinations"] = best_combos

        for combo in best_combos:
            self._print_combo_summary(combo)

        # ── 3. Optimisation multi-objectif de Pareto ──
        if use_pareto and individual:
            print("\n── ÉTAPE 3 : Optimisation Pareto ──")
            try:
                from src.models.advanced_reasoner import MultiObjectiveOptimizer

                optimizer = MultiObjectiveOptimizer(
                    objective_names=["efficacité", "sécurité", "biodisponibilité", "solubilité"],
                    maximize=[True, True, True, True],
                )

                for i, mol in enumerate(individual):
                    optimizer.add_solution(
                        index=i,
                        objectives=[
                            mol.get("efficacy", {}).get("probabilite_activite", 0) / 100,
                            mol.get("safety_score", 0) / 100,
                            mol.get("bioavailability", {}).get("probabilite", 0) / 100,
                            {"Très soluble": 1, "Soluble": 0.8, "Modérément soluble": 0.5,
                             "Peu soluble": 0.2, "Insoluble": 0}.get(
                                mol.get("solubility", {}).get("interpretation", ""), 0.5
                            ),
                        ],
                        metadata={"smiles": mol["smiles"]},
                    )

                front = optimizer.get_pareto_front()
                best_compromise = optimizer.suggest_best_compromise()

                pareto_results = []
                for sol in front:
                    pareto_results.append({
                        "smiles": sol.metadata.get("smiles", ""),
                        "objectives": dict(zip(optimizer.objective_names, sol.objectives, strict=False)),
                    })

                report["pareto_front"] = pareto_results
                if best_compromise:
                    report["pareto_best_compromise"] = {
                        "smiles": best_compromise.metadata.get("smiles", ""),
                        "objectives": dict(zip(optimizer.objective_names, best_compromise.objectives, strict=False)),
                    }
                    print(f"  Front de Pareto: {len(front)} solutions non-dominées")
                    print(f"  Meilleur compromis: {best_compromise.metadata.get('smiles', '')[:40]}...")

            except Exception as e:
                logger.warning(f"Pareto échoué: {e}")

        # ── 4. Raisonnement avancé sur la meilleure combinaison ──
        if best_combos:
            print("\n── ÉTAPE 4 : Raisonnement avancé ──")
            top_combo = best_combos[0]
            try:
                from src.models.advanced_reasoner import AdvancedMolecularReasoner

                adv_reasoner = AdvancedMolecularReasoner(
                    nn_reasoner=self.reasoner,
                    encoder=self.encoder,
                    device=self.device,
                    dose_levels=self.dose_levels,
                )

                top_embs = []
                for smi in top_combo["molecules"]:
                    emb = self._get_embedding(smi)
                    if emb is not None:
                        top_embs.append(emb)

                if len(top_embs) >= 2:
                    emb_tensor = torch.cat(top_embs, dim=0)
                    adv_results = adv_reasoner.full_analysis(
                        mol_embeddings=emb_tensor,
                        mol_names=top_combo["molecules"],
                        indication=indication,
                        use_bayesian=True,
                        use_web_search=use_web,
                    )

                    report["advanced_reasoning"] = {
                        "final_score": adv_results["final_score"],
                        "uncertainty": adv_results["uncertainty"],
                        "pair_analyses": adv_results["pair_analyses"],
                        "dose_optimization": adv_results["dose_optimization"],
                    }
                    print(adv_results["report"])

            except Exception as e:
                logger.warning(f"Raisonnement avancé échoué: {e}")
                print(f"  Raisonnement avancé indisponible: {e}")

        # ── 5. Vérification web (optionnel) ──
        if use_web and best_combos:
            print("\n── ÉTAPE 5 : Vérification bases de données ──")
            try:
                from src.knowledge.web_search import WebResearchEngine
                engine = WebResearchEngine()

                web_results = []
                for smi in best_combos[0]["molecules"][:3]:
                    research = engine.research_molecule(smi)
                    if research.get("sources"):
                        web_results.append({
                            "smiles": smi[:40],
                            "sources": research["sources"],
                            "pubchem": research.get("pubchem"),
                            "chembl": research.get("chembl"),
                        })
                        print(f"  {smi[:40]}: trouvé dans {', '.join(research['sources'])}")

                if indication:
                    verification = engine.verify_hypothesis(
                        f"{indication} drug combination therapy"
                    )
                    report["hypothesis_verification"] = verification
                    print(f"  Littérature: {verification['articles_found']} articles "
                          f"(niveau: {verification['evidence_level']})")

                report["web_research"] = web_results

            except Exception as e:
                logger.warning(f"Recherche web échouée: {e}")
                print(f"  Recherche web indisponible: {e}")

        # ── Sauvegarde ──
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n  Rapport sauvegardé : {output_file}")

        gpu.clear_memory()
        return report


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    p = argparse.ArgumentParser(description="Analyse Combinatoire Panacée")
    p.add_argument("--smiles", type=str, default=None,
                   help="SMILES séparés par des virgules")
    p.add_argument("--smiles_file", type=str, default=None,
                   help="CSV avec colonne 'smiles'")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Chemin du checkpoint Phase 3")
    p.add_argument("--combo_size", type=int, default=2,
                   help="Taille des combinaisons")
    p.add_argument("--top_k", type=int, default=5,
                   help="Nombre de meilleures combinaisons")
    p.add_argument("--output", type=str, default=None,
                   help="Fichier de sortie JSON")
    p.add_argument("--predict_only", action="store_true",
                   help="Prédire les propriétés sans analyse combinatoire")
    args = p.parse_args()

    # Récupérer les SMILES
    smiles_list = []
    if args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(",") if s.strip()]
    elif args.smiles_file:
        if not os.path.exists(args.smiles_file):
            print(f"❌ Fichier introuvable : {args.smiles_file}")
            sys.exit(1)
        df = pd.read_csv(args.smiles_file)
        smiles_col = None
        for candidate in ["smiles", "SMILES", "canonical_smiles", "ids", "mol"]:
            if candidate in df.columns:
                smiles_col = candidate
                break
        if smiles_col is None:
            print(f"❌ Pas de colonne SMILES trouvée dans {args.smiles_file}")
            sys.exit(1)
        smiles_list = df[smiles_col].dropna().astype(str).tolist()
    else:
        print("❌ Spécifiez --smiles ou --smiles_file")
        sys.exit(1)

    if not smiles_list:
        print("❌ Aucune molécule fournie")
        sys.exit(1)

    print(f"📥 {len(smiles_list)} molécules à analyser")

    # Charger le modèle
    try:
        analyzer = PanaceeAnalyzer(checkpoint_path=args.checkpoint)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.predict_only:
        # Mode prédiction simple
        print("\n📊 PRÉDICTIONS INDIVIDUELLES")
        for smi in smiles_list:
            result = analyzer.predict_properties(smi)
            if result:
                analyzer._print_molecule_summary(result)
    else:
        # Rapport complet
        output_file = args.output or f"rapport_panacee_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.generate_report(
            smiles_list,
            combo_size=args.combo_size,
            top_k=args.top_k,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()
