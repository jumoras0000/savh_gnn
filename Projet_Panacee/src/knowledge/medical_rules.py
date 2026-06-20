"""
Base de connaissances médicales, chimiques et physiques.

Intègre des règles expertes validées pour évaluer les molécules :
  - Lipinski Rule of 5 (drug-likeness)
  - Règles ADMET (Absorption, Distribution, Métabolisme, Excrétion, Toxicité)
  - Alertes structurales (PAINS, groupes réactifs toxiques)
  - Pharmacocinétique de base
  - Interactions médicamenteuses connues
  - Propriétés physico-chimiques
"""
import math
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("panacee.knowledge")


# ─────────────────────────────────────────────
#  Structures de données
# ─────────────────────────────────────────────

@dataclass
class MolecularProfile:
    """Profil complet d'une molécule."""
    smiles: str
    mw: float = 0.0          # Masse moléculaire
    logp: float = 0.0        # Coefficient de partage octanol/eau
    hbd: int = 0             # Donneurs de liaisons H
    hba: int = 0             # Accepteurs de liaisons H
    tpsa: float = 0.0        # Surface polaire topologique
    rotatable_bonds: int = 0 # Liaisons rotatives
    aromatic_rings: int = 0  # Cycles aromatiques
    heavy_atoms: int = 0     # Atomes lourds

    # Scores calculés
    lipinski_score: float = 0.0
    drug_likeness: float = 0.0
    admet_scores: Dict[str, float] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


@dataclass
class DrugInteraction:
    """Interaction médicamenteuse connue."""
    drug_a: str
    drug_b: str
    severity: str       # "major", "moderate", "minor"
    mechanism: str
    effect: str


# ─────────────────────────────────────────────
#  Règles de Lipinski (Rule of 5)
# ─────────────────────────────────────────────

class LipinskiEvaluator:
    """
    Évalue la drug-likeness selon Lipinski's Rule of 5.
    Un composé est "drug-like" s'il ne viole pas plus de 1 règle.

    Règles :
      1. MW ≤ 500 Da
      2. LogP ≤ 5.0
      3. HBD ≤ 5
      4. HBA ≤ 10
    """

    RULES = {
        "MW ≤ 500 Da": lambda p: p.mw <= 500,
        "LogP ≤ 5.0": lambda p: p.logp <= 5.0,
        "HBD ≤ 5": lambda p: p.hbd <= 5,
        "HBA ≤ 10": lambda p: p.hba <= 10,
    }

    @staticmethod
    def evaluate(profile: MolecularProfile) -> Tuple[float, List[str]]:
        """
        Returns:
            (score 0-1, liste des violations)
        """
        violations = []
        for rule_name, check in LipinskiEvaluator.RULES.items():
            if not check(profile):
                violations.append(rule_name)

        # Score: 1.0 si aucune violation, décroissance graduelle
        score = max(0.0, 1.0 - len(violations) * 0.3)
        return score, violations


# ─────────────────────────────────────────────
#  Règles ADMET
# ─────────────────────────────────────────────

class ADMETEvaluator:
    """
    Évalue les propriétés ADMET basées sur des modèles physico-chimiques.

    A - Absorption : biodisponibilité orale
    D - Distribution : passage des barrières biologiques
    M - Métabolisme : stabilité métabolique
    E - Excrétion : clairance
    T - Toxicité : alertes structurales
    """

    @staticmethod
    def evaluate_absorption(profile: MolecularProfile) -> Dict[str, float]:
        """Évalue la probabilité d'absorption orale."""
        scores = {}

        # Règle de Veber : bonne absorption si TPSA ≤ 140 et rotatable bonds ≤ 10
        veber_tpsa = 1.0 if profile.tpsa <= 140 else max(0, 1 - (profile.tpsa - 140) / 100)
        veber_rot = 1.0 if profile.rotatable_bonds <= 10 else max(0, 1 - (profile.rotatable_bonds - 10) / 10)
        scores["veber_oral_absorption"] = (veber_tpsa + veber_rot) / 2

        # Perméabilité intestinale (corrélée inversement avec TPSA)
        scores["intestinal_permeability"] = max(0, 1 - profile.tpsa / 200)

        # Solubilité aqueuse (corrélée inversement avec LogP)
        if profile.logp < -1:
            scores["aqueous_solubility"] = 0.7
        elif profile.logp <= 3:
            scores["aqueous_solubility"] = 1.0
        elif profile.logp <= 5:
            scores["aqueous_solubility"] = max(0, 1 - (profile.logp - 3) / 4)
        else:
            scores["aqueous_solubility"] = max(0, 0.5 - (profile.logp - 5) / 10)

        return scores

    @staticmethod
    def evaluate_distribution(profile: MolecularProfile) -> Dict[str, float]:
        """Évalue la distribution dans l'organisme."""
        scores = {}

        # Passage de la barrière hémato-encéphalique (BBB)
        # Modèle simplifié : LogP 1-3 et TPSA < 90 favorables
        bbb_logp = 1.0 if 1 <= profile.logp <= 3 else max(0, 1 - abs(profile.logp - 2) / 5)
        bbb_tpsa = 1.0 if profile.tpsa < 60 else max(0, 1 - (profile.tpsa - 60) / 120)
        bbb_mw = 1.0 if profile.mw < 400 else max(0, 1 - (profile.mw - 400) / 200)
        scores["bbb_penetration"] = (bbb_logp + bbb_tpsa + bbb_mw) / 3

        # Volume de distribution (Vd)
        # LogP élevé → grand Vd (distribution dans les tissus)
        scores["volume_distribution"] = min(1.0, max(0, profile.logp / 5))

        # Liaison aux protéines plasmatiques
        # LogP élevé → forte liaison
        scores["plasma_protein_binding"] = min(1.0, max(0, 0.3 + profile.logp * 0.15))

        return scores

    @staticmethod
    def evaluate_metabolism(profile: MolecularProfile) -> Dict[str, float]:
        """Évalue la stabilité métabolique."""
        scores = {}

        # Stabilité métabolique (heuristique)
        # MW élevé et nombreux cycles aromatiques → plus stable mais plus lent
        stab = 1.0
        if profile.rotatable_bonds > 7:
            stab -= 0.2
        if profile.aromatic_rings > 3:
            stab -= 0.15
        if profile.logp > 4:
            stab -= 0.2
        scores["metabolic_stability"] = max(0, min(1, stab))

        # Risque d'inhibition CYP450
        cyp_risk = 0.0
        if profile.logp > 3:
            cyp_risk += 0.3
        if profile.mw > 400:
            cyp_risk += 0.2
        if profile.aromatic_rings >= 3:
            cyp_risk += 0.2
        scores["cyp_inhibition_risk"] = min(1.0, cyp_risk)

        return scores

    @staticmethod
    def evaluate_excretion(profile: MolecularProfile) -> Dict[str, float]:
        """Évalue les propriétés d'excrétion."""
        scores = {}

        # Clairance rénale (molécules petites et polaires)
        renal = 0.5
        if profile.mw < 300 and profile.logp < 0:
            renal = 0.9
        elif profile.mw > 500:
            renal = 0.2
        scores["renal_clearance"] = renal

        # Demi-vie estimée (heuristique basée sur stabilité et clairance)
        # Plus la molécule est lipophile, plus longue la demi-vie
        half_life = min(1.0, max(0.1, 0.3 + profile.logp * 0.1 + profile.mw / 1000))
        scores["half_life_score"] = half_life

        return scores

    @staticmethod
    def evaluate_all(profile: MolecularProfile) -> Dict[str, float]:
        """Évalue toutes les propriétés ADMET."""
        scores = {}
        scores.update(ADMETEvaluator.evaluate_absorption(profile))
        scores.update(ADMETEvaluator.evaluate_distribution(profile))
        scores.update(ADMETEvaluator.evaluate_metabolism(profile))
        scores.update(ADMETEvaluator.evaluate_excretion(profile))

        # Score ADMET global
        if scores:
            scores["admet_global"] = sum(scores.values()) / len(scores)

        return scores


# ─────────────────────────────────────────────
#  Alertes structurales (PAINS et groupes toxiques)
# ─────────────────────────────────────────────

# SMARTS patterns pour groupes fonctionnels problématiques
STRUCTURAL_ALERTS = {
    # Groupes réactifs / toxiques
    "aldehyde": "[CX3H1](=O)[#6]",
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "epoxide": "C1OC1",
    "michael_acceptor": "[CX3]=[CX3][CX3](=O)",
    "nitro_aromatic": "[$(c1ccccc1[N+](=O)[O-])]",
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "peroxide": "[OX2][OX2]",
    "thiol": "[SX2H]",

    # Motifs PAINS (Pan Assay Interference compouNdS)
    "catechol": "c1cc(O)c(O)cc1",
    "quinone": "O=C1C=CC(=O)C=C1",
    "hydrazine": "[NX3][NX3]",
    "hydroxamic_acid": "[CX3](=O)[NX3][OX2H]",
    "sulfonamide": "[#16X4](=[OX1])(=[OX1])([NX3])",
}


def check_structural_alerts(smiles: str) -> List[str]:
    """
    Vérifie les alertes structurales d'une molécule.

    Args:
        smiles: SMILES de la molécule

    Returns:
        Liste des alertes trouvées
    """
    alerts = []
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ["smiles_invalide"]

        for alert_name, smarts in STRUCTURAL_ALERTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                alerts.append(alert_name)

    except ImportError:
        logger.debug("RDKit non disponible, alertes structurales désactivées")
    except Exception as e:
        logger.debug(f"Erreur vérification structurale: {e}")

    return alerts


# ─────────────────────────────────────────────
#  Calcul de propriétés avec RDKit
# ─────────────────────────────────────────────

def compute_molecular_profile(smiles: str) -> Optional[MolecularProfile]:
    """
    Calcule le profil moléculaire complet à partir du SMILES.
    Requiert RDKit.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        profile = MolecularProfile(
            smiles=smiles,
            mw=Descriptors.MolWt(mol),
            logp=Descriptors.MolLogP(mol),
            hbd=rdMolDescriptors.CalcNumHBD(mol),
            hba=rdMolDescriptors.CalcNumHBA(mol),
            tpsa=Descriptors.TPSA(mol),
            rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
            aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
            heavy_atoms=mol.GetNumHeavyAtoms(),
        )

        # Lipinski
        profile.lipinski_score, lipinski_violations = LipinskiEvaluator.evaluate(profile)

        # ADMET
        profile.admet_scores = ADMETEvaluator.evaluate_all(profile)

        # Alertes structurales
        profile.alerts = check_structural_alerts(smiles)

        # Drug-likeness composite
        admet_global = profile.admet_scores.get("admet_global", 0.5)
        alert_penalty = min(1.0, len(profile.alerts) * 0.2)
        profile.drug_likeness = (
            profile.lipinski_score * 0.4
            + admet_global * 0.4
            + (1 - alert_penalty) * 0.2
        )

        return profile

    except ImportError:
        logger.warning("RDKit non disponible - profil moléculaire limité")
        return MolecularProfile(smiles=smiles)
    except Exception as e:
        logger.error(f"Erreur calcul profil moléculaire: {e}")
        return None


# ─────────────────────────────────────────────
#  Base de connaissances pharmacologiques
# ─────────────────────────────────────────────

# Cibles thérapeutiques connues et leurs propriétés idéales
THERAPEUTIC_TARGETS = {
    "kinase_inhibitor": {
        "ideal_mw": (300, 550),
        "ideal_logp": (1.5, 4.5),
        "ideal_hba": (3, 8),
        "ideal_hbd": (1, 3),
        "description": "Inhibiteurs de kinases - cibles oncologiques majeures",
    },
    "gpcr_ligand": {
        "ideal_mw": (200, 500),
        "ideal_logp": (1.0, 4.0),
        "ideal_hba": (2, 7),
        "ideal_hbd": (0, 3),
        "description": "Ligands GPCR - récepteurs membranaires",
    },
    "protease_inhibitor": {
        "ideal_mw": (400, 700),
        "ideal_logp": (1.0, 5.0),
        "ideal_hba": (4, 10),
        "ideal_hbd": (2, 5),
        "description": "Inhibiteurs de protéases - antiviraux, anticancéreux",
    },
    "cns_drug": {
        "ideal_mw": (150, 450),
        "ideal_logp": (1.0, 3.5),
        "ideal_hba": (2, 5),
        "ideal_hbd": (0, 2),
        "description": "Médicaments SNC - passage BBB nécessaire",
    },
    "antibiotic": {
        "ideal_mw": (200, 600),
        "ideal_logp": (-2.0, 3.0),
        "ideal_hba": (3, 10),
        "ideal_hbd": (1, 5),
        "description": "Antibiotiques - large spectre d'activité requis",
    },
}


def evaluate_target_suitability(
    profile: MolecularProfile, target: str
) -> Tuple[float, str]:
    """
    Évalue l'adéquation d'une molécule pour une cible thérapeutique.

    Returns:
        (score 0-1, explication)
    """
    if target not in THERAPEUTIC_TARGETS:
        return 0.5, f"Cible '{target}' non répertoriée"

    t = THERAPEUTIC_TARGETS[target]
    scores = []
    explanations = []

    # MW
    lo, hi = t["ideal_mw"]
    if lo <= profile.mw <= hi:
        scores.append(1.0)
    else:
        dist = min(abs(profile.mw - lo), abs(profile.mw - hi))
        s = max(0, 1 - dist / 200)
        scores.append(s)
        explanations.append(f"MW={profile.mw:.0f} (idéal: {lo}-{hi})")

    # LogP
    lo, hi = t["ideal_logp"]
    if lo <= profile.logp <= hi:
        scores.append(1.0)
    else:
        dist = min(abs(profile.logp - lo), abs(profile.logp - hi))
        s = max(0, 1 - dist / 3)
        scores.append(s)
        explanations.append(f"LogP={profile.logp:.1f} (idéal: {lo}-{hi})")

    # HBA
    lo, hi = t["ideal_hba"]
    if lo <= profile.hba <= hi:
        scores.append(1.0)
    else:
        dist = min(abs(profile.hba - lo), abs(profile.hba - hi))
        s = max(0, 1 - dist / 5)
        scores.append(s)
        explanations.append(f"HBA={profile.hba} (idéal: {lo}-{hi})")

    # HBD
    lo, hi = t["ideal_hbd"]
    if lo <= profile.hbd <= hi:
        scores.append(1.0)
    else:
        dist = min(abs(profile.hbd - lo), abs(profile.hbd - hi))
        s = max(0, 1 - dist / 3)
        scores.append(s)
        explanations.append(f"HBD={profile.hbd} (idéal: {lo}-{hi})")

    avg_score = sum(scores) / len(scores) if scores else 0.5
    expl = f"{t['description']}. "
    if explanations:
        expl += "Écarts: " + "; ".join(explanations)
    else:
        expl += "Toutes les propriétés sont dans les plages idéales."

    return avg_score, expl


# ─────────────────────────────────────────────
#  Règles physico-chimiques
# ─────────────────────────────────────────────

def estimate_bioavailability(profile: MolecularProfile) -> float:
    """
    Estimation rapide de la biodisponibilité orale (%).
    Basée sur les travaux de Martin (2005) et Veber (2002).
    """
    score = 100.0

    # Poids moléculaire
    if profile.mw > 500:
        score -= min(30, (profile.mw - 500) * 0.1)

    # LogP
    if profile.logp > 5:
        score -= min(25, (profile.logp - 5) * 5)
    elif profile.logp < -1:
        score -= min(15, abs(profile.logp + 1) * 5)

    # TPSA
    if profile.tpsa > 140:
        score -= min(30, (profile.tpsa - 140) * 0.3)

    # Liaisons rotatives
    if profile.rotatable_bonds > 10:
        score -= min(20, (profile.rotatable_bonds - 10) * 4)

    # HBD
    if profile.hbd > 5:
        score -= min(15, (profile.hbd - 5) * 5)

    return max(0, min(100, score))


# ─────────────────────────────────────────────
#  Interactions connues (base simplifiée)
# ─────────────────────────────────────────────

KNOWN_INTERACTIONS = [
    DrugInteraction("warfarin", "aspirin", "major",
                    "Inhibition plaquettaire additive",
                    "Risque hémorragique accru"),
    DrugInteraction("metformin", "contrast_dye", "major",
                    "Acidose lactique",
                    "Arrêter metformin 48h avant"),
    DrugInteraction("ssri", "maoi", "major",
                    "Syndrome sérotoninergique",
                    "Association contre-indiquée"),
    DrugInteraction("statin", "fibrate", "moderate",
                    "Rhabdomyolyse",
                    "Surveillance CPK recommandée"),
    DrugInteraction("ace_inhibitor", "potassium", "moderate",
                    "Hyperkaliémie",
                    "Surveillance kaliémie"),
    DrugInteraction("nsaid", "anticoagulant", "moderate",
                    "Risque hémorragique",
                    "Association à éviter si possible"),
]


def check_known_interactions(drug_classes: List[str]) -> List[DrugInteraction]:
    """
    Vérifie les interactions connues entre classes thérapeutiques.

    Args:
        drug_classes: liste des classes thérapeutiques des molécules

    Returns:
        Liste des interactions trouvées
    """
    found = []
    classes_lower = [c.lower() for c in drug_classes]

    for interaction in KNOWN_INTERACTIONS:
        if (interaction.drug_a in classes_lower and
                interaction.drug_b in classes_lower):
            found.append(interaction)
        elif (interaction.drug_b in classes_lower and
              interaction.drug_a in classes_lower):
            found.append(interaction)

    return found


# ─────────────────────────────────────────────
#  Fonction d'évaluation globale
# ─────────────────────────────────────────────

def comprehensive_evaluation(smiles: str, target: Optional[str] = None) -> Dict:
    """
    Évaluation complète d'une molécule combinant toutes les règles.

    Args:
        smiles: SMILES de la molécule
        target: cible thérapeutique (optionnel)

    Returns:
        Dictionnaire complet d'évaluation
    """
    result = {"smiles": smiles, "valid": False}

    profile = compute_molecular_profile(smiles)
    if profile is None:
        result["error"] = "Impossible de calculer le profil moléculaire"
        return result

    result["valid"] = True
    result["molecular_weight"] = profile.mw
    result["logP"] = profile.logp
    result["lipinski_score"] = profile.lipinski_score
    result["drug_likeness"] = profile.drug_likeness
    result["admet"] = profile.admet_scores
    result["structural_alerts"] = profile.alerts
    result["estimated_bioavailability_pct"] = estimate_bioavailability(profile)

    if target:
        target_score, target_expl = evaluate_target_suitability(profile, target)
        result["target_suitability"] = target_score
        result["target_explanation"] = target_expl

    # Score final pondéré
    result["overall_score"] = (
        profile.drug_likeness * 0.35
        + profile.admet_scores.get("admet_global", 0.5) * 0.35
        + (result["estimated_bioavailability_pct"] / 100) * 0.15
        + (1 - min(1, len(profile.alerts) * 0.25)) * 0.15
    )

    return result
