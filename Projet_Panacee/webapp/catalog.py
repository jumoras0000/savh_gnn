# -*- coding: utf-8 -*-
"""
Catalogue (données pures, aucune dépendance) :
  - CAPABILITIES   : tout ce qu'on peut faire avec l'application + le modèle.
  - LAB_EQUIVALENCE: équivalent « paillasse » de chaque analyse in silico.
  - LIBRARIES      : bibliothèques de molécules de référence (validées par RDKit
                     au moment de l'usage ; les SMILES invalides sont ignorés).
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# Ce que l'utilisateur peut faire (affiché dans l'onglet Guide)
# ──────────────────────────────────────────────────────────────────────
CAPABILITIES = [
    {
        "group": "🎛️ Entraînement",
        "items": [
            "Lancer / arrêter les Phases 1 (MGM), 2 (toxicité), 3 (multi-propriétés) depuis l'UI.",
            "Suivre en temps réel (SSE) loss, AUC, sécurité — avant / pendant / après.",
            "Régler epochs, nombre de molécules, cross-validation, EMA, téléchargement des données.",
            "Console de logs et statut du process en direct.",
        ],
    },
    {
        "group": "🏥 Évaluation clinique",
        "items": [
            "Évaluer un checkpoint sur un CSV de validation (par endpoint toxicologique).",
            "Métriques sensibles : sensibilité, spécificité, FNR, précision, F1, ROC-AUC, PR-AUC, calibration (ECE).",
            "Verdict de sécurité (déployable / à surveiller / dangereux) + alertes triées par gravité.",
            "Lecture automatique « observation & risque » de chaque métrique.",
        ],
    },
    {
        "group": "🧪 Recherche moléculaire",
        "items": [
            "Analyser une molécule réelle (SMILES) : toxicité 12 endpoints, sécurité, efficacité, solubilité, lipophilie, biodisponibilité, stabilité, drug-likeness.",
            "Cheminformatique TOUJOURS disponible (sans modèle) : MW, LogP, TPSA, HBD/HBA, liaisons rotatives, cycles, QED, règle de Lipinski.",
            "Visualisation 2D de la structure (SVG).",
            "Évaluation de risque par molécule (OK / à surveiller / danger).",
            "Fonctionne avec un modèle Phase 3 (complet), Phase 2 (toxicité seule) ou en mode descripteurs seuls.",
        ],
    },
    {
        "group": "🧬 Criblage virtuel (dont anti-VIH)",
        "items": [
            "Cribler une bibliothèque de molécules (collée, importée en CSV, ou de référence) et la classer.",
            "Objectifs : efficacité anti-VIH (Phase 3), sécurité, drug-likeness (QED — sans modèle).",
            "Identifier les meilleurs candidats médicaments à partir de l'existant.",
            "Export des résultats (JSON / CSV).",
        ],
    },
    {
        "group": "🔗 Combinaisons & raisonnement",
        "items": [
            "Analyser une combinaison de molécules : synergie par paire, doses optimales, score de réussite, confiance.",
            "Interprétation IA (MolecularReasoner).",
        ],
    },
    {
        "group": "📁 Données & export",
        "items": [
            "Sélecteurs automatiques des checkpoints (.pth) et CSV ; import de fichiers.",
            "Comparaison de plusieurs runs.",
            "Export des analyses et du criblage.",
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────
# Équivalence « laboratoire » de chaque analyse in silico
# ──────────────────────────────────────────────────────────────────────
LAB_EQUIVALENCE = [
    {"analyse": "Toxicité Tox21 (12 endpoints)",
     "in_silico": "Probabilité d'activité sur récepteurs nucléaires / réponse au stress",
     "labo": "Panel d'essais in vitro Tox21 (HTS sur lignées cellulaires, dosages luciférase)"},
    {"analyse": "FNR / Sensibilité",
     "in_silico": "Toxiques manqués / détectés par le modèle",
     "labo": "Taux de faux négatifs d'un test de dépistage toxicologique réglementaire"},
    {"analyse": "Efficacité anti-VIH",
     "in_silico": "Probabilité d'activité antivirale prédite",
     "labo": "Essai antiviral cellulaire (EC50/IC50 sur cellules infectées par le VIH)"},
    {"analyse": "Solubilité (LogS)",
     "in_silico": "Solubilité aqueuse prédite",
     "labo": "Solubilité cinétique (shake-flask, néphélométrie)"},
    {"analyse": "Lipophilie (LogP)",
     "in_silico": "Coefficient de partage prédit",
     "labo": "LogP octanol/eau mesuré (shake-flask / HPLC)"},
    {"analyse": "Biodisponibilité",
     "in_silico": "Probabilité d'absorption orale",
     "labo": "Perméabilité Caco-2/PAMPA, %F en pharmacocinétique in vivo"},
    {"analyse": "Stabilité métabolique",
     "in_silico": "Probabilité de stabilité",
     "labo": "Incubation microsomes hépatiques (t½, clairance intrinsèque)"},
    {"analyse": "Drug-likeness / Lipinski",
     "in_silico": "Score composite + règle des 5",
     "labo": "Filtres ADME précliniques, profilage physico-chimique"},
    {"analyse": "Synergie de combinaison",
     "in_silico": "Matrice de synergie prédite",
     "labo": "Damier (checkerboard), modèles Bliss / Loewe, indice de combinaison"},
    {"analyse": "Dose optimale",
     "in_silico": "Distribution de dose recommandée",
     "labo": "Courbe dose-réponse, étude d'escalade de dose"},
    {"analyse": "Criblage virtuel",
     "in_silico": "Classement d'une bibliothèque par le modèle",
     "labo": "Criblage à haut débit (HTS) en plaques 96/384 puits"},
]

# ──────────────────────────────────────────────────────────────────────
# Bibliothèques de molécules de référence (SMILES validés à l'usage)
# ──────────────────────────────────────────────────────────────────────
LIBRARIES = {
    "hiv_reference": {
        "label": "Antirétroviraux de référence (VIH)",
        "note": "Médicaments anti-VIH connus — utiles pour étalonner la prédiction d'efficacité.",
        "molecules": [
            {"name": "Zidovudine (AZT)", "smiles": "CC1=CN(C2CC(N=[N+]=[N-])C(CO)O2)C(=O)NC1=O"},
            {"name": "Lamivudine (3TC)", "smiles": "OCC1OC(N2C=CC(N)=NC2=O)CS1"},
            {"name": "Stavudine (d4T)", "smiles": "CC1=CN(C2C=CC(CO)O2)C(=O)NC1=O"},
            {"name": "Efavirenz", "smiles": "FC(F)(F)C1(C#CC2CC2)OC(=O)Nc2ccc(Cl)cc21"},
            {"name": "Emtricitabine (FTC)", "smiles": "OCC1OC(N2C=C(F)C(N)=NC2=O)CS1"},
            {"name": "Didanosine (ddI)", "smiles": "C1CC(OC1CO)N2C=NC3=C2N=CNC3=O"},
        ],
    },
    "reference_drugs": {
        "label": "Médicaments courants (référence)",
        "note": "Petites molécules très connues, pour tester / comparer rapidement.",
        "molecules": [
            {"name": "Paracétamol", "smiles": "CC(=O)Nc1ccc(O)cc1"},
            {"name": "Aspirine", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"name": "Caféine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
            {"name": "Ibuprofène", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
            {"name": "Pénicilline G", "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"},
        ],
    },
}
