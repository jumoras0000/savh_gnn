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
            "Sécurité structurelle SANS modèle : toxicophores par endpoint (mutagénicité Ames, hépatotoxicité DILI, cardiotoxicité hERG) + domaine d'applicabilité (fiabilité de la prédiction).",
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
    {"analyse": "Alertes structurelles (Ames/DILI/hERG)",
     "in_silico": "Détection de toxicophores par motifs SMARTS",
     "labo": "Test d'Ames (mutagénicité), bilan hépatique, patch-clamp hERG"},
    {"analyse": "Domaine d'applicabilité",
     "in_silico": "Similarité de Tanimoto au jeu de référence",
     "labo": "Jugement d'expert sur la pertinence d'un analogue / série chimique"},
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
# Vue d'ensemble du projet (onglet Lexique) — « c'est quoi, ça sert à quoi »
# ──────────────────────────────────────────────────────────────────────
PROJECT_OVERVIEW = {
    "pitch": (
        "Panacée est un laboratoire pharmaceutique « in silico » (sur ordinateur). "
        "Une intelligence artificielle apprend à LIRE la structure des molécules pour "
        "prédire si elles sont TOXIQUES, EFFICACES (notamment contre le VIH) et "
        "ADMINISTRABLES comme médicament — avant tout essai en laboratoire."
    ),
    "objectif": (
        "Accélérer et sécuriser la découverte de médicaments : trier des milliers de "
        "molécules en quelques minutes, écarter tôt les candidats dangereux, et faire "
        "remonter les plus prometteurs vers la validation expérimentale."
    ),
    "peut_faire": [
        "Entraîner le modèle (Phases 1→3) et suivre l'apprentissage en TEMPS RÉEL (courbes, sécurité).",
        "Analyser une vraie molécule (SMILES) : toxicité, efficacité, solubilité, drug-likeness.",
        "Cribler une bibliothèque entière et classer les meilleurs candidats (dont anti-VIH).",
        "Évaluer un modèle endpoint par endpoint et rendre un verdict de sécurité clinique.",
        "Étudier des combinaisons de molécules (synergie, doses).",
    ],
    "resultat_final": (
        "Le livrable est un MODÈLE validé + une liste classée de molécules candidates "
        "avec, pour chacune, un profil de sécurité et d'efficacité. Ce n'est PAS un "
        "diagnostic : c'est une aide à la décision qui oriente les chimistes vers les "
        "molécules à tester réellement (paillasse, in vitro, in vivo)."
    ),
    "avertissement": (
        "Outil d'aide à la décision in-silico. Toute conclusion doit être confirmée "
        "expérimentalement avant tout usage médical."
    ),
}

# Ce qui se passe à chaque phase d'entraînement
PHASES = [
    {
        "id": "Phase 1",
        "nom": "Pré-entraînement (MGM)",
        "but": "Apprendre la « grammaire » de la chimie sans étiquettes.",
        "entree": "~250 000 molécules brutes (SMILES, ZINC).",
        "sortie": "Un encodeur GNN qui comprend la structure moléculaire.",
        "analogie": "Apprendre à lire avant d'apprendre une matière précise.",
    },
    {
        "id": "Phase 2",
        "nom": "Affinage toxicité",
        "but": "Spécialiser le modèle sur la toxicité (12 endpoints Tox21).",
        "entree": "Le modèle Phase 1 + données Tox21 étiquetées.",
        "sortie": "Un modèle qui prédit la toxicité + un verdict de sécurité.",
        "analogie": "Se spécialiser en toxicologie après avoir appris à lire.",
    },
    {
        "id": "Phase 3",
        "nom": "Multi-propriétés",
        "but": "Ajouter efficacité (VIH), solubilité, lipophilie, BBB, etc.",
        "entree": "Le meilleur modèle Phase 2 (warm-start) + datasets multiples.",
        "sortie": "Le modèle complet utilisé pour la Recherche et le Criblage.",
        "analogie": "Devenir médecin généraliste : juger plusieurs aspects à la fois.",
    },
]

# ──────────────────────────────────────────────────────────────────────
# Lexique : chaque mot-clé du projet, expliqué simplement + exemple
# ──────────────────────────────────────────────────────────────────────
GLOSSARY = [
    {
        "group": "🧠 Intelligence artificielle & entraînement",
        "terms": [
            {"term": "GNN (Graph Neural Network)",
             "def": "Réseau de neurones qui travaille sur des graphes plutôt que sur du texte ou des images.",
             "ex": "Une molécule devient un graphe : atomes = points, liaisons = traits. Le GNN « lit » cette forme."},
            {"term": "Epoch",
             "def": "Un passage complet du modèle sur tout le jeu d'entraînement.",
             "ex": "20 epochs = le modèle a revu l'ensemble des données 20 fois."},
            {"term": "Loss (perte)",
             "def": "Mesure de l'erreur du modèle ; on l'entraîne à la faire baisser.",
             "ex": "Loss qui passe de 1.3 à 0.4 → le modèle s'améliore."},
            {"term": "Learning rate (taux d'apprentissage)",
             "def": "Vitesse à laquelle le modèle ajuste ses poids à chaque pas.",
             "ex": "Trop grand = instable ; trop petit = apprentissage très lent."},
            {"term": "Fine-tuning (affinage)",
             "def": "Adapter un modèle déjà pré-entraîné à une tâche précise.",
             "ex": "Partir du modèle Phase 1 (MGM) pour le spécialiser sur la toxicité."},
            {"term": "MGM (Masked Graph Modeling)",
             "def": "Pré-entraînement où l'on masque des atomes et le modèle doit les deviner.",
             "ex": "Apprend la logique chimique sans avoir besoin d'étiquettes."},
            {"term": "Checkpoint (.pth)",
             "def": "Sauvegarde sur disque de l'état (poids) du modèle à un instant donné.",
             "ex": "best_model.pth = la meilleure version retenue par la supervision."},
            {"term": "EMA (moyenne mobile exponentielle)",
             "def": "Version lissée des poids qui stabilise le modèle d'une epoch à l'autre.",
             "ex": "Réduit le bruit et donne souvent une meilleure validation."},
            {"term": "Warm-start (démarrage à chaud)",
             "def": "Reprendre le meilleur modèle précédent au lieu de repartir de zéro.",
             "ex": "La Phase 3 démarre à partir du meilleur modèle de la Phase 2."},
            {"term": "Overfitting (surapprentissage)",
             "def": "Le modèle mémorise les données vues au lieu de généraliser.",
             "ex": "AUC train 0.95 mais AUC validation 0.60 → il triche par cœur."},
            {"term": "Early stopping (arrêt anticipé)",
             "def": "Arrêter l'entraînement quand le modèle ne progresse plus.",
             "ex": "Stop à l'epoch 12 si le score clinique ne s'améliore plus."},
            {"term": "Attention",
             "def": "Mécanisme qui fait peser davantage les parties importantes du graphe.",
             "ex": "Le modèle se concentre sur les atomes liés à la toxicité."},
            {"term": "Score clinique (supervision)",
             "def": "Note combinée orientée sécurité qui choisit la meilleure epoch (⭐).",
             "ex": "0.45·AUC + 0.30·sensibilité + 0.25·(1−FNR) − 0.05·n_danger."},
            {"term": "SSE (Server-Sent Events)",
             "def": "Flux temps réel du serveur vers le navigateur.",
             "ex": "Les courbes d'entraînement se remplissent en direct, sans recharger la page."},
        ],
    },
    {
        "group": "🧪 Chimie & molécules",
        "terms": [
            {"term": "SMILES",
             "def": "Texte court qui encode la structure d'une molécule.",
             "ex": "« CCO » = éthanol ; « CC(=O)Oc1ccccc1C(=O)O » = aspirine."},
            {"term": "Descripteur moléculaire",
             "def": "Nombre calculé qui résume une caractéristique de la molécule.",
             "ex": "Poids moléculaire, nombre de cycles, surface polaire…"},
            {"term": "MW (poids moléculaire)",
             "def": "Masse de la molécule en g/mol.",
             "ex": "Eau = 18 ; aspirine ≈ 180. Lipinski recommande ≤ 500."},
            {"term": "LogP (lipophilie)",
             "def": "Affinité de la molécule pour l'huile vs l'eau.",
             "ex": "2–5 = bonne absorption ; > 5 = risque d'accumulation."},
            {"term": "LogS (solubilité)",
             "def": "À quel point la molécule se dissout dans l'eau.",
             "ex": "Plus c'est haut, mieux le médicament se dissout."},
            {"term": "TPSA (surface polaire)",
             "def": "Surface des atomes polaires ; liée à l'absorption.",
             "ex": "> 140 Å² → mauvaise absorption orale en général."},
            {"term": "HBD / HBA",
             "def": "Donneurs / accepteurs de liaisons hydrogène.",
             "ex": "Comptés par la règle de Lipinski (HBD ≤ 5, HBA ≤ 10)."},
            {"term": "QED",
             "def": "Score de 0 à 1 de « ressemblance à un médicament ».",
             "ex": "0.9 = très drug-like ; 0.2 = peu probable comme médicament oral."},
            {"term": "Règle de Lipinski (règle des 5)",
             "def": "Critères d'absorption orale : MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.",
             "ex": "L'aspirine respecte les 4 → bon profil oral."},
            {"term": "Drug-likeness",
             "def": "À quel point une molécule ressemble à un médicament viable.",
             "ex": "Combine MW, LogP, QED, Lipinski en une appréciation globale."},
            {"term": "Scaffold (squelette)",
             "def": "Cœur structurel commun à une famille de molécules.",
             "ex": "Sert à séparer honnêtement train/test (scaffold split)."},
        ],
    },
    {
        "group": "🏥 Toxicologie & métriques cliniques",
        "terms": [
            {"term": "Tox21",
             "def": "Jeu de données public de 12 tests de toxicité de référence.",
             "ex": "Récepteurs nucléaires (NR-*) et réponse au stress (SR-*)."},
            {"term": "Endpoint (cible toxicologique)",
             "def": "Un test biologique précis prédit par le modèle.",
             "ex": "NR-AR = récepteur aux androgènes ; SR-p53 = stress/ADN."},
            {"term": "ROC-AUC",
             "def": "Capacité à séparer toxique / non-toxique, tous seuils confondus.",
             "ex": "1.0 = parfait, 0.5 = hasard. Cible ≥ 0.85."},
            {"term": "PR-AUC",
             "def": "AUC précision/rappel, pertinente quand les toxiques sont rares.",
             "ex": "Plus fiable que ROC-AUC sur classes déséquilibrées."},
            {"term": "Sensibilité (rappel)",
             "def": "Proportion de composés RÉELLEMENT toxiques bien détectés.",
             "ex": "80 % = 8 toxiques sur 10 repérés. Priorité en sécurité."},
            {"term": "Spécificité",
             "def": "Proportion de non-toxiques correctement classés.",
             "ex": "Haute spécificité = peu de fausses alertes."},
            {"term": "FNR (taux de faux négatifs)",
             "def": "= 1 − sensibilité. Proportion de toxiques MANQUÉS.",
             "ex": "L'erreur la plus grave (toxique dit « sûr ») ; plafonné < 30 %."},
            {"term": "Faux négatif",
             "def": "Une molécule toxique classée « sûre » par erreur.",
             "ex": "Le danger n°1 en clinique : on veut le minimiser."},
            {"term": "ECE (calibration)",
             "def": "Écart entre la confiance annoncée et la justesse réelle.",
             "ex": "ECE bas → quand le modèle dit « 90 % sûr », c'est crédible."},
            {"term": "n_danger",
             "def": "Nombre d'endpoints en zone de danger sur un epoch.",
             "ex": "0 = bon ; plusieurs → modèle non déployable."},
            {"term": "Verdict de sécurité",
             "def": "Conclusion synthétique : OK / À surveiller / DANGER.",
             "ex": "DANGER = trop de toxiques manqués → non déployable."},
            {"term": "Toxicophore",
             "def": "Motif chimique connu pour être associé à une toxicité.",
             "ex": "Un nitro aromatique signale un risque mutagène (test d'Ames)."},
            {"term": "Test d'Ames (mutagénicité)",
             "def": "Test de référence du pouvoir mutagène (dommage à l'ADN).",
             "ex": "Détecté ici par des toxicophores (nitro, amine aromatique…)."},
            {"term": "hERG (cardiotoxicité)",
             "def": "Canal potassique cardiaque ; son blocage cause des arythmies.",
             "ex": "Amine basique + forte lipophilie = pharmacophore hERG à risque."},
            {"term": "DILI (hépatotoxicité)",
             "def": "Atteinte du foie induite par un médicament.",
             "ex": "Le paracétamol en surdose en est l'exemple classique."},
            {"term": "Domaine d'applicabilité",
             "def": "Zone chimique où les prédictions du modèle sont fiables.",
             "ex": "Une molécule très différente de l'entraînement = extrapolation risquée."},
        ],
    },
    {
        "group": "🦠 VIH, efficacité & criblage",
        "terms": [
            {"term": "VIH",
             "def": "Virus de l'immunodéficience humaine, responsable du SIDA.",
             "ex": "Cible thérapeutique principale du projet."},
            {"term": "Antirétroviral (ARV)",
             "def": "Médicament qui bloque la réplication du VIH.",
             "ex": "Zidovudine (AZT), Lamivudine (3TC), Efavirenz."},
            {"term": "Criblage virtuel",
             "def": "Trier par ordinateur une grande bibliothèque pour repérer des candidats.",
             "ex": "Alternative rapide au criblage haut débit (HTS) en laboratoire."},
            {"term": "EC50 / IC50",
             "def": "Concentration nécessaire pour 50 % d'effet / d'inhibition.",
             "ex": "Plus la valeur est basse, plus la molécule est puissante."},
            {"term": "Synergie",
             "def": "Deux molécules plus efficaces ensemble que séparément.",
             "ex": "Principe des trithérapies anti-VIH."},
            {"term": "Biodisponibilité",
             "def": "Fraction du médicament qui atteint réellement la circulation.",
             "ex": "Notée %F ; faible = peu de principe actif disponible."},
            {"term": "Stabilité métabolique",
             "def": "Résistance de la molécule à la dégradation par le foie.",
             "ex": "Trop instable = effet trop court."},
        ],
    },
    {
        "group": "📊 Données & validation",
        "terms": [
            {"term": "Train / Validation / Test",
             "def": "Jeux pour apprendre / régler / évaluer honnêtement.",
             "ex": "On ne juge JAMAIS le modèle sur les données qu'il a apprises."},
            {"term": "Scaffold split",
             "def": "Séparer les jeux par squelette moléculaire pour éviter de tricher.",
             "ex": "Empêche des molécules quasi identiques d'être des deux côtés."},
            {"term": "BBBP",
             "def": "Dataset : passage de la barrière hémato-encéphalique.",
             "ex": "Important pour les médicaments visant le cerveau."},
            {"term": "ClinTox",
             "def": "Dataset : toxicité clinique (médicaments retirés/échoués).",
             "ex": "Apprend les signatures d'échec en clinique."},
            {"term": "ESOL (Delaney)",
             "def": "Dataset : solubilité aqueuse mesurée.",
             "ex": "Sert à prédire LogS."},
            {"term": "LIPO",
             "def": "Dataset : lipophilie (LogP) mesurée.",
             "ex": "~4 200 molécules de référence."},
            {"term": "HIV (dataset)",
             "def": "Dataset NCI : activité anti-VIH de ~41 000 molécules.",
             "ex": "Base de l'apprentissage d'efficacité antivirale."},
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────
# Bibliothèques de molécules de référence (SMILES validés à l'usage)
# ──────────────────────────────────────────────────────────────────────
LIBRARIES = {
    # ── Anti-VIH, toutes classes confondues (clé historique, gardée pour le chatbot) ──
    "hiv_reference": {
        "label": "Antirétroviraux de référence (VIH, toutes classes)",
        "note": "Panel représentatif des grandes classes d'ARV — étalonne la prédiction d'efficacité.",
        "molecules": [
            {"name": "Zidovudine (AZT, NRTI)", "smiles": "CC1=CN(C2CC(N=[N+]=[N-])C(CO)O2)C(=O)NC1=O"},
            {"name": "Lamivudine (3TC, NRTI)", "smiles": "OCC1OC(N2C=CC(N)=NC2=O)CS1"},
            {"name": "Tenofovir (NtRTI)", "smiles": "Nc1ncnc2c1ncn2CC(C)OCP(=O)(O)O"},
            {"name": "Efavirenz (NNRTI)", "smiles": "FC(F)(F)C1(C#CC2CC2)OC(=O)Nc2ccc(Cl)cc21"},
            {"name": "Rilpivirine (NNRTI)", "smiles": "Cc1cc(/C=C/C#N)cc(C)c1Nc1ccnc(Nc2ccc(C#N)cc2)n1"},
            {"name": "Darunavir (PI)", "smiles": "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1COC2OCCC12)S(=O)(=O)c1ccc(N)cc1"},
            {"name": "Dolutegravir (INSTI)", "smiles": "CC1CCOC2CN3C=C(C(=O)NCc4ccc(F)cc4F)C(=O)C(O)=C3C(=O)N12"},
            {"name": "Maraviroc (entrée/CCR5)", "smiles": "CC(C)C1=NN=C(C)N1C1CCC(NC(C2CC2)c2ccccc2)CC1CCN1CCC(F)(F)CC1"},
        ],
    },
    "hiv_nrti": {
        "label": "VIH — Inhibiteurs nucléosidiques de la RT (NRTI)",
        "note": "Bloquent la transcriptase inverse en s'incorporant à l'ADN viral.",
        "molecules": [
            {"name": "Zidovudine (AZT)", "smiles": "CC1=CN(C2CC(N=[N+]=[N-])C(CO)O2)C(=O)NC1=O"},
            {"name": "Lamivudine (3TC)", "smiles": "OCC1OC(N2C=CC(N)=NC2=O)CS1"},
            {"name": "Stavudine (d4T)", "smiles": "CC1=CN(C2C=CC(CO)O2)C(=O)NC1=O"},
            {"name": "Emtricitabine (FTC)", "smiles": "OCC1OC(N2C=C(F)C(N)=NC2=O)CS1"},
            {"name": "Didanosine (ddI)", "smiles": "C1CC(OC1CO)N2C=NC3=C2N=CNC3=O"},
            {"name": "Abacavir", "smiles": "Nc1nc(NC2CC2)c2ncn(C3CC(CO)C=C3)c2n1"},
            {"name": "Tenofovir", "smiles": "Nc1ncnc2c1ncn2CC(C)OCP(=O)(O)O"},
        ],
    },
    "hiv_nnrti": {
        "label": "VIH — Inhibiteurs non-nucléosidiques de la RT (NNRTI)",
        "note": "Se fixent directement sur la transcriptase inverse et la bloquent.",
        "molecules": [
            {"name": "Efavirenz", "smiles": "FC(F)(F)C1(C#CC2CC2)OC(=O)Nc2ccc(Cl)cc21"},
            {"name": "Nevirapine", "smiles": "Cc1ccnc2c1NC(=O)c1cccnc1N2C1CC1"},
            {"name": "Rilpivirine", "smiles": "Cc1cc(/C=C/C#N)cc(C)c1Nc1ccnc(Nc2ccc(C#N)cc2)n1"},
            {"name": "Etravirine", "smiles": "Cc1cc(C#N)cc(C)c1Oc1nc(N)nc(Nc2ccc(C#N)cc2)c1Br"},
            {"name": "Delavirdine", "smiles": "CC(C)Nc1cccnc1N1CCN(C(=O)c2cc3cc(NS(C)(=O)=O)ccc3[nH]2)CC1"},
        ],
    },
    "hiv_pi": {
        "label": "VIH — Inhibiteurs de protéase (PI)",
        "note": "Empêchent la maturation des protéines virales (clivage par la protéase).",
        "molecules": [
            {"name": "Ritonavir", "smiles": "CC(C)c1nc(CN(C)C(=O)NC(C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1"},
            {"name": "Lopinavir", "smiles": "CC(C)c1nc(CN(C)C(=O)NC(C(C)C)C(=O)NC(Cc2ccccc2)CC(O)C(Cc2ccccc2)NC(=O)COc2c(C)cccc2C)cs1"},
            {"name": "Atazanavir", "smiles": "COC(=O)NC(C(=O)NC(Cc1ccc(-c2ccccn2)cc1)C(O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)C(NC(=O)OC)C(C)(C)C)C(C)(C)C"},
            {"name": "Darunavir", "smiles": "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1COC2OCCC12)S(=O)(=O)c1ccc(N)cc1"},
            {"name": "Saquinavir", "smiles": "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(Cc1ccccc1)NC(=O)C(CC(N)=O)NC(=O)c1ccc2ccccc2n1"},
            {"name": "Indinavir", "smiles": "CC(C)(C)NC(=O)C1CN(Cc2cccnc2)CCN1CC(O)CC(Cc1ccccc1)C(=O)NC1c2ccccc2CC1O"},
        ],
    },
    "hiv_insti": {
        "label": "VIH — Inhibiteurs d'intégrase (INSTI)",
        "note": "Bloquent l'intégration de l'ADN viral dans le génome de la cellule hôte.",
        "molecules": [
            {"name": "Raltegravir", "smiles": "CC1=NN=C(O1)C(C)(C)NC(=O)c1nc(C(=O)NCc2ccc(F)cc2)c(O)c(=O)n1C"},
            {"name": "Dolutegravir", "smiles": "CC1CCOC2CN3C=C(C(=O)NCc4ccc(F)cc4F)C(=O)C(O)=C3C(=O)N12"},
            {"name": "Bictegravir", "smiles": "OC1=C2N(CC3OCC(F)(F)C3)C=C1C(=O)NC1c3ccc(F)cc3CCC12"},
        ],
    },
    # ── Médicaments courants (clé historique, gardée pour le chatbot) ──
    "reference_drugs": {
        "label": "Médicaments courants (référence)",
        "note": "Petites molécules très connues, pour tester / comparer rapidement.",
        "molecules": [
            {"name": "Paracétamol", "smiles": "CC(=O)Nc1ccc(O)cc1"},
            {"name": "Aspirine", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
            {"name": "Caféine", "smiles": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"},
            {"name": "Ibuprofène", "smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"},
            {"name": "Pénicilline G", "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"},
            {"name": "Amoxicilline", "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"},
            {"name": "Métformine", "smiles": "CN(C)C(=N)NC(N)=N"},
            {"name": "Oméprazole", "smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"},
            {"name": "Atorvastatine", "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"},
            {"name": "Warfarine", "smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"},
            {"name": "Diazépam", "smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"},
            {"name": "Loratadine", "smiles": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1"},
            {"name": "Ciprofloxacine", "smiles": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"},
        ],
    },
    "natural_products": {
        "label": "Produits naturels (phytochimie)",
        "note": "Composés naturels souvent étudiés en pharmacologie — bonne diversité de squelettes.",
        "molecules": [
            {"name": "Curcumine", "smiles": "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"},
            {"name": "Quercétine", "smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"},
            {"name": "Resvératrol", "smiles": "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1"},
            {"name": "EGCG (thé vert)", "smiles": "O=C(OC1Cc2c(O)cc(O)cc2OC1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"},
            {"name": "Génistéine", "smiles": "O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12"},
            {"name": "Capsaïcine", "smiles": "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O"},
            {"name": "Menthol", "smiles": "CC(C)C1CCC(C)CC1O"},
            {"name": "Nicotine", "smiles": "CN1CCCC1c1cccnc1"},
        ],
    },
    "repurposing": {
        "label": "Repositionnement (antiviraux / candidats)",
        "note": "Médicaments existants explorés pour de nouvelles indications antivirales.",
        "molecules": [
            {"name": "Chloroquine", "smiles": "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"},
            {"name": "Hydroxychloroquine", "smiles": "CCN(CCO)CCCC(C)Nc1ccnc2cc(Cl)ccc12"},
            {"name": "Remdesivir", "smiles": "CCC(CC)COC(=O)C(C)NP(=O)(OCC1OC(n2ccc3c(N)ncnc32)(C#N)C(O)C1O)Oc1ccccc1"},
            {"name": "Sildénafil", "smiles": "CCCc1nn(C)c2c1nc(-c1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC)[nH]c2=O"},
        ],
    },
    "toxic_controls": {
        "label": "Contrôles toxiques (positifs)",
        "note": "Composés à toxicité connue — contrôles pour vérifier que le modèle DÉTECTE le danger.",
        "molecules": [
            {"name": "Thalidomide", "smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1"},
            {"name": "Aflatoxine B1", "smiles": "COc1cc2c(c3c1OC1C=COC13)C(=O)CC1(O)CCC=C1O2"},
            {"name": "Benzène", "smiles": "c1ccccc1"},
            {"name": "Formaldéhyde", "smiles": "C=O"},
            {"name": "Acrylamide", "smiles": "C=CC(N)=O"},
        ],
    },
}
