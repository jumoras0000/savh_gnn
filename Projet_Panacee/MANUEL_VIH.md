# 🧬 Manuel A → Z — Découvrir des médicaments anti‑VIH avec Panacée

Ce guide explique, **pas à pas**, comment utiliser le modèle GNN Panacée et son
tableau de bord web pour **identifier des candidats médicaments contre le VIH**,
**vérifier que tout fonctionne**, et **juger si un candidat est exploitable**.

> ⚠️ **À lire avant tout.** Panacée est un outil d'**aide à la décision in‑silico**
> (sur ordinateur). Il *priorise* des molécules prometteuses ; il **ne remplace pas**
> les tests en laboratoire ni les essais cliniques. Un candidat « excellent » à
> l'écran reste une **hypothèse** tant qu'il n'a pas été validé expérimentalement
> (voir §6 « Équivalence laboratoire »).

---

## 0. En une phrase

Tu entraînes le modèle (Phase 2 toxicité → Phase 3 multi‑propriétés **dont
l'efficacité anti‑VIH**), tu vérifies sa qualité sur le tableau de bord, puis tu
**cribles une bibliothèque de molécules** pour faire remonter les meilleurs
candidats anti‑VIH (efficaces **et** sûrs), que tu analyses ensuite en détail.

```
Phase 1 (optionnelle)      Phase 2                 Phase 3                     Criblage + Recherche
pré‑entraînement     →     toxicité (Tox21)   →    +efficacité anti‑VIH   →    classement des candidats
(ZINC, MGM)                12 endpoints            +ADME +raisonnement IA      + analyse détaillée
```

---

## 1. Prérequis & installation

```bash
cd Projet_Panacee
pip install -r requirements.txt          # torch, torch-geometric, rdkit, deepchem, sklearn…
pip install -r webapp/requirements.txt   # starlette + uvicorn (souvent déjà présents)
```

Vérifie que tout est importable :

```bash
python -c "import torch, torch_geometric, rdkit, deepchem, sklearn; print('OK dépendances')"
```

GPU recommandé pour l'entraînement (sinon CPU, plus lent). L'entraînement lourd
se fait typiquement **sur Kaggle** ; le tableau de bord et l'analyse tournent en
local.

---

## 2. Démarrer le tableau de bord

```bash
cd Projet_Panacee
python -m webapp.run                # → http://127.0.0.1:8000
```

Ouvre **http://127.0.0.1:8000**. Tu verras les onglets :
**Évolution · Métriques cliniques · Sécurité · Comparaison · Entraînement ·
Recherche · Criblage VIH · Guide**.

> 💡 **Important :** si tu as mis à jour le code, **arrête (Ctrl+C) puis relance**
> le serveur. Un serveur resté ouvert sur une ancienne version peut renvoyer
> `404` sur de nouvelles routes (ex. `/api/libraries`). La trace affichée au
> Ctrl+C est normale (arrêt du flux temps réel), ce n'est pas une erreur.

---

## 3. Étape A — Entraîner le modèle

L'efficacité **anti‑VIH** est apprise en **Phase 3**. Il faut donc faire
**Phase 2 puis Phase 3** (la Phase 1 est optionnelle, elle améliore l'encodeur).

### Option 1 — Depuis l'interface (onglet 🎛️ Entraînement)
1. Choisis la **phase**, le nombre d'**epochs**, éventuellement *max molécules*
   (0 = tout), coche **Télécharger les données**.
2. Clique **▶ Lancer**. La console de logs et le statut s'affichent.
3. Le suivi temps réel apparaît dans l'onglet **Évolution** dès le 1ᵉʳ epoch.

Ordre conseillé :
- **Phase 2** : ~40–80 epochs (toxicité Tox21, 12 endpoints).
- **Phase 3** : ~50–100 epochs (ajoute efficacité anti‑VIH + ADME + raisonnement).

### Option 2 — En ligne de commande (ou sur Kaggle)
```bash
python run_phase2.py --download --epochs 60          # toxicité
python run_phase3.py --download --epochs 80          # + efficacité anti‑VIH
# (optionnel, avant la Phase 2) :
python run_phase1.py --download --epochs 50          # pré‑entraînement
```

Les modèles sont écrits dans `checkpoints/phase2/best_toxicity_model.pth` et
`checkpoints/phase3/panacee_phase3_complete.pth`.

> ℹ️ Pendant l'entraînement, le message « **782 molécules, 12 tâches** » est
> **normal** : c'est le **jeu de validation** (Tox21 : ~6258 en entraînement,
> ~782 en validation, ~783 en test). Ce n'est pas une erreur.

---

## 4. Étape B — Vérifier que c'est FONCTIONNEL

Avant de chercher des médicaments, assure‑toi que le modèle a **vraiment appris**.

### 4.1 Pendant l'entraînement (onglet Évolution + Sécurité)
- **Perte (loss)** train et val qui **descendent** et restent proches
  (gros écart = surapprentissage).
- **ROC‑AUC (val)** qui **monte** vers la cible (ligne pointillée 0.85).
- **FNR** (faux négatifs = toxiques manqués) qui **descend** sous la zone rouge.
- Le **verdict** en haut de page passe de 🔴/🟠 vers 🟢.

> Avec **1 seul epoch**, l'AUC est proche de 0.5 (hasard) — c'est attendu.
> Il faut plusieurs dizaines d'epochs pour une AUC correcte.

### 4.2 Valeurs repères (ce qui est « bon »)
| Indicateur | Faible / à revoir | Correct | Bon |
|---|---|---|---|
| ROC‑AUC toxicité (val) | < 0.70 | 0.75–0.85 | > 0.85 |
| Sensibilité (macro) | < 0.50 | 0.65–0.75 | > 0.75 |
| FNR (macro) | > 0.50 | 0.30–0.40 | < 0.30 |
| Endpoints en DANGER | plusieurs | 1–2 | 0 |
| AUC efficacité anti‑VIH (Phase 3) | < 0.65 | 0.70–0.78 | > 0.78 |

### 4.3 Évaluation approfondie (onglet 🏥 Métriques cliniques)
1. Sélectionne le **checkpoint** Phase 2 et le **CSV de validation**
   (`data/external/tox21/tox21_val.csv`) — listes déroulantes, pas de saisie.
2. Clique **Évaluer** → tableau par endpoint (sensibilité, spécificité, FNR,
   précision, F1, ROC‑AUC, PR‑AUC, calibration ECE) + alertes.

### 4.4 Test de bon sens avec des références connues
Onglet **🧬 Criblage VIH** → bibliothèque **« Antirétroviraux de référence »**,
objectif **Efficacité anti‑VIH** → ces médicaments connus doivent **bien se
classer**. S'ils ressortent en haut, le modèle « comprend » l'activité anti‑VIH.
*(Sinon : entraîne davantage la Phase 3, ou augmente les données.)*

✅ **Le modèle est fonctionnel si** : AUC val ≥ 0.75 (toxicité) **et** ≥ 0.70
(efficacité VIH), FNR macro < 0.30, peu/pas d'endpoints en DANGER, et les
antirétroviraux connus remontent dans le criblage.

---

## 5. Étape C — Découvrir des médicaments anti‑VIH (de A à Z)

### 5.1 Cribler une bibliothèque (onglet 🧬 Criblage VIH)
1. **Bibliothèque** : choisis « Antirétroviraux de référence », OU **colle tes
   propres SMILES** (un par ligne), OU importe un **CSV** (colonne `smiles`).
2. **Objectif** :
   - **Efficacité anti‑VIH** *(nécessite la Phase 3)* — classe par activité antivirale prédite.
   - **Sécurité** *(Phase 2 ou 3)* — classe par faible toxicité.
   - **Drug‑likeness (QED)** *(sans modèle)* — classe par « caractère médicament ».
3. Clique **🧬 Lancer le criblage** → tableau **classé** (score, MW, LogP, QED,
   Lipinski, risque). **⬇️ Exporter (CSV)** pour garder le palmarès.

👉 **Stratégie recommandée** : crible une **grande** bibliothèque par
**Efficacité anti‑VIH**, garde le **top 20–50**, puis re‑crible ce top par
**Sécurité** pour ne retenir que les molécules **efficaces ET sûres**.

### 5.2 Analyser les meilleurs candidats (onglet 🧪 Recherche)
Colle les SMILES des candidats retenus → **🔬 Analyser**. Pour chacun :
- **Structure 2D**, **descripteurs** (MW, LogP, TPSA, HBD/HBA, QED, Lipinski) —
  disponibles **même sans modèle**.
- **Prédictions** : toxicité (12 endpoints), efficacité anti‑VIH, solubilité,
  lipophilie, biodisponibilité, stabilité, drug‑likeness.
- **Observation & risque** : verdict 🟢/🟠/🔴 + explications.
- **⬇️ Exporter (JSON)** le rapport.

### 5.3 Tester des combinaisons (multithérapie — clé contre le VIH)
Dans **Recherche**, mets **plusieurs** SMILES puis **🔗 Analyser la combinaison**
*(Phase 3)* : **synergie** par paire, **doses optimales**, **score de réussite**
et **confiance**. La trithérapie étant le standard anti‑VIH, vise des
combinaisons **synergiques** et **sûres**.

---

## 6. Étape D — Le candidat est‑il UTILISABLE ?

Un candidat est **prometteur in‑silico** (à proposer pour validation) si, dans
l'onglet Recherche, il coche :

| Critère | Seuil indicatif | Où le voir |
|---|---|---|
| Efficacité anti‑VIH | ≥ 70 % | Recherche / Criblage |
| Sécurité (1 − toxicité) | ≥ 70 % | Recherche |
| Endpoints Tox21 toxiques | 0 (idéalement) | Recherche |
| Drug‑likeness / QED | QED ≥ 0.5 | Descripteurs |
| Règle de Lipinski | ✅ (≤ 1 violation) | Descripteurs |
| Biodisponibilité | ≥ 50 % | Recherche |
| Risque global | 🟢 OK | Observation & risque |
| Comparaison aux références | proche/meilleur que les antirétroviraux connus | Criblage |

> **Mais « utilisable » au sens médical** = bien plus que l'écran. Un médicament
> réel doit ensuite passer : tests **in vitro** (efficacité/toxicité en cellules),
> **in vivo** (animal), **pharmacocinétique**, puis **essais cliniques** (phases I‑III)
> et **autorisation réglementaire**. Panacée te dit **où chercher en priorité**,
> pas qu'un composé est un médicament approuvé.

### Équivalence laboratoire (onglet Guide → table complète)
| Analyse Panacée | Équivalent à la paillasse |
|---|---|
| Toxicité Tox21 | Panel d'essais in vitro (récepteurs nucléaires / stress) |
| Efficacité anti‑VIH | Essai antiviral cellulaire (EC50/IC50 sur cellules infectées) |
| Solubilité (LogS) | Solubilité cinétique (shake‑flask) |
| Lipophilie (LogP) | Partage octanol/eau mesuré |
| Biodisponibilité | Perméabilité Caco‑2 / %F in vivo |
| Stabilité métabolique | Microsomes hépatiques (t½, clairance) |
| Synergie | Damier (Bliss / Loewe), indice de combinaison |
| Dose optimale | Courbe dose‑réponse / escalade de dose |
| Criblage virtuel | Criblage à haut débit (HTS) en plaques |

---

## 7. Récapitulatif des commandes

```bash
# Tableau de bord
python -m webapp.run                       # http://127.0.0.1:8000

# Entraînement (UI onglet Entraînement, ou CLI / Kaggle)
python run_phase2.py --download --epochs 60
python run_phase3.py --download --epochs 80

# Analyse en ligne de commande (équivalent des onglets)
python predict_molecules.py --smiles "CC(=O)Nc1ccc(O)cc1" --predict_only
python predict_molecules.py --smiles_file candidats.csv --report rapport.json

# Tests de l'application web
pip install httpx pytest && python -m pytest tests/test_webapp.py -v
```

---

## 8. Dépannage

| Symptôme | Cause / solution |
|---|---|
| `404` sur `/api/libraries`, `/api/screen`… | Serveur lancé avant la mise à jour → **Ctrl+C puis relance**. |
| Trace `CancelledError` au Ctrl+C | Normal (arrêt du flux SSE). Sans gravité. |
| « 782 molécules, 12 tâches » | Jeu de **validation** (normal). Entraînement = ~6258. |
| Criblage **efficacité** → erreur 404 | Il faut un **modèle Phase 3**. Entraîne la Phase 3, ou choisis l'objectif *drug‑likeness* / *sécurité*. |
| Recherche en « mode descripteurs » | Aucun modèle détecté → entraîne Phase 2 (toxicité) ou Phase 3 (complet). |
| AUC ≈ 0.5 | Trop peu d'epochs → relance avec plus d'epochs. |
| Pas de structure 2D | SMILES invalide → vérifie la syntaxe. |

---

## 9. Pour aller plus loin

- Augmente la **bibliothèque criblée** (plus de candidats = plus de chances).
- Entraîne la **Phase 1** (pré‑entraînement) pour un meilleur encodeur.
- Réentraîne la **Phase 3** plus longtemps pour fiabiliser l'efficacité anti‑VIH.
- Exporte les palmarès (CSV/JSON) et confronte les **meilleurs candidats** à la
  littérature (PubChem/ChEMBL) avant toute validation expérimentale.

*Bon criblage — et rappelle‑toi : l'ordinateur propose, le laboratoire dispose.*
