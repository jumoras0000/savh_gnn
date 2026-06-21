# 🧬 Panacée — GNN souverain pour la découverte de médicaments

Pipeline complet de **réseau de neurones sur graphes (GNN)** pour prédire la
**toxicité**, l'**efficacité anti‑VIH** et les propriétés **ADME** de molécules,
avec un **tableau de bord web temps réel** et un **assistant conversationnel**.

> ⚠️ Outil d'aide à la décision **in‑silico**. Toute conclusion doit être validée
> en laboratoire (voir l'équivalence laboratoire dans l'onglet Guide / le manuel).

---

## Architecture (3 phases)

```
Phase 1 — Pré-entraînement MGM (ZINC)      → encodeur moléculaire souverain
Phase 2 — Fine-tuning toxicité (Tox21)     → 12 endpoints toxicologiques
Phase 3 — Multi-propriétés + IA raisonnement → efficacité anti-VIH, ADME,
                                               synergie/doses de combinaisons
```

Encodeur : message passing edge‑aware (GATv2), résidus, triple pooling.
Détails techniques : voir `src/` et les guides ci‑dessous.

---

## Installation

```bash
cd Projet_Panacee
# 1) PyTorch (GPU CUDA 12.1 ou CPU) — voir requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
# 2) le reste
pip install -r requirements.txt
# 3) tableau de bord web (léger)
pip install -r webapp/requirements.txt
# 4) (optionnel) chatbot synchronisé avec Claude
pip install anthropic        # + export ANTHROPIC_API_KEY=...
```

---

## Démarrage rapide

### Tableau de bord web

```bash
python -m webapp.run            # http://127.0.0.1:8000
python -m webapp.run --demo     # démo temps réel sans GPU
```

Onglets : **Évolution · Métriques cliniques · Sécurité · Comparaison ·
Entraînement · Recherche · Criblage VIH · Assistant · Guide** — plus le **mode
clair/sombre** (bouton 🌙/☀️). Documentation : [webapp/README.md](webapp/README.md).

### Entraînement (UI onglet Entraînement, ou CLI / Kaggle)

```bash
python run_phase2.py --download --epochs 60     # toxicité
python run_phase3.py --download --epochs 80     # + efficacité anti-VIH
# (optionnel) python run_phase1.py --download --epochs 50
```

Suivi **temps réel** dans l'onglet Évolution (même depuis Kaggle via push HTTP —
voir [webapp/README.md](webapp/README.md#temps-réel-depuis-kaggle)).

### Découvrir des médicaments anti‑VIH

Guide pas‑à‑pas complet (A→Z, critères « médicament utilisable », équivalence
laboratoire, dépannage) : **[MANUEL_VIH.md](MANUEL_VIH.md)**.

En résumé : entraîne Phase 2 puis Phase 3 → onglet **Criblage VIH** (objectif
*efficacité anti‑VIH*) pour classer une bibliothèque → onglet **Recherche** pour
analyser les meilleurs candidats (propriétés + risque + combinaisons).

### Analyse en ligne de commande

```bash
python predict_molecules.py --smiles "CC(=O)Nc1ccc(O)cc1" --predict_only
python predict_molecules.py --smiles_file candidats.csv --report rapport.json
```

---

## Tests

```bash
pip install pytest httpx
python -m pytest tests/test_webapp.py -v          # interface web (37 tests)
python -m pytest tests/test_bugfixes.py tests/test_clinical_metrics.py -v
```

---

## Documentation

| Document | Contenu |
|----------|---------|
| [MANUEL_VIH.md](MANUEL_VIH.md) | Découverte de médicaments anti‑VIH de A à Z |
| [webapp/README.md](webapp/README.md) | Tableau de bord, API, temps réel Kaggle, chatbot, mode clair |
| [GUIDE_VALIDATION_ACADEMIQUE.md](GUIDE_VALIDATION_ACADEMIQUE.md) | Validation scientifique (CV scaffold, calibration, reproductibilité) |
| [FIX_DEEPCHEM_WINDOWS.md](FIX_DEEPCHEM_WINDOWS.md) | Dépannage DeepChem sous Windows |

---

## Structure

```
Projet_Panacee/
├── src/                  modèles, préprocessing, entraînement, validation, IA
├── webapp/               tableau de bord (backend Starlette + frontend autonome)
├── tests/                tests unitaires & d'intégration
├── run_phase1/2/3.py     lanceurs d'entraînement
├── predict_molecules.py  inférence / analyse de molécules
└── MANUEL_VIH.md         guide découverte anti-VIH
```
