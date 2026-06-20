# 📊 Tableau de bord Panacée

Dashboard de suivi d'entraînement + **sécurité médicale**, en **temps réel**.

## Installation
```bash
pip install -r dashboard/requirements.txt
```

## Lancement
```bash
cd Projet_Panacee
streamlit run dashboard/app.py
```
Le navigateur s'ouvre sur `http://localhost:8501`.

## Comment ça marche
Pendant l'entraînement, la Phase 2 écrit un point par epoch dans
`checkpoints/phase2/live_metrics.jsonl` (loss, AUC, **sensibilité**, **FNR**,
nombre d'endpoints en danger, AUC par tâche). Le dashboard lit/tail ce fichier
et se rafraîchit automatiquement → **monitoring temps réel** sans réseau.

## Onglets
| Onglet | Contenu |
|--------|---------|
| 📈 Évolution | Courbes loss / AUC / sécurité (sensibilité & FNR) dans le temps, détection de surapprentissage |
| 🏥 Métriques cliniques | Par endpoint : sensibilité, spécificité, **FNR**, précision, F1, ROC-AUC, PR-AUC, calibration (ECE) — depuis un checkpoint + CSV de validation |
| 🚨 Sécurité | Alertes **DANGER / WARN** : un faux négatif = molécule toxique prédite « sûre » |
| 🔬 Comparaison | Attendu vs obtenu + comparaison de plusieurs runs |

## Sécurité (pourquoi le FNR ?)
En toxicologie, l'erreur grave est le **faux négatif** : laisser passer un composé
toxique en le déclarant sûr. Le dashboard met donc en avant la **sensibilité**
(rappel sur les toxiques) et le **FNR**, avec un barème :
- **DANGER** : FNR ≥ 50 % ou sensibilité < 50 % ou AUC < 0.60
- **WARN** : FNR ≥ 30 % ou AUC < 0.70

## Workflow Kaggle
1. Entraîne sur Kaggle → télécharge `checkpoints/phase2/` (contient `live_metrics.jsonl` + `best_toxicity_model.pth`).
2. Lance le dashboard en local en pointant sur ce dossier.
