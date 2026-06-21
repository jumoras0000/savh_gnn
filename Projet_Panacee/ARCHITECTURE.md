# Architecture — Projet Panacée

Découverte de médicaments *in silico* : un réseau de neurones sur graphes (GNN)
apprend à lire la structure des molécules pour prédire **toxicité**, **efficacité
anti-VIH** et propriétés **ADME**, avec un tableau de bord de supervision temps réel.

> ⚠️ Outil d'aide à la décision. Toute conclusion doit être validée
> expérimentalement avant tout usage médical.

## Vue d'ensemble

```
                 SMILES (texte)                  ┌─────────────────────────┐
                      │                          │   Tableau de bord web   │
                      ▼                          │  (Starlette + SSE)      │
        ┌──────────────────────────┐            │  webapp/                │
        │  graph_builder            │            │  ├─ server.py  (ASGI)   │
        │  SMILES → graphe PyG      │            │  ├─ service.py (pur)    │
        └────────────┬─────────────┘            │  ├─ research.py (préd.) │
                     ▼                           │  ├─ cheminfo.py (RDKit) │
        ┌──────────────────────────┐            │  ├─ catalog.py (données)│
        │  MolecularEncoder (GNN)   │            │  ├─ chatbot.py (Claude) │
        │  src/models/encoder.py    │            │  └─ store.py   (SQLite) │
        └────────────┬─────────────┘            └────────────┬────────────┘
                     ▼                                        ▲
   ┌─────────────────────────────────┐         live_metrics.jsonl (tail/SSE)
   │ Têtes : toxicité / multi-prop.  │                        │
   │ + raisonneur (combinaisons)     │──── entraînement ──────┘
   └─────────────────────────────────┘     (local OU Kaggle → /api/ingest)
```

## Les trois phases d'entraînement

| Phase | Script | Rôle | Entrée → Sortie |
|-------|--------|------|-----------------|
| **1** | `run_phase1.py` → `src/training/pretrain_gnn.py` | Pré-entraînement auto-supervisé (Masked Graph Modeling) | ~250k SMILES → encodeur GNN |
| **2** | `run_phase2.py` → `src/training/finetune_toxicity.py` | Affinage toxicité (12 endpoints Tox21) | encodeur + Tox21 → modèle toxicité |
| **3** | `run_phase3.py` → `src/training/train_phase3.py` | Multi-propriétés (efficacité VIH, solubilité, LogP, BBB…) | meilleur modèle P2 (*warm-start*) → modèle complet |

La **supervision clinique** ([clinical_metrics.py](src/validation/clinical_metrics.py))
choisit la meilleure epoch via un score orienté sécurité :
`0.45·AUC + 0.30·sensibilité + 0.25·(1−FNR) − 0.05·n_danger`.

## Modules clés

### `src/`
- **models/** — `encoder.py` (GNN : attention GATv2 *edge-aware* ou MPNN), `toxicity_classifier.py`, `multi_property_head.py`, `reasoner.py` / `advanced_reasoner.py` (combinaisons & synergie).
- **preprocessing/** — `graph_builder.py` (SMILES → graphe, features normalisées, robuste aux SMILES invalides), `*_loader.py`, `scaffold_split.py` (séparation honnête train/test).
- **training/** — les 3 phases + `graphcl.py` (contraste de graphes).
- **validation/** — métriques cliniques, calibration (ECE), reproductibilité (seeds, versioning).
- **utils/** — `live_logger.py` (JSONL + push HTTP Kaggle), `safe_load.py` (chargement sûr des checkpoints), `uncertainty.py` (MC-Dropout + ensemble), `ema.py`, `gpu_manager.py`.
- **analysis/** — `combinatorial_engine.py` (`PanaceeAnalyzer` : prédiction Phase 3 + combinaisons).
- **knowledge/** — `medical_rules.py` (règles ADMET / alertes structurales), `web_search.py`.

### `webapp/` (tableau de bord)
- **server.py** — application ASGI Starlette : REST + **SSE** (flux temps réel). Endpoints clés : `/api/runs`, `/api/run`, `/api/stream`, `/api/ingest` (réception Kaggle), `/api/predict`, `/api/safety`, `/api/evaluate`, `/api/glossary`, `/api/chat`.
- **service.py** — **fonctions PURES** (sans Starlette) : découverte de runs, score clinique, verdict, suppression — c'est la couche la plus testée.
- **cheminfo.py** — analyses **sans modèle** (toujours dispo) : descripteurs RDKit, alertes structurelles (Ames/DILI/hERG), domaine d'applicabilité.
- **research.py** — prédictions par molécule (Phase 3 > Phase 2 > descripteurs), incertitude MC-Dropout.
- **catalog.py** — données pures : capacités, lexique, **bibliothèques de molécules** (SMILES validés à l'usage).
- **store.py** — base SQLite des conversations de l'assistant.
- **static/** — frontend autonome (`index.html`, `app.js`, `style.css`), aucun build.

## Temps réel & intégration Kaggle

L'entraînement écrit un point JSON par epoch dans `checkpoints/<run>/live_metrics.jsonl`
([live_logger.py](src/utils/live_logger.py)). Le dashboard *tail* ce fichier et pousse
les mises à jour par **SSE**. Pour un entraînement **distant** (Kaggle), définir
`PANACEE_PUSH_URL` (+ `PANACEE_PUSH_TOKEN`) côté Kaggle : chaque point est aussi
envoyé en POST à `/api/ingest` du dashboard (exposé via un tunnel type ngrok).
Le push est *best-effort* (n'interrompt jamais l'entraînement). Voir
[KAGGLE_GUIDE.md](../KAGGLE_GUIDE.md) et [.env.example](.env.example).

## Flux de données (résumé)

1. **Entraînement** (local/Kaggle) → `live_metrics.jsonl` (+ checkpoints `.pth`).
2. **Dashboard** lit les runs, diffuse en SSE, calcule verdicts cliniques.
3. **Recherche/Criblage** charge un checkpoint (chargement **sécurisé**) et prédit
   pour de vraies molécules, avec incertitude + alertes structurelles.
4. **Assistant** orchestre ces outils en langage naturel (avec ou sans clé Claude).
