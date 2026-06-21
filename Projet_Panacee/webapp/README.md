# 🧬 Panacée — Tableau de bord web (console de signes vitaux)

Interface web **complète (front + backend)** pour suivre l'entraînement du GNN
**avant / pendant / après**, avec des métriques cliniques sensibles et une
signalisation de danger. Le suivi **temps réel** se fait par SSE (Server-Sent
Events) : le backend lit `live_metrics.jsonl` écrit par l'entraînement et pousse
chaque epoch au navigateur, sans WebSocket ni couplage réseau.

## Pourquoi cette stack ?

| Couche   | Choix | Raison |
|----------|-------|--------|
| Backend  | **Starlette + uvicorn** | Déjà présents dans l'env, async SSE propre, zéro dépendance lourde ajoutée. Pas besoin de FastAPI. |
| Frontend | **HTML/CSS/JS autonome** | Graphiques dessinés sur `<canvas>` faits main → **aucune dépendance CDN**, fonctionne hors-ligne et sur Kaggle. |
| Temps réel | **SSE** (`/api/stream`) | Le serveur *tail* le `.jsonl` et pousse les nouveaux epochs. Reconnexion automatique côté navigateur. |

## Lancement

```bash
cd Projet_Panacee
pip install -r webapp/requirements.txt        # starlette + uvicorn (déjà là le plus souvent)

# Suivi d'un entraînement (lance-le depuis l'onglet 🎛️ Entraînement ou en CLI) :
python run_phase2.py --download               # écrit checkpoints/phase2/live_metrics.jsonl
python -m webapp.run                           # ouvre http://127.0.0.1:8000
```

Puis ouvre **http://127.0.0.1:8000**.

Options : `--host 0.0.0.0 --port 8080`, `--root <dossier_checkpoints>`.

## Ce que montre le tableau de bord

- **Verdict clinique** (haut de page) : conclusion `🟢 OK / 🟠 À surveiller / 🔴 NON déployable`
  avec les raisons (FNR, sensibilité, endpoints en danger).
- **KPI temps réel** : epoch + progression + ETA, ROC-AUC, sensibilité, FNR, endpoints en DANGER (avec sparklines).
- **ECG animé** : bat plus vite quand l'entraînement est *en cours*, vire au rouge en cas de DANGER.
- **Évolution** : courbes loss / AUC (train vs val) et **signes vitaux de sécurité** (sensibilité & FNR) avec lignes de cible et zone de danger, plus un panneau **Observation & risque** (lecture automatique des métriques).
- **Métriques cliniques** : tableau par endpoint toxicologique (sensibilité, spécificité, FNR, précision, F1, ROC-AUC, PR-AUC, ECE). Les checkpoints et CSV sont **listés automatiquement** (sélecteurs) — plus besoin de saisir un chemin — et on peut **importer un fichier** (.pth / .csv).
- **🎛️ Entraînement** : lance / arrête une phase (1, 2 ou 3) directement depuis l'UI, avec console de logs et statut. Le suivi temps réel bascule automatiquement sur le run lancé.
- **🧪 Recherche** : analyse de **molécules réelles** (SMILES) → **structure 2D**, **descripteurs RDKit** (MW, LogP, TPSA, HBD/HBA, QED, Lipinski) *toujours disponibles sans modèle*, puis toxicité / efficacité / ADME selon le modèle dispo. **Mode adaptatif** : Phase 3 (complet) → Phase 2 (toxicité seule) → descripteurs seuls. Évaluation de **risque** par molécule + **export JSON**. Combinaison → synergie / doses / score (MolecularReasoner, Phase 3).
- **🧬 Criblage VIH** : *virtual screening* — classe une bibliothèque (de référence, collée, ou importée) par **efficacité anti-VIH** (Phase 3), **sécurité** (Phase 2-3) ou **drug-likeness/QED** (sans modèle). Bibliothèques intégrées (antirétroviraux de référence, médicaments courants). **Export CSV**. Équivalent in-silico d'un criblage à haut débit (HTS).
- **ℹ️ Guide** enrichi : catalogue complet des capacités + **équivalence laboratoire** de chaque analyse (ce que ça représente « à la paillasse »).
- **Sécurité** : alertes DANGER/WARN triées par gravité + barème.
- **Comparaison** : *attendu vs obtenu*, ROC-AUC par endpoint vs cible, et comparaison de tous les runs.
- **ℹ️ Guide** : explication de chaque métrique, lecture du risque, flux de travail recommandé.

### Suivi multi-phases
Les trois phases écrivent désormais `live_metrics.jsonl` :
- **Phase 1 (MGM)** : `train_loss` / `val_loss` (reconstruction — pas d'AUC/toxicité).
- **Phase 2 (toxicité)** : AUC, sensibilité, FNR, dangers, per-endpoint.
- **Phase 3 (multi-propriétés)** : pertes + AUC/R² par propriété + sécurité issue de la tête toxicité.
Le frontend s'adapte aux clés présentes (l'onglet Sécurité reste pertinent là où la toxicité est évaluée).

## API

| Méthode | Route | Rôle |
|---------|-------|------|
| GET  | `/`                   | Frontend (SPA) |
| GET  | `/api/health`         | Ping |
| GET  | `/api/config`         | Cibles attendues + seuils de danger |
| GET  | `/api/runs`           | Liste des runs (résumés + statut) |
| GET  | `/api/run?id=<id>`    | Détail complet d'un run |
| GET  | `/api/compare`        | Comparaison de tous les runs |
| GET  | `/api/files`          | Checkpoints (.pth) et CSV détectés (sélecteurs) |
| POST | `/api/upload?name=…`  | Import de fichier (corps brut : .pth/.csv/.smi/.txt) |
| POST | `/api/evaluate`       | Métriques cliniques d'un checkpoint (`{checkpoint, val_csv}`) |
| GET  | `/api/train/status`   | Statut de l'entraînement (état, pid, logs) |
| POST | `/api/train/start`    | Lance une phase (`{phase, epochs, max_molecules, …}`) |
| POST | `/api/train/stop`     | Arrête l'entraînement en cours |
| POST | `/api/predict`        | Propriétés + risque + descripteurs (`{smiles, checkpoint?}`, mode adaptatif) |
| POST | `/api/combo`          | Synergie / doses d'une combinaison (`{smiles[], checkpoint?}`) |
| POST | `/api/descriptors`    | Descripteurs RDKit (sans modèle) (`{smiles}`) |
| GET  | `/api/depict?smiles=` | Structure 2D (SVG) |
| GET  | `/api/libraries`      | Bibliothèques de molécules de référence |
| POST | `/api/screen`         | Criblage virtuel (`{library` ou `smiles, objective, checkpoint?}`) |
| GET  | `/api/capabilities`   | Catalogue des capacités + équivalence laboratoire |
| POST | `/api/ingest?run=<id>`| Réception d'un point de métriques distant (Kaggle → dashboard) |
| POST | `/api/chat`           | Chatbot (Claude si clé, sinon assistant local) (`{messages}`) |
| GET  | `/api/chat/status`    | Synchronisation Claude active ? + modèle |
| GET  | `/api/stream?id=<id>` | Flux **SSE** temps réel (snapshot + epochs + statut) |

## Temps réel depuis Kaggle

Le dashboard tourne chez toi, l'entraînement sur Kaggle, et le suivi reste **temps
réel** : l'entraînement **pousse** chaque point vers ton dashboard (exposé via un
tunnel). Dans le notebook Kaggle, **avant** de lancer l'entraînement :

```python
import os
os.environ["PANACEE_PUSH_URL"]   = "https://TON-TUNNEL.ngrok.io"  # ton dashboard exposé
os.environ["PANACEE_PUSH_RUN"]   = "phase2"                        # id du run
os.environ["PANACEE_PUSH_TOKEN"] = "secret"                        # optionnel
# puis : python run_phase2.py --download --epochs 60
```

Côté dashboard, protège l'ingestion avec le même secret :
`PANACEE_INGEST_TOKEN=secret python -m webapp.run --host 0.0.0.0`. Chaque epoch
arrive via `POST /api/ingest` et apparaît dans l'onglet Évolution comme un run local.
Sans tunnel, l'alternative reste l'**import** du `live_metrics.jsonl` téléchargé.

## Assistant (chatbot)

Onglet **💬 Assistant** : converse avec le modèle GNN. Avec **Claude
(claude-opus-4-8)**, il orchestre les outils du modèle (toxicité, efficacité VIH,
descripteurs, criblage, synergie, statut) pour des analyses avancées ; sans clé,
un **assistant local** appelle quand même ces outils (détection SMILES + intention).

Fonctionnalités :
- **Multi-conversations** : créer / changer / renommer / supprimer (panneau latéral).
- **Recherche** dans l'historique des chats.
- **Streaming token-par-token** des réponses.
- **Images** : joindre une image (analyse **vision** via Claude) ; **structures 2D**
  des molécules **générées automatiquement** dans la réponse.
- **Export** d'une conversation précise (JSON).
- **Clé API Anthropic** : se met directement dans l'UI (champ dédié, stockée
  localement) ou via `ANTHROPIC_API_KEY`. Installe le SDK : `pip install anthropic`.

**Base de données** : tout l'historique (conversations, messages, réglages) est
persisté dans **SQLite** (`data/panacee.db`, ignoré par git) ; les images dans
`data/chat_images/`. Endpoints : `/api/conversations*`, `/api/chat`,
`/api/chat/stream`, `/api/settings/apikey`, `/api/chat/image`.

## Thème clair / sombre

Bouton 🌙/☀️ dans la barre — bascule clair/sombre, mémorisé (localStorage), avec
rafraîchissement des couleurs de tous les graphiques.

## Tests

```bash
cd Projet_Panacee
pip install httpx           # requis par le TestClient
python -m pytest tests/test_webapp.py -v
```

Les tests couvrent : endpoints REST, découverte/résolution de runs (avec garde
anti-traversée), verdict clinique, évaluation (mock), et le flux SSE de bout en bout.
