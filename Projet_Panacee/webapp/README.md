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

# 1) Mode démo (sans GPU ni dataset) — génère un run synthétique en temps réel :
python -m webapp.run --demo

# 2) Suivi d'un vrai entraînement (lance-le dans un autre terminal) :
python run_phase2.py --download               # écrit checkpoints/phase2/live_metrics.jsonl
python -m webapp.run                           # ouvre http://127.0.0.1:8000
```

Puis ouvre **http://127.0.0.1:8000**.

Options : `--host 0.0.0.0 --port 8080`, `--root <dossier_checkpoints>`,
`--demo-epochs 60 --demo-delay 0.5`.

## Ce que montre le tableau de bord

- **Verdict clinique** (haut de page) : conclusion `🟢 OK / 🟠 À surveiller / 🔴 NON déployable`
  avec les raisons (FNR, sensibilité, endpoints en danger).
- **KPI temps réel** : epoch + progression + ETA, ROC-AUC, sensibilité, FNR, endpoints en DANGER (avec sparklines).
- **ECG animé** : bat plus vite quand l'entraînement est *en cours*, vire au rouge en cas de DANGER.
- **Évolution** : courbes loss / AUC (train vs val) et **signes vitaux de sécurité** (sensibilité & FNR) avec lignes de cible et zone de danger.
- **Métriques cliniques** : tableau par endpoint toxicologique (sensibilité, spécificité, FNR, précision, F1, ROC-AUC, PR-AUC, ECE) — via le live, ou via une **évaluation de checkpoint** (POST `/api/evaluate`).
- **Sécurité** : alertes DANGER/WARN triées par gravité + barème.
- **Comparaison** : *attendu vs obtenu*, ROC-AUC par endpoint vs cible, et comparaison de tous les runs.

## API

| Méthode | Route | Rôle |
|---------|-------|------|
| GET  | `/`                   | Frontend (SPA) |
| GET  | `/api/health`         | Ping |
| GET  | `/api/config`         | Cibles attendues + seuils de danger |
| GET  | `/api/runs`           | Liste des runs (résumés + statut) |
| GET  | `/api/run?id=<id>`    | Détail complet d'un run |
| GET  | `/api/compare`        | Comparaison de tous les runs |
| POST | `/api/evaluate`       | Métriques cliniques d'un checkpoint (`{checkpoint, val_csv}`) |
| GET  | `/api/stream?id=<id>` | Flux **SSE** temps réel (snapshot + epochs + statut) |

## Tests

```bash
cd Projet_Panacee
pip install httpx           # requis par le TestClient
python -m pytest tests/test_webapp.py -v
```

Les tests couvrent : endpoints REST, découverte/résolution de runs (avec garde
anti-traversée), verdict clinique, évaluation (mock), et le flux SSE de bout en bout.
