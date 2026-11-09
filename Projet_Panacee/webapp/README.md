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

### Exécution entièrement sur ce PC

Depuis `Projet_Panacee/` :

```bash
pip install -r requirements.txt
pip install -r webapp/requirements.txt
python -m webapp.run --host 127.0.0.1 --port 8000
```

Ouvrir `http://127.0.0.1:8000`. Le serveur refuse une liaison réseau externe
par défaut. `--root D:\chemin\vers\checkpoints` permet de sélectionner un
dossier de poids local hors du dépôt. Pour A'', ce dossier doit contenir
`model_A_prime_augmented/manifest.json`, les quatre `rf_*.joblib` et le
checkpoint GNN référencé par le manifeste (ou son équivalent sous `phase3/`).
Sans ces fichiers, RDKit reste utilisable ; les modes GNN sont indiqués absents.

### Actualisation des données d'entraînement

Le tableau de bord lit `checkpoints/` sur ce PC, et non les fichiers restés sur
un serveur distant. Pour synchroniser un journal téléchargé ou un fichier sur
un volume réseau monté, définir `PANACEE_SYNC_SOURCE` (chemin du
`live_metrics.jsonl`) et `PANACEE_SYNC_RUN_ID` dans `.env`, puis redémarrer.
L'interface interroge `/api/runs` toutes les 15 secondes : un journal plus
avancé du même run est importé automatiquement. Un journal plus ancien ou
incompatible n'écrase jamais les données locales. La date et l'epoch des
données réellement affichées sont visibles au-dessus du tableau de bord.
Cette synchronisation ne se connecte pas directement à un serveur SSH non
configuré ; le fichier source doit être accessible localement.

Pour le serveur JupyterHub déjà enregistré dans VS Code, le rapatriement est
préparé sans recopier de jeton dans le code :

```bash
python -m webapp.jupyter_sync             # état du serveur personnel
python -m webapp.jupyter_sync --sync      # Phase 1-3 : journaux ; Phase 4 : résultats
python -m webapp.jupyter_sync --sync --weights  # + checkpoints, RF et manifestes
```

Le script lit la connexion VS Code ou `PANACEE_JUPYTER_URL` et
`PANACEE_JUPYTER_TOKEN`, puis utilise l'API JupyterHub et l'API Contents du
serveur personnel. Il ne démarre pas le serveur et n'importe rien s'il est
arrêté. Les journaux ne remplacent que des runs compatibles plus avancés ;
les poids déjà présents sont sauvegardés avant remplacement. Les fichiers
Phase 4 (`phase4/*.csv`, résumés JSON, interprétations) sont aussi rapatriés.
`GET /api/phases` distingue les quatre phases documentées des artefacts
réellement présents sur ce PC ; le laboratoire intègre les comptes et premiers
hits Phase 4 locaux dans chaque nouveau dossier.
La section « Modèle VIH livré depuis Jupyter » affiche le manifeste du dernier
ensemble **complet** disponible et sa date déclarée. Le sélecteur des runs et
ses courbes n'affichent par défaut que les entraînements postérieurs à ce
modèle ; les anciens runs restent stockés sans apparaître dans l'interface.
Le laboratoire ne propose par défaut que le dernier ensemble complet et RDKit
sans modèle. `PANACEE_SHOW_HISTORICAL_MODELS=1` restaure les anciens choix pour
une comparaison volontaire.
Avec `PANACEE_JUPYTER_AUTO_SYNC=1`, le webapp vérifie le serveur toutes les cinq
minutes en arrière-plan et importe automatiquement les nouveaux journaux et
résultats Phase 4 dès
qu'il est actif. L'état distant apparaît dans la barre de fraîcheur ; les poids
restent un transfert explicite avec `--weights`.

Le dépôt GitHub contient désormais les modèles finaux Jupyter via Git LFS,
leurs manifestes et les résultats sélectionnés des phases 1–4. Les checkpoints
intermédiaires de chaque époque ne sont pas publiés. Un clone exige Git LFS
pour récupérer les poids ; `git pull` seul ne prouve pas que le serveur Jupyter
n'a pas produit de nouvelles versions depuis le dernier commit.

`--offline` interdit PubMed, les API et les CLI cloud. Seul un modèle Ollama
installé localement, appelé par `codex exec --oss --local-provider ollama`,
peut alors produire une réponse IA. `--demo` désactive les commandes
d'administration et les appels IA.

Claude Code et Codex CLI peuvent être utilisés depuis ce PC avec leurs sessions
déjà connectées. Leur exécutable est local mais, en mode standard, **l'inférence
est effectuée par leurs services distants**. Codex/Ollama est l'option hors ligne.

### Laboratoire de recherche VIH

L'onglet **Laboratoire VIH · Phase 4** réunit une question, jusqu'à cinq molécules
ou une bibliothèque, un modèle Panacée choisi, des extraits sourcés du code,
des documents, rapports et commits Git locaux, des notices PubMed facultatives
et jusqu'à dix hypothèses. Les choix Panacée comprennent RDKit sans modèle,
Phase 2, Phase 3 et les ensembles A/A'' lorsque leurs artefacts existent.
L'ensemble produit un **rang relatif au lot**, jamais une probabilité d'activité.

Claude et OpenAI/Codex peuvent être sélectionnés séparément ou ensemble, via
API ou CLI locale. Les modèles CLI installés sont détectés ; des listes de
modèles autorisés se règlent via `PANACEE_CLAUDE_CLI_MODELS`,
`PANACEE_CODEX_CLI_MODELS`, `PANACEE_CLAUDE_MODELS` et
`PANACEE_OPENAI_MODELS`. La connexion d'abonnement ChatGPT locale reste
disponible dans le laboratoire pour les modèles autorisés par ce compte.
Les deux IA reçoivent le **même dossier de preuves** et produisent d'abord des
analyses indépendantes, puis chacune critique l'autre. Le rapport distingue
documents, prédictions, hypothèses et éléments manquants ; l'accord des IA ne
constitue pas une validation expérimentale.

Les dossiers sont stockés localement dans `data/panacee_lab.db` et exportables en
JSON. L'index du projet est stocké dans `data/panacee_lab_index.db` ; il ne lit
ni secrets, ni poids, ni lignes des datasets (colonnes seulement). Un dossier
antérieur peut être repris : les cinq derniers dossiers de la chaîne servent
de mémoire bornée. L'interface affiche la chronologie des questions et laisse
le champ libre pour la question suivante ; chaque nouveau dossier conserve
l'identifiant de son parent. API :
`GET /api/lab/models`, `GET /api/lab/index`, `GET /api/lab/reports`,
`GET /api/lab/reports/{id}`, `GET /api/lab/conversation?id={id}`,
`POST /api/lab/run`.

### Connexion ChatGPT par abonnement

Le laboratoire propose « Connecter l'abonnement ChatGPT local » pour un compte
éligible, distinct de la clé API OpenAI. L'application vérifie les modèles
autorisés par ce compte et n'utilise pas la clé API comme secours silencieux.
Les jetons sont chiffrés localement et ignorés par Git. Cette connexion exige
`http://127.0.0.1:8000` dans le navigateur et reste indépendante de Codex CLI.

Options : `--port 8080`, `--root <dossier_checkpoints>`, `--offline`, `--demo`.

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
os.environ["PANACEE_PUSH_TOKEN"] = "secret"                        # requis pour accès distant
# puis : python run_phase2.py --download --epochs 60
```

Côté dashboard, protège l'ingestion avec le même secret :
`PANACEE_INGEST_TOKEN=secret python -m webapp.run --host 0.0.0.0`. Chaque epoch
arrive via `POST /api/ingest` et apparaît dans l'onglet Évolution comme un run local.
Le jeton d'ingestion ne permet aucun accès aux autres API. Un accès API distant
distinct exige `PANACEE_API_TOKEN` dans l'en-tête `X-Panacee-Token` ; sans ce
jeton, ces routes restent fermées. Sans tunnel, l'alternative reste l'**import**
du `live_metrics.jsonl` téléchargé.

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
