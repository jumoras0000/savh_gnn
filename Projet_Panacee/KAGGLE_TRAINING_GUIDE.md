# Guide : Entraînement Panacée sur Kaggle (A → Z)

Lancez un entraînement (Phase 1/2/3) sur Kaggle avec supervision temps réel depuis
votre dashboard local. Les métriques arrivent en continu via SSE sans interrompre l'entraînement.

> **Durée totale** : ~5 min de setup + entraînement (quelques heures). Vous pouvez fermer
> le notebook, les données sont sauvegardées — le dashboard les récupère au prochain refresh.

---

## Étape 1 : Préparer l'accès HTTP (exposer votre dashboard)

Votre dashboard tourne localement sur `http://127.0.0.1:8000`. Kaggle a besoin d'un
**URL public** pour envoyer les métriques. Utilisez un tunnel (ngrok = le plus simple) :

```bash
# 1) Installer ngrok (https://ngrok.com/download ou pip install pyngrok)
# Windows : télécharger l'exe ou via chocolatey/scoop
# Mac/Linux : brew install ngrok

# 2) Lancer ngrok (depuis n'importe quel terminal)
ngrok http 8000
# Output :
# Forwarding    http://abc123def456.ngrok.io -> http://127.0.0.1:8000
# Copier l'URL publique (abc123def456.ngrok.io) → vous en aurez besoin ci-dessous
```

> Si ngrok n'est pas dispo, alternatives : **localtunnel** (`npx lt --port 8000`),
> **Cloudflare Tunnel** (`cloudflared tunnel run`), ou déployer le dashboard sur un serveur
> (Heroku, AWS, Render) avec une URL fixe.

---

## Étape 2 : Configurer l'authentification Kaggle

Générez un token d'authentification :
- Aller à https://www.kaggle.com/account → **API** → **Create new API Token**
- Cela télécharge `kaggle.json` → le placer dans `~/.kaggle/kaggle.json`
  - Windows : `C:\Users\<votreuser>\.kaggle\kaggle.json`
  - Mac/Linux : `~/.kaggle/kaggle.json`
- Chmod 600 sur Unix : `chmod 600 ~/.kaggle/kaggle.json`

```bash
# Vérifier que c'est bon
kaggle datasets list | head -3
# Si ça marche, l'API est prête
```

---

## Étape 3 : Créer un notebook Kaggle

1. **Kaggle** → **Notebooks** → **+ New Notebook**
2. Choisir **Python** (GPU ou TPU pour la Phase 1, CPU pour Phase 2/3 légers)
3. **Copier/coller** le code ci-dessous dans la **première cellule** :

### Phase 2 (finetune toxicité) — *Recommandé pour débuter*

```python
# === Panacée Phase 2 sur Kaggle ===
import os, sys, json
from pathlib import Path

# CONFIGURATION
PUSH_URL = "http://abc123def456.ngrok.io/api/ingest"  # URL ngrok de votre dashboard
PUSH_TOKEN = "secret_token_42"  # Peut être n'importe quoi (validation basic)
RUN_NAME = "phase2_kaggle_01"  # Nom du run (affiché dans le dashboard)
PHASE = 2  # 1 (préentraînement) ou 2 (toxicité) ou 3 (multi-prop)

# Variables d'environnement pour la transmission
os.environ["PANACEE_PUSH_URL"] = PUSH_URL
os.environ["PANACEE_PUSH_TOKEN"] = PUSH_TOKEN
os.environ["PANACEE_INGEST_TOKEN"] = PUSH_TOKEN
os.environ["PYTHONIOENCODING"] = "utf-8"

# Cloner ou monter le repo
!git clone https://github.com/jumoras0000/savh_gnn.git
os.chdir("savh_gnn/Projet_Panacee")
sys.path.insert(0, str(Path.cwd()))

# Installer les dépendances
!pip install -q torch torch-geometric rdkit starlette httpx -q

# Importer et lancer
if PHASE == 2:
    from run_phase2 import main
    main(run_name=RUN_NAME, push_url=PUSH_URL, push_token=PUSH_TOKEN)
```

### Phase 3 (multi-propriétés) — *Après Phase 2*

```python
import os, sys, json
from pathlib import Path

PUSH_URL = "http://abc123def456.ngrok.io/api/ingest"
PUSH_TOKEN = "secret_token_42"
RUN_NAME = "phase3_kaggle_01"
PHASE = 3

os.environ["PANACEE_PUSH_URL"] = PUSH_URL
os.environ["PANACEE_PUSH_TOKEN"] = PUSH_TOKEN
os.environ["PYTHONIOENCODING"] = "utf-8"

!git clone https://github.com/jumoras0000/savh_gnn.git
os.chdir("savh_gnn/Projet_Panacee")
sys.path.insert(0, str(Path.cwd()))

!pip install -q torch torch-geometric rdkit starlette httpx

from run_phase3 import main
main(run_name=RUN_NAME, push_url=PUSH_URL, push_token=PUSH_TOKEN)
```

---

## Étape 4 : Configurer `.env.example` localement

Avant de lancer Kaggle, préparez votre dashboard local :

```bash
cd d:/02_PROJETS/AI_GNN_Panacee/Projet_Panacee

# Copier .env.example en .env
cp .env.example .env

# Éditer .env et remplir (optionnel, sinon les défauts suffisent) :
PANACEE_INGEST_TOKEN=secret_token_42  # Doit correspondre au PUSH_TOKEN Kaggle
PANACEE_CKPT_ROOT=./checkpoints
```

---

## Étape 5 : Lancer le dashboard local

```bash
# Terminal 1 : le dashboard
cd d:/02_PROJETS/AI_GNN_Panacee/Projet_Panacee
python -m webapp.run
# → http://127.0.0.1:8000

# Terminal 2 : ngrok (le tunnel)
ngrok http 8000
# → http://abc123def456.ngrok.io

# Garder les deux tournants pendant que Kaggle s'exécute
```

---

## Étape 6 : Lancer le notebook Kaggle

1. Remplacer `PUSH_URL` par votre URL ngrok
2. **Run** (les deux triangles ▶▶ en haut)
3. Kaggle commencera le téléchargement du repo + installation des libs
4. L'entraînement démarre → **les logs s'affichent dans le notebook**

---

## Étape 7 : Surveiller en temps réel sur le dashboard

Ouvrez http://127.0.0.1:8000 dans votre navigateur :

- **Onglet Runs** : le nouveau run (ex. `phase2_kaggle_01`) apparaît
- **Graphique live** : les métriques (AUC, sensibilité, etc.) arrivent en continu
- **Onglet Évaluation** : verdict clinique (OK/WARN/DANGER) mis à jour chaque epoch
- **Sélectionner l'epoch** : le best epoch selon la supervision clinique est mis en avant

Les données arrivent via **SSE** (Server-Sent Events) — pas de polling, juste un flux
continu.

---

## Étape 8 : Après l'entraînement

1. **Kaggle s'arrête** → le notebook affiche un résumé (AUC final, best epoch, etc.)
2. **Le checkpoint est sauvegardé** dans `/kaggle/working/checkpoints/phase2_kaggle_01/best.pth`
   - Il n'est pas poussé vers votre dashboard (Kaggle ferme après), mais
   - Vous pouvez **télécharger le notebook** ou **exporter les fichiers**
3. **Monter le checkpoint localement** pour continuer la Phase 3 ou faire de la recherche

### Récupérer le checkpoint depuis Kaggle

```bash
# Option A : via l'UI Kaggle
# → Notebook → Sessions → ⬇ Download (télécharge tout)

# Option B : directement depuis le notebook (avant la fin)
import shutil
shutil.copy("checkpoints/phase2_kaggle_01/best.pth", "/kaggle/output/best.pth")
# Ensuite : Notebook → Output → Fichiers téléchargeables
```

Ensuite, placer le fichier localement :
```bash
cp ~/Downloads/best.pth d:/02_PROJETS/AI_GNN_Panacee/Projet_Panacee/checkpoints/phase2_kaggle_01/
```

Puis sur le dashboard, **Recherche** → charger ce checkpoint et cribler des molécules.

---

## Dépannage

| Problème | Cause | Solution |
|----------|-------|----------|
| `Connection refused` (notebook) | ngrok pas lancé / URL incorrecte | Vérifier `PUSH_URL` = votre URL ngrok |
| Pas de données dans le dashboard | Token invalide | Vérifier `PANACEE_INGEST_TOKEN` = `PUSH_TOKEN` |
| Run n'apparaît pas dans le dashboard | Mauvaise URL ngrok ou expired | Relancer ngrok, obtenir une nouvelle URL |
| `403 Forbidden` sur `/api/ingest` | Token != PANACEE_INGEST_TOKEN | Ajuster le token dans `.env` |
| GPU out of memory | Batch size trop grand | Réduire `batch_size` dans `run_phase2.py` |

---

## Cas d'usage courants

### Lancer 3 runs en parallèle (Phase 2 + 2 Phase 3 différentes)

1. **Notebook 1** : Phase 2 avec `RUN_NAME = "p2_lr0.001"`
2. **Notebook 2** : Phase 3 warm-start avec `RUN_NAME = "p3_v1"` (charge `p2_lr0.001`)
3. **Notebook 3** : Phase 3 autre hyperparamètre `RUN_NAME = "p3_v2"`

Tous les 3 envoient des métriques au **même dashboard** en parallèle → comparaison en temps réel
via l'onglet **Comparaison**.

### Traiter les checkpoints comme des artefacts Kaggle

Si vous avez un compte **Kaggle Datasets**, créer un dataset d'output :
```python
# À la fin du notebook
!mkdir -p /kaggle/output/checkpoints
!cp -r checkpoints/* /kaggle/output/checkpoints/
# Ensuite : Share as Dataset, future notebooks peuvent `!kaggle datasets download -d username/panacee-checkpoints`
```

---

## Env. variables complets (dans `.env` local)

```bash
# Outils de développement
PYTHONIOENCODING=utf-8

# Tableau de bord
PANACEE_CKPT_ROOT=./checkpoints         # Racine des runs locaux
PANACEE_DB=panacee.db                   # SQLite pour les chats

# Réception Kaggle (pour /api/ingest)
PANACEE_INGEST_TOKEN=secret_token_42    # Authentifier les pushes distants

# Optionnel : Claude API (pour l'assistant)
ANTHROPIC_API_KEY=sk-ant-...
```

Et dans le notebook Kaggle :
```python
os.environ["PANACEE_PUSH_URL"] = "..."       # URL ngrok public
os.environ["PANACEE_PUSH_TOKEN"] = "..."     # Même token que INGEST_TOKEN
```

---

## Ressources

- **ARCHITECTURE.md** — Schéma complet, flux temps réel, modules
- **CONTRIBUTING.md** — Installation locale, tests, conventions
- **.env.example** — Toutes les variables documentées
- **src/utils/live_logger.py** — Implémentation du push HTTP
- **webapp/server.py** — Handler `/api/ingest`

Bon entraînement ! 🚀
