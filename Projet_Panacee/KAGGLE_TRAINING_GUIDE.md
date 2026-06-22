# Guide : Entraînement Panacée sur Kaggle (A → Z)

Durée effective : **5 minutes de setup + durée d'entraînement** (quelques heures sur GPU Kaggle).

---

## 1. Démarrer le dashboard local

```powershell
# Copier .env.example → .env et choisir un token
copy .env.example .env
# Éditer .env → PANACEE_INGEST_TOKEN=mon_token_secret_42

# Démarrer le dashboard (charge .env automatiquement)
python -m webapp.run
```

La bannière affiche le statut du token :
```
  Réception Kaggle : /api/ingest  [token=****_42]
```

---

## 2. Exposer le dashboard publiquement (ngrok)

Kaggle a besoin d'une URL HTTPS publique pour envoyer les métriques.

```powershell
# Télécharger ngrok : https://ngrok.com/download
# Puis dans un 2ème terminal :
ngrok http 8000
```

Sortie attendue :
```
Forwarding  https://abc123def.ngrok-free.app -> http://127.0.0.1:8000
```

**Copier cette URL** → vous en aurez besoin dans le notebook Kaggle.

> **Alternatives à ngrok** :
> - `npx localtunnel --port 8000`
> - `cloudflared tunnel --url http://localhost:8000`
> - Déployer le dashboard sur Render/Railway (URL fixe)

---

## 3. Créer un notebook Kaggle

### Option A (recommandée) : uploader le notebook prêt

1. Aller sur [kaggle.com/notebooks](https://www.kaggle.com/code)
2. Cliquer **+ New Notebook**
3. **File → Import Notebook** → uploader `notebooks/kaggle_phase2.ipynb`

> Ce fichier est dans le repo à `Projet_Panacee/notebooks/kaggle_phase2.ipynb`.

### Option B : copier-coller

Créer un nouveau notebook et coller ce code dans des cellules séparées :

**Cellule 1 — Configuration**
```python
# ╔══════════════════════════════════════════════╗
# ║  CONFIGURATION — modifier ces 4 lignes ONLY  ║
# ╚══════════════════════════════════════════════╝

NGROK_URL     = "https://abc123def.ngrok-free.app"  # ← votre URL ngrok
TOKEN         = "mon_token_secret_42"               # ← identique à PANACEE_INGEST_TOKEN
RUN_NAME      = "kaggle_phase2_run01"               # ← nom affiché dans le dashboard
N_EPOCHS      = 50                                  # ← 50-100 sur GPU, 10-20 sur CPU
MAX_MOLECULES = None                                # ← None = tout Tox21, ou ex: 2000
```

**Cellule 2 — Vérification connexion**
```python
import urllib.request, json

try:
    r = urllib.request.urlopen(NGROK_URL.rstrip('/') + '/api/health', timeout=8)
    print(f"✅ Dashboard joignable — {json.loads(r.read())}")
except Exception as e:
    print(f"❌ NON joignable : {e}")
    raise SystemExit("Corrigez NGROK_URL avant de continuer.")
```

**Cellule 3 — Env vars + dépendances**
```python
import os, subprocess, sys
from pathlib import Path

os.environ["PANACEE_PUSH_URL"]   = NGROK_URL
os.environ["PANACEE_PUSH_TOKEN"] = TOKEN
os.environ["PANACEE_PUSH_RUN"]   = RUN_NAME
os.environ["PYTHONIOENCODING"]   = "utf-8"

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("torch-geometric", "rdkit", "deepchem")
print("OK ✓")
```

**Cellule 4 — Clone du repo**
```python
CLONE_DIR = Path("/kaggle/working/panacee")
REPO_URL  = "https://github.com/jumoras0000/savh_gnn.git"

if not CLONE_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(CLONE_DIR), "pull", "--ff-only"], check=False)

PROJET = CLONE_DIR / "Projet_Panacee"
if str(PROJET) not in sys.path:
    sys.path.insert(0, str(PROJET))

SAVE_DIR = f"/kaggle/working/checkpoints/{RUN_NAME}"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
print(f"Prêt : {PROJET}")
```

**Cellule 5 — Lancer l'entraînement**
```python
import os as _os
_os.chdir(str(PROJET))

sys.argv = [
    "run_phase2.py",
    "--download",
    "--save_dir",  SAVE_DIR,
    "--epochs",    str(N_EPOCHS),
]
if MAX_MOLECULES:
    sys.argv += ["--max_molecules", str(MAX_MOLECULES)]

from run_phase2 import main
main()
```

---

## 4. Activer le GPU Kaggle (optionnel mais recommandé)

1. Panneau de droite → **Session options** → Accelerator : **GPU T4 x2**
2. L'entraînement détecte CUDA automatiquement et s'accélère ~10×

---

## 5. Surveiller en temps réel

Dès que l'entraînement démarre, ouvrir [http://127.0.0.1:8000](http://127.0.0.1:8000) :

| Onglet | Ce qu'on voit |
|--------|---------------|
| **Runs** | Le run `kaggle_phase2_run01` apparaît automatiquement |
| **Entraînement** | Courbes AUC, loss, LR en direct (SSE) |
| **Évaluation** | Verdict clinique mis à jour chaque epoch |
| **Sécurité** | Alertes FNR / dangers |
| **Comparaison** | Comparaison avec d'autres runs |

---

## 6. Récupérer le checkpoint

À la fin de l'entraînement (ou à n'importe quel moment) :

**Depuis Kaggle :**
1. Panneau **Output** (à droite) → naviguer dans `checkpoints/<RUN_NAME>/`
2. Télécharger `best_toxicity_model.pth`

**Pour exporter vers un Dataset Kaggle (Phase 3 → Phase 3) :**
```python
# Ajouter une cellule à la fin du notebook Phase 2
import shutil, os
os.makedirs("/kaggle/output/checkpoints", exist_ok=True)
shutil.copytree(SAVE_DIR, f"/kaggle/output/checkpoints/{RUN_NAME}")
# Ensuite : Save Version → Output → Share as Dataset
```

**Placer le checkpoint localement (pour cribler des molécules) :**
```powershell
# Copier dans le dossier checkpoints local
copy Downloads\best_toxicity_model.pth checkpoints\kaggle_phase2_run01\
```
Puis dans le dashboard → onglet Recherche → sélectionner ce checkpoint.

---

## 7. Phase 3 (après Phase 2)

Uploader `notebooks/kaggle_phase3.ipynb` sur Kaggle, et en **Input** ajouter :
- Le dataset contenant `best_toxicity_model.pth` (créé à l'étape précédente)

Configurer `PHASE2_CKPT` avec le chemin dans `/kaggle/input/...`.

---

## Dépannage

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `❌ NON joignable` (cellule 2) | ngrok pas lancé / URL copiée incorrectement | Vérifier terminal ngrok, recopy l'URL exacte |
| Run n'apparaît PAS dans le dashboard | Token incorrect (`403 Forbidden` côté Kaggle) | Vérifier que `TOKEN` == `PANACEE_INGEST_TOKEN` dans `.env` |
| `403 Forbidden` dans les logs Kaggle | Token manquant ou mauvais | Redémarrer le dashboard après avoir mis `.env` |
| Pas de métriques temps réel | ngrok expiré (session > 2h sur compte gratuit) | Relancer ngrok, mettre à jour `NGROK_URL` |
| `CUDA out of memory` | Batch trop grand | Ajouter `--batch_size 16` dans `sys.argv` |
| `deepchem install failed` | Conflict de dépendances | `pip("deepchem", "--no-deps")` puis `pip("scikit-learn")` |

---

## Fichiers du repo

| Fichier | Rôle |
|---------|------|
| `notebooks/kaggle_phase2.ipynb` | Notebook Phase 2 prêt à uploader sur Kaggle |
| `notebooks/kaggle_phase3.ipynb` | Notebook Phase 3 prêt à uploader sur Kaggle |
| `.env.example` | Template des variables d'environnement locales |
| `src/utils/live_logger.py` | Logger qui pousse les métriques vers le dashboard |
| `webapp/server.py` → `/api/ingest` | Endpoint qui reçoit les métriques depuis Kaggle |
| `webapp/run.py` | Lanceur du dashboard (charge `.env` automatiquement) |
