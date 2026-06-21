# Contribuer à Panacée

Merci de contribuer ! Ce guide couvre l'installation, les tests, le style et les
conventions. Pour la vue d'ensemble technique, voir [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Installation

```bash
# 1) PyTorch (choisir GPU ou CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU CUDA 12.1

# 2) PyTorch Geometric
pip install torch-geometric

# 3) Le reste
pip install -r requirements.txt

# 4) Outils de dev (lint + tests)
pip install ruff pytest httpx  # cf. [project.optional-dependencies].dev de pyproject.toml
```

> Le projet s'exécute via un *bootstrap* `sys.path` (pas comme paquet installé) :
> lance les scripts depuis la racine `Projet_Panacee/`, sans `pip install -e .`.

Copier `.env.example` en `.env` au besoin (toutes les variables sont optionnelles).

## 2. Lancer le tableau de bord

```bash
python -m webapp.run                 # http://127.0.0.1:8000
python -m webapp.run --host 0.0.0.0 --port 8080
```

Ou via Docker : `docker build -t panacee-dashboard . && docker run -p 8000:8000 panacee-dashboard`.

## 3. Tests & lint (à faire AVANT chaque PR)

```bash
ruff check .          # lint (doit être clean)
ruff check . --fix    # corrige automatiquement le corrigeable
pytest                # suite de tests (config dans pyproject.toml)
```

La **CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) rejoue `ruff` + `pytest`
sur chaque push/PR vers `main`. Une PR doit passer la CI.

### Écrire des tests
- Les tests vivent dans `tests/` ; fixtures partagées dans `tests/conftest.py`
  (`runs_root` = racine de runs isolée, `client` = TestClient ASGI).
- Privilégier les **fonctions pures** de `webapp/service.py` (faciles à tester).
- Les dépendances lourdes se sautent proprement :
  `pytest.importorskip("torch")` / `"torch_geometric"` / `"rdkit"`.
  La CI installe torch CPU + rdkit ; les tests qui exigent `torch_geometric`
  s'exécutent en local.

## 4. Style & conventions

- **Python ≥ 3.10**, formaté/lint par **ruff** (config dans `pyproject.toml`,
  ligne ≤ 100). Les imports sont triés automatiquement (`ruff check . --fix`).
- **Commentaires et docstrings en français**, concis, expliquant le *pourquoi*.
- **Sécurité** : charger les checkpoints via `src/utils/safe_load.py`
  (`safe_load_checkpoint`), jamais `torch.load(..., weights_only=False)` direct sur
  des fichiers potentiellement importés.
- **Frontend** : pas de build, pas de framework. JS vanilla dans `webapp/static/app.js`.

## 5. Recettes courantes

### Ajouter une bibliothèque de molécules
Dans [webapp/catalog.py](webapp/catalog.py), ajouter une entrée à `LIBRARIES` :
```python
"ma_lib": {"label": "…", "note": "…",
           "molecules": [{"name": "…", "smiles": "…"}]},
```
Les SMILES sont validés par RDKit à l'usage (les invalides sont ignorés).
Vérifier : `pytest tests/test_catalog.py`.

### Ajouter un terme au lexique
Ajouter `{"term", "def", "ex"}` au bon groupe de `GLOSSARY` dans `catalog.py`.

### Ajouter un endpoint API
1. Écrire un handler `async def api_xxx(request)` dans `webapp/server.py`.
2. L'enregistrer dans la liste `routes`.
3. Mettre la logique pure dans `service.py`/`cheminfo.py` et la **tester**.

## 6. Branches, commits, PR

- Travailler sur une branche dédiée (jamais directement sur `main`).
- Messages de commit clairs et impératifs (« Ajoute… », « Corrige… »).
- Une PR = un sujet cohérent, CI verte, tests pour tout nouveau comportement.

## 7. Sécurité & données

- Ne **jamais** committer de secrets (`.env` est ignoré ; utiliser `.env.example`).
- Les données/poids lourds ne vont pas dans git (voir `.gitignore` / `.dockerignore`).
- Signaler toute faille de sécurité en privé plutôt que dans une issue publique.
