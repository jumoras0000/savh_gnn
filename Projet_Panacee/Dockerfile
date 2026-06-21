# ════════════════════════════════════════════════════════════════════
#  Panacée — image du TABLEAU DE BORD (serveur web + chimie + inférence CPU)
#
#  L'ENTRAÎNEMENT lourd (GPU) se fait ailleurs (Kaggle) ; cette image sert le
#  dashboard, la cheminformatique (RDKit) et l'évaluation/inférence sur CPU.
#
#  Build :  docker build -t panacee-dashboard .
#  Run   :  docker run -p 8000:8000 \
#             -v $(pwd)/checkpoints:/app/checkpoints \
#             --env-file .env panacee-dashboard
# ════════════════════════════════════════════════════════════════════
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PANACEE_CKPT_ROOT=/app/checkpoints

WORKDIR /app

# PyTorch CPU (index dédié) — suffit pour servir / évaluer (pas d'entraînement GPU)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Pile d'inférence + serveur. torch-geometric pour le modèle GNN, rdkit pour la
# chimie. (deepchem n'est requis que pour télécharger les données d'entraînement.)
RUN pip install \
      torch-geometric>=2.3.0 \
      rdkit>=2023.3.1 \
      numpy scipy scikit-learn pandas tqdm \
      starlette>=0.40 "uvicorn[standard]>=0.30"

# Code de l'application
COPY . /app

EXPOSE 8000

# Démarre le serveur ASGI (lié à 0.0.0.0 pour être joignable hors conteneur)
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000"]
