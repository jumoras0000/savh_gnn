# Reprise du projet Panacée

Lire d'abord [`docs/CLAUDE_HANDOFF_2026-11-09.md`](docs/CLAUDE_HANDOFF_2026-11-09.md),
puis [`docs/TRANSLATIONAL_ROADMAP.md`](docs/TRANSLATIONAL_ROADMAP.md) et
[`docs/STUDIO_V2_DECISIONS.md`](docs/STUDIO_V2_DECISIONS.md).
La revue scientifique des quatre phases est
[`reports/revue_scientifique_panacee_2026-11.pdf`](reports/revue_scientifique_panacee_2026-11.pdf) ;
sa construction traçable se trouve dans `scripts/build_scientific_review.py`.

L'objectif est de transformer le classement VIH A'' en **candidats testables**,
sans présenter une prédiction informatique comme une efficacité biologique.
Tout fonctionne localement. Les poids A'' du 6 novembre 2026 sont la référence
actuelle ; ne pas substituer les anciens modèles Kaggle ou les checkpoints
intermédiaires. Le paquet aveugle et sa clé sont locaux et exclus de Git.

Avant toute modification : vérifier `git status`, `git fetch origin main`, les
empreintes avec `webapp.studio_support.verify_registry(..., datasets=True)` et
l'état de `http://127.0.0.1:8000/api/studio/development`. Ne pas écraser le
paquet gelé après réception de résultats. Ne pas committer `note`,
`.playwright-mcp/`, les données expérimentales ou des identifiants Jupyter.
