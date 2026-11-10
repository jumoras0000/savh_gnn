# Reprise du projet Panacée

Lire d'abord [`docs/CLAUDE_HANDOFF_2026-11-09.md`](docs/CLAUDE_HANDOFF_2026-11-09.md),
puis [`docs/TRANSLATIONAL_ROADMAP.md`](docs/TRANSLATIONAL_ROADMAP.md) et
[`docs/STUDIO_V2_DECISIONS.md`](docs/STUDIO_V2_DECISIONS.md).
Le protocole de données et de sélection du prochain modèle est
[`docs/VNEXT_MODEL_PROTOCOL.md`](docs/VNEXT_MODEL_PROTOCOL.md).
Le guide utilisateur accessible sur `/guide` explique la finalité, les quatre
phases Jupyter, chaque partie du Studio et le sens des indicateurs. Sa source
est `webapp/static/guide.html` ; garder les explications synchronisées avec
les métriques reproduites et les états réels des API.
L'ancien onglet Assistant et l'écran d'évaluation toxicologique par endpoint
ont été retirés de `/legacy` : ils étaient redondants ou présélectionnaient
des données non adaptées à A''. Le Laboratoire conserve la recherche croisée
Panacée–Claude–ChatGPT ; les adaptateurs Claude restent au backend pour ce
Laboratoire et la phase 4. Ne pas réintroduire un sélecteur tox21/ChEMBL
positive-only comme validation du modèle VIH A''.
Le Laboratoire de recherche a été vérifié de bout en bout avec A'', Claude
Code et ChatGPT connecté : deux analyses puis deux critiques croisées sur un
dossier commun. Son score A'' passe par `studio.screen`. L'environnement
d'inférence des forêts RF doit utiliser `scikit-learn==1.9.1` (voir
`requirements-a-prime-runtime.txt`).
La revue scientifique des quatre phases est
[`reports/revue_scientifique_panacee_2026-11.pdf`](reports/revue_scientifique_panacee_2026-11.pdf) ;
sa construction traçable se trouve dans `scripts/build_scientific_review.py`.
L'audit ChEMBL du 10 novembre est dans
`reports/chembl_source_audit_2026-11.json` ; il établit pourquoi les données
actuelles ne suffisent pas pour un modèle d'EC50/CC50 cellulaire.

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
