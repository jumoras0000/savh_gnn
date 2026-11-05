# Rapport scientifique — Panacée GNN anti-VIH, Phase 3

**Objet :** criblage virtuel d'inhibiteurs anti-VIH par réseau de neurones sur graphes (GNN), optimisation du pouvoir de classement (ranking) pour la découverte de médicaments.
**Dernière mise à jour :** 2026-11-05
**Statut :** Model A (ensemble) figé et validé ; augmentation de données (B) en cours.

> Tous les chiffres de ce rapport proviennent des rapports JSON / logs bruts produits par les scripts d'évaluation, pas d'estimations. Les affirmations causales sont marquées *mesuré* ou *hypothèse*.

---

## 1. Données

Split scaffold global (anti-fuite de scaffold vérifiée), cible `hiv_HIV_active` (MoleculeNet HIV).

| Split | Molécules | Actifs | Taux actifs |
|-------|----------:|-------:|------------:|
| Train | 33 570 | 1 249 | 3.72 % |
| Val | 3 772 | — | — |
| Test | 3 778 | 84 | 2.22 % |

**Intégrité du gate DrugBank (vérifiée, `check_arv_leakage.py`) :** sur les 34 ARV approuvés servant de vérité terrain, **2 seulement** sont présents dans le train (Zidovudine, Stavudine, labellisés actifs) ; **32/34 absents** de tous les splits. Le gate teste donc majoritairement de la généralisation (généralisation honnête = 22 trouvés / 32 propres).

---

## 2. Métriques

Criblage virtuel standard (Truchon-Bayly 2007 ; Jiménez-Luna 2020) :
- **ROC-AUC** : séparation globale.
- **EF@X%** (enrichment factor) : enrichissement en actifs dans le top X%.
- **BEDROC α=20** : reconnaissance précoce (pondère le haut du classement).
- **hit rate top-K**, **recall@top-X%**, **PRC-AUC**.
- **Gate DrugBank** : recall des 34 ARV approuvés dans le top-10 % (pool ARV + test). Seuil de réussite : ≥ 25/34.

---

## 3. Modèles évalués (test held-out, chiffres vérifiés)

| Modèle | AUC | EF@1% | EF@5% | BEDROC | hit@100 | DrugBank |
|--------|----:|------:|------:|-------:|--------:|---------:|
| GNN seul (GATv2) | **0.7194** | 15.39 | 5.00 | 0.2845 | 0.160 | 17/34 |
| RF + ECFP4 | 0.6524 | 17.75 | 6.19 | 0.3282 | 0.230 | 20/34 |
| LR + ECFP4 | 0.6206 | 8.29 | 3.81 | 0.2072 | 0.130 | 15/34 |
| RF + ECFP6 | 0.6030 | 16.57 | — | 0.3273 | 0.230 | 19/34 |
| RF + MACCS | 0.6779 | 16.57 | — | 0.3109 | 0.200 | 16/34 |
| RF + RDKit-FP | 0.6811 | 11.84 | — | 0.3274 | 0.200 | 17/34 |
| Hybride GNN+Morgan (fusion interne) | 0.6749 | 11.84 | 5.24 | 0.2809 | 0.190 | 20/34 |
| Ensemble rank_avg 2 (GNN+ECFP4) | 0.700 | 17.75 | 6.43 | 0.3345 | 0.220 | 22/34 |
| **Ensemble rank_avg 5 — Model A** | 0.6801 | 17.75 | 6.19 | **0.3477** | **0.240** | **24/34** |

---

## 4. Résultats clés et conclusions

### 4.1 Le GNN gagne l'AUC, les fingerprints gagnent le ranking pharma *(mesuré)*
Le GNN domine l'AUC globale (0.719 vs 0.652 pour RF) mais **perd sur toutes les métriques de reconnaissance précoce** (EF@1%, BEDROC, hit rate, DrugBank). Interprétation : les fingerprints circulaires capturent les sous-structures pharmacophores pertinentes pour le haut du classement, là où les features apprises du GNN optimisent la séparation moyenne.

### 4.2 La fusion interne (hybride concat) échoue *(mesuré)*
Concaténer le Morgan FP (2048-d) à l'embedding GNN (256-d) avant la couche partagée donne un modèle **dominé partout** : pire AUC que le GNN, pire ranking que RF. Gap val→test 0.80→0.67 = surapprentissage. *Hypothèse* : déséquilibre dimensionnel + redondance (une seule couche linéaire est moins expressive qu'un RF à 500 arbres sur les mêmes features). La fusion externe (stacking) est retenue à la place.

### 4.3 L'ensemble par rank-average domine ses composants *(mesuré + bootstrap IC95 %, B=2000)*
Le rank-average (non-paramétrique, invariant d'échelle) **bat le méta-apprenant LR** : le LR optimise la vraisemblance (≈AUC) et non le ranking.

Significativité (IC95 % de la différence paired exclut 0) :
- **rank_avg_2 vs RF** : ΔAUC +0.048 ✓, ΔDrugBank +2 ✓ ; EF/BEDROC = égalité (bruit).
- **rank_avg_2 vs GNN** : ΔBEDROC +0.050 ✓, Δhit@100 +0.06 ✓, ΔPRC +0.048 ✓, ΔDrugBank +5 ✓ ; ΔAUC −0.019 = non significatif (pas de perte réelle).

→ **L'ensemble n'est significativement pire que personne sur aucune métrique**, et significativement meilleur sur plusieurs. Résultat conforme à la littérature (un ensemble bat ses composants).

### 4.4 Model A (rank_avg 5) : meilleur modèle honnête, 24/34
L'ajout de représentations diverses (ECFP6, MACCS, RDKit-FP) porte le DrugBank à **24/34** (meilleur score du projet). Mais vs rank_avg_2, seul ΔDrugBank +2 est significatif ; le reste est du bruit. **Gate non franchi** (24 < 25).

Model A est **figé et re-vérifié** (`save_model_a.py` : artefact rechargé depuis disque reproduit exactement 24/34). Conservé comme repli.

### 4.5 Le plafond de 24/34 est un problème de DONNÉES, pas de modèle *(mesuré)*
Diagnostic des 10 ARV manqués (`diagnose_arv.py`, Tanimoto max vs actifs du train) :
- **9/10 franchement hors-domaine** (sim 0.19–0.25) : inhibiteurs d'intégrase (Raltegravir, Dolutegravir, Bictegravir, Cabotegravir), NNRTI récents (Efavirenz, Rilpivirine, Doravirine), entrée (Maraviroc CCR5, Fostemsavir gp120). Ces classes sont **absentes de l'espace chimique du training** → aucun fingerprint ne les rattrape.
- **1/10 limite in-domain** (Tenofovir, sim 0.467, rang 58.7 %) : échec de classement isolé, trop profond pour un gain réaliste.

→ Le fix légitime pour dépasser 24/34 n'est pas « plus de modèles » (ce serait sur-ajuster le gate), mais **enrichir le training en actifs des classes manquantes**.

---

## 5. Augmentation de données (B) — en cours

**Source :** actifs ChEMBL confirmés (pChEMBL ≥ 5) des 5 cibles anti-VIH, déjà téléchargés et vérifiés :

| Cible ChEMBL | Rôle | Actifs |
|--------------|------|-------:|
| CHEMBL247 | Transcriptase inverse | 4 908 |
| CHEMBL243 | Protéase | 4 471 |
| CHEMBL3471 | **Intégrase** | 2 496 |
| CHEMBL274 | **CCR5** | 2 173 |
| CHEMBL5658 | **gp120** | 1 794 |

Couverture directe des classes OOD manquées.

**Protocole anti-fuite des actifs ChEMBL ajoutés au train :**
1. **Gate ARV** : exclusion des 34 ARV par SMILES canonique exact (les médicaments eux-mêmes). Apprendre d'autres composés actifs sur la même cible = transfert légitime de pharmacophore. Le nombre d'ajouts partageant un scaffold Bemis-Murcko avec un ARV est **compté et rapporté** (transparence sur l'aide au gate).
2. **Éval** : exclusion par **scaffold de Bemis-Murcko** de tout ChEMBL partageant un squelette avec val/test (pas seulement canonique exact) → ferme la fuite d'**analogues**, cohérent avec le scaffold split du projet.
3. Déduplication canonique vs train existant.

Logique d'anti-fuite et de bootstrap testée par exécution (jeux jouets, réponses connues).

**Plan :** ré-entraîner les RF sur le train augmenté (levier le moins coûteux ; les RF portent le DrugBank) → Model A'. Évaluation sur le **même** test held-out + **même** gate. Bootstrap paired A' vs A. GNN inchangé dans un premier temps ; fine-tune GNN (`finetune_hiv_chembl.py`, anti-fuite intégré) si insuffisant.

**Model A conservé intact** comme repli si A' ne franchit pas les benchmarks.

---

## 6. Limites assumées

- Métriques = estimations ponctuelles confirmées par bootstrap IC95 % (percentile, approximation, pas BCa).
- EF@1 % est discret (top 1 % ≈ 38 molécules) → faibles différences non détectables.
- 2/34 ARV mémorisés (dans le train) gonflent le gate de 2.
- Mélange d'actifs phénotypiques (MoleculeNet) et de liaison cible (ChEMBL) en B : sémantique de label légèrement hétérogène, trade-off AUC MoleculeNet / recall DrugBank à mesurer.

---

## 7. Reproductibilité

Scripts (seed=42 partout) : `benchmark_baselines.py`, `train_phase3_hybrid.py`, `stack_gnn_rf.py`, `bootstrap_ci.py`, `push_ensemble.py`, `diagnose_arv.py`, `check_arv_leakage.py`, `save_model_a.py`, `model_a.py`, `train_augmented_rf.py`. Rapports JSON dans `checkpoints/phase3_stacking/`. Artefact Model A : `checkpoints/model_A_rank_avg_5/`.
