# Rapport évaluation Phase 3 — Panacée GNN anti-VIH
**Checkpoint** : `/home/jovyan/panacee/Projet_Panacee/checkpoints/phase3/panacee_phase3_complete.pth`
**Device** : cuda

## 1. Métriques pharma virtual screening (HIV test)
| Métrique | Valeur |
|---|---|
| n_total | 3778 |
| n_actives | 84 |
| actives_fraction | 0.0222 |
| roc_auc | 0.6582 |
| prc_auc | 0.1661 |
| ef_1pct | 16.5702 |
| ef_5pct | 5.2353 |
| ef_10pct | 3.0936 |
| bedroc_alpha20 | 1.0000 |
| bedroc_alpha80 | 1.0000 |
| hit_rate_top_100 | 0.2000 |
| hit_rate_top_500 | 0.0600 |
| hit_rate_top_1000 | 0.0380 |
| recall_top_1pct | 0.1667 |
| recall_top_5pct | 0.2619 |
| recall_top_10pct | 0.3095 |

### Novelty vs training
- **novel_fraction** : 0.9438856537850715
- **mean_max_sim** : 0.2631365667479107
- **median_max_sim** : 0.25
- **novelty_threshold** : 0.4

## 2. Validation DrugBank 35 ARV approuvés
- **Verdict** : FAIL
- **Recall@top-10%** : 15/34 (44.1%)
- **ARV trouvés** : Lenacapavir, Indinavir, Doravirine, Zalcitabine, Emtricitabine, Lamivudine, Nevirapine, Fosamprenavir, Saquinavir, Darunavir, Stavudine, Ritonavir, Didanosine, Amprenavir, Zidovudine
- **ARV manqués** : Efavirenz, Elvitegravir, Dolutegravir, Islatravir, Etravirine, Nelfinavir, Rilpivirine, Tenofovir, Cabotegravir, Maraviroc, Fostemsavir, Bictegravir, Lopinavir, Tenofovir_alafenamide, Tipranavir, Delavirdine, Atazanavir, Raltegravir, Abacavir

## 3. Conformal prediction (coverage garantie)
- **Coverage empirique** : 91.6% (cible ≥ 90.0%)
- **Coverage valide** : True
- **Décisions** : {'inactive': 930, 'ambiguous': 2514, 'active': 334}

## 4. Tanimoto Applicability Domain
- Similarité moyenne max au train : 0.434
- **in_domain_high** : 193
- **in_domain_moderate** : 1782
- **out_of_domain** : 1803
- **invalid** : 0
