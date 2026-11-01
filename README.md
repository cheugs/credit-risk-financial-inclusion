# Étude du risque de crédit et de l'inclusion financière

**Français** | [English](README_en.md)

**Problématique :** Quels facteurs sont associés au défaut de remboursement, et comment les
prêteurs peuvent-ils évaluer de manière équitable les clients mal desservis ?

**Outils :** Python, pandas, SQL (SQLite), SciPy, statsmodels, scikit-learn, SHAP.

**Données :** [Statlog German Credit](https://doi.org/10.24432/C5NC77) (UCI), 1 000 crédits à la
consommation, 30 % de mauvais crédits. La table des codes fournie avec ces données est fausse pour
8 variables : nous utilisons donc la table corrigée de Grömping (2019). Voir le rapport, section 3.2.

## Principaux résultats

| Thème | Résultat |
| --- | --- |
| Facteurs les plus forts | Compte courant, historique de crédit, durée du prêt, épargne |
| Modèle | Régression logistique sans sexe, âge ni statut de travailleur étranger. ROC-AUC sur le test = 0,80 |
| Règle de décision | Seuil P(défaut) = 1/6, issu de la matrice des coûts 5:1. Coût inférieur de 22 % à « tout refuser » |
| Accès au crédit | À ce seuil, seulement 40 % des demandeurs et 51 % des bons clients sont acceptés |
| Classes de risque | Taux de défaut de 6,5 % (classe la plus sûre) à 62,5 % (classe la plus risquée) |
| Équité | Les clients de 25 ans ou moins sont beaucoup moins acceptés, alors que le modèle n'utilise pas l'âge (impact disparate = 0,65) |

> Ces résultats sont des corrélations observées dans des données historiques de 1973-1975. Ils ne
> prouvent pas qu'une personne est solvable ou non.

## Rapports

- Rapport en français : [report/rapport_fr.md](report/rapport_fr.md) ([PDF](report/rapport_fr.pdf))
- Rapport en anglais : [report/report_en.md](report/report_en.md) ([PDF](report/report_en.pdf))
- Notebook pas à pas en français : [notebooks/etude_risque_credit.ipynb](notebooks/etude_risque_credit.ipynb)
- Notebook pas à pas en anglais : [notebooks/credit_risk_study.ipynb](notebooks/credit_risk_study.ipynb)

## Structure du projet

```
data/raw/            german.data (données d'origine) et german.doc (documentation d'origine)
data/processed/      CSV nettoyé et base SQLite (créés par l'étape 1)
src/
  common.py          paramètres, table des codes corrigée, chargement des données, modèle et coûts
  01_data_prep.py    étape 1 : préparation des données et requêtes SQL
  02_statistics.py   étape 2 : khi-deux, Mann-Whitney, régression logistique (odds ratios)
  03_model.py        étape 3 : modèle, déséquilibre des classes, seuil de décision, classes de risque
  04_shap.py         étape 4 : explications SHAP
  05_fairness.py     étape 5 : audit d'équité avec intervalles de confiance bootstrap
  run_all.py         lance les étapes 1 à 5
notebooks/           notebooks pas à pas avec les résultats (français et anglais)
results/             graphiques, tableaux et modèle enregistré (créés par les scripts)
report/              rapport complet en français et en anglais (Markdown et PDF)
```

## Exécution

```bash
pip install -r requirements.txt
python src/run_all.py
```

La graine aléatoire est fixée : chaque exécution donne les mêmes résultats.

## Méthode en cinq étapes

1. **Données et SQL :** application de la table des codes corrigée, stockage dans SQLite, calcul
   des taux de défaut par segment en SQL.
2. **Statistiques :** tests du khi-deux avec V de Cramér, tests de Mann-Whitney U avec corrélation
   bisériale de rang, correction de Bonferroni, et régression logistique avec odds ratios.
3. **Modèle :** régression logistique sans données sensibles. Comparaison d'un modèle normal et
   d'un modèle « équilibré » (même classement, mais seul le modèle normal donne des probabilités
   réalistes). Seuil issu de la matrice des coûts, comparaison avec des règles simples, 5 classes
   de risque.
4. **SHAP :** importance globale des variables et explications pour un client donné.
5. **Équité :** impact disparate et égalité des chances selon l'âge et le statut de travailleur
   étranger, avec intervalles de confiance bootstrap, et comparaison d'un modèle « aveugle » et
   d'un modèle « complet ».

## Limites

- Seuls les clients acceptés figurent dans les données (biais de sélection), et les mauvais
  crédits ont été sur-représentés (30 % dans l'échantillon, environ 5 % en réalité).
- Données petites et anciennes (1 000 crédits, sud de l'Allemagne, 1973-1975).
- Le sexe ne peut pas être retrouvé dans les données : il n'est donc pas audité. Seuls 37 clients
  sont des travailleurs étrangers : les résultats pour ce groupe sont incertains.

## Principales références

- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set*. Report 4/2019, Beuth University of Applied Sciences Berlin.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
- Kozodoi, N., Jacob, J., & Lessmann, S. (2022). Fairness in credit scoring. *European Journal of Operational Research, 297*(3).

La liste complète des références se trouve à la fin de chaque rapport.
