# OpenFoodFacts - Modèle de Régression (Nutri-Score)

Ce répertoire contient les travaux de modélisation par régression visant à prédire le **Nutri-Score (`nutriscore_score`)** des produits alimentaires à partir de leurs valeurs nutritionnelles, en utilisant les données de la base de données OpenFoodFacts.

## 📌 Objectif du Projet

L'objectif principal est de développer un modèle d'apprentissage automatique (Random Forest) capable d'estimer avec précision le score nutritionnel continu (`nutriscore_score`) d'un produit en se basant sur ses caractéristiques intrinsèques (nutriments pour 100g).

## 🛠️ Méthodologie et Expérimentations

Le développement du modèle s'est déroulé en deux itérations principales. Ce processus a permis de mettre en évidence l'importance cruciale de la qualité et de la densité des données par rapport à leur simple volume.

### Itération 1 : Le "Plafond de Verre"

Lors de la première phase, une stratégie de conservation maximale des données a été adoptée :
- **Prétraitement :** Conservation des lignes où seul le nom du produit (`product_name`) était présent, avec une imputation par zéro (0) des données nutritionnelles manquantes.
- **Problème rencontré :** Cette approche a généré un volume massif (plus de 400 000 produits) de vecteurs mathématiquement identiques (valeurs nutritionnelles à 0) mais possédant des cibles (`nutriscore_score`) réelles très variées.
- **Conséquence :** Le modèle s'est retrouvé face à des contradictions mathématiques insolubles, engendrant un bruit statistique insurmontable qui a fortement limité l'apprentissage (score $R^2$ d'environ 0.63).

### Itération 2 : Amélioration de la Qualité des Données

Face aux limites de la première itération, un changement radical de stratégie a été opéré, plaçant la densité de l'information au cœur du processus de modélisation :
- **Nettoyage drastique :** 
  - Suppression définitive de la colonne textuelle `product_name` qui introduisait du bruit.
  - Suppression stricte des valeurs manquantes (`NaN`) dans la variable cible `nutriscore_score`.
  - Application de filtres de réalisme physique pour éliminer les valeurs aberrantes (ex: nutriments bornés entre 0 et 100g, énergie maximale à 4000 kJ).
- **Résultat du prétraitement :** L'application de ces filtres a permis de supprimer de facto l'intégralité des données incomplètes sans recours à l'imputation. Le dataset final obtenu compte **460 262 lignes** de profils nutritionnels réels, cohérents et complets.

## 🚀 Modélisation et Performances

Le modèle final a été optimisé à l'aide de la librairie **Optuna** sur ce jeu de données assaini, permettant de trouver la meilleure configuration d'hyperparamètres pour un modèle de *Random Forest*.

### Hyperparamètres Optimaux (Random Forest)
- `n_estimators`: 55
- `max_depth`: 18
- `min_samples_split`: 8
- `min_samples_leaf`: 2

### Performances sur le Jeu de Test
- **Score $R^2$ : `0.9626`**
  *(Le modèle explique désormais plus de 96% de la variance du Nutri-Score, une progression fulgurante par rapport à la première itération).*
- **Erreur Moyenne Absolue (MAE) : `0.9740` points**
  *(En moyenne, l'erreur de prédiction est inférieure à une unité sur l'échelle du score).*
- **Meilleur score de validation croisée (CV) :** `-0.9950`

### Analyse de l'Apprentissage
L'analyse des courbes d'apprentissage confirme l'efficacité de la nouvelle approche :
- **Convergence réussie :** Les erreurs d'entraînement et de validation convergent et se stabilisent sous la barre de 1.0 point de MAE.
- **Stabilité et Robustesse :** Une réduction drastique du biais et une excellente capacité de généralisation sans signe de surapprentissage (overfitting) significatif, prouvant que le modèle capte désormais la finesse du signal nutritionnel.

## 📂 Structure du Répertoire

- `notebooks/index.ipynb` : Notebook de référence contenant l'intégralité de l'exploration, des choix méthodologiques, du nettoyage, des visualisations, et de l'entraînement itératif du modèle détaillé ci-dessus.
- `scripts/` : Dossier contenant les scripts de traitement de données et d'entraînement.
- `images/` : Graphiques de preuves et d'analyses (ex. répartition des produits, courbes d'apprentissage).
- `main.py` : Point d'entrée principal des exécutions.

## ⚙️ Exécution

Les dépendances de ce projet sont gérées via `uv` / `poetry`. Les librairies requises peuvent être installées à partir des fichiers de verrouillage correspondants (`uv.lock`, `pyproject.toml`).