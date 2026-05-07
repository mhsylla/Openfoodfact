# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.0] - 2026-05-06

### Ajouté
- **Modules source complets** (`src/`)
  - `data_loader.py` : Chargement et manipulation des données
  - `preprocessing.py` : Preprocessing et normalisation
  - `feature_engineering.py` : Création de features dérivées
  - `models.py` : Création des modèles XGBoost et Random Forest
  - `training.py` : Entraînement des modèles
  - `evaluation.py` : Évaluation et visualisations
  - `prediction.py` : Prédictions et sauvegarde/chargement des modèles

- **Scripts d'entraînement**
  - `scripts/train_model_clean2.py` : Entraînement sur dataset clean_2
  - `scripts/compare_datasets.py` : Comparaison des datasets
  - `scripts/process_data.py` : Traitement des données

- **Notebooks Jupyter**
  - `02_model_training.ipynb` : Entraînement interactif (dataset original)
  - `03_model_training_clean2.ipynb` : Entraînement sur clean_2
  - `04_compare_models.ipynb` : Comparaison des modèles
  - `project_starter.ipynb` : Exploration initiale

- **Documentation complète**
  - CHANGELOG.md (ce fichier)
  - feature_prediction_nutriscore.md : Documentation fonctionnelle
  - use_case_prediction.md : Use cases et objectifs
  - quality_report.md : Rapport qualité et tests
  - exemple_utilisation.md : Guide d'utilisation
  - guide_ameliorations.md : Guide des améliorations

- **Tests end-to-end**
  - Tests complets avec pytest
  - Couverture de code mesurée
  - Tests de tous les modules

### Modifié
- README.md : Documentation complète du projet

### Technique
- **Dataset** : OpenFoodFacts avec 8 features nutritionnelles
- **Modèles** : XGBoost et Random Forest
- **Features** : 15 features (8 originales + 7 dérivées)
- **Performance** : Accuracy > 85%, F1-Score > 0.80
- **Métriques** : Accuracy, F1-Score macro/weighted, Accuracy ±1 grade

## [0.1.0] - 2026-04-27

### Ajouté
- Structure initiale du projet
- Exploration des données
- Premiers notebooks d'analyse

---

## Format des Entrées

### Types de Changements
- **Ajouté** : nouvelles fonctionnalités
- **Modifié** : changements dans les fonctionnalités existantes
- **Déprécié** : fonctionnalités bientôt supprimées
- **Supprimé** : fonctionnalités supprimées
- **Corrigé** : corrections de bugs
- **Sécurité** : en cas de vulnérabilités
