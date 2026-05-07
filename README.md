# 🥗 Prédiction du Nutri-Score

Modèle de Machine Learning pour prédire le grade Nutri-Score (a, b, c, d, e) d'un produit alimentaire à partir de ses valeurs nutritionnelles.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](tests/)

---

## 📋 Table des Matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Démarrage Rapide](#démarrage-rapide)
- [Documentation](#documentation)
- [Structure du Projet](#structure-du-projet)
- [Tests](#tests)
- [Performance](#performance)
- [Contribuer](#contribuer)
- [License](#license)

---

## 🎯 Aperçu

Ce projet utilise des algorithmes de Machine Learning (XGBoost et Random Forest) pour prédire automatiquement le Nutri-Score d'un produit alimentaire basé sur 8 valeurs nutritionnelles pour 100g.

### Nutri-Score
Le Nutri-Score est un système d'étiquetage nutritionnel à 5 niveaux (a, b, c, d, e) qui permet de comparer facilement la qualité nutritionnelle des aliments.

### Dataset
- **Source** : [OpenFoodFacts](https://world.openfoodfacts.org/)
- **Taille** : 460 000+ produits
- **Features** : 8 valeurs nutritionnelles + 7 features dérivées

---

## ✨ Fonctionnalités

- ✅ **Prédiction précise** : Accuracy > 88%, F1-Score > 0.84
- ✅ **Feature Engineering** : 15 features (8 originales + 7 dérivées)
- ✅ **Deux modèles** : XGBoost et Random Forest
- ✅ **Gestion du déséquilibre** : Poids d'échantillons et class balancing
- ✅ **Prédictions en batch** : Traitement de milliers de produits
- ✅ **Visualisations** : Matrice de confusion, importance des features
- ✅ **Tests complets** : Tests end-to-end avec pytest
- ✅ **Documentation complète** : Guides, use cases, exemples

---

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip ou uv

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/mhsylla/Openfoodfact.git
cd Openfoodfact

# Installer les dépendances
pip install -r requirements.txt

# Ou avec uv
uv sync
```

---

## ⚡ Démarrage Rapide

### 1. Prédiction Simple

```python
from src.prediction import load_model, predict_with_confidence
import pandas as pd

# Charger le modèle pré-entraîné
model_data = load_model('models/nutriscore_model_clean2.pkl')

# Préparer les données d'un produit
produit = pd.DataFrame([{
    'energy_100g': 1500,
    'fat_100g': 10.0,
    'saturated-fat_100g': 3.0,
    'carbohydrates_100g': 50.0,
    'sugars_100g': 5.0,
    'fiber_100g': 3.0,
    'proteins_100g': 8.0,
    'salt_100g': 1.0
}])

# Prédire
results = predict_with_confidence(model_data, produit)
print(results)
```

**Sortie** :
```
  nutriscore_predicted  confidence  proba_a  proba_b  proba_c  proba_d  proba_e
0                    c        0.87     0.05     0.15     0.65     0.12     0.03
```

### 2. Entraîner un Modèle

```bash
# Avec un script
python scripts/train_model_clean2.py

# Avec un notebook
jupyter notebook notebooks/03_model_training_clean2.ipynb
```

### 3. Lancer les Tests

```bash
# Tous les tests
pytest

# Avec couverture de code
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_end_to_end.py -v
```

---

## 📚 Documentation

### Documentation Complète
- **[Guide d'Utilisation](docs/exemple_utilisation.md)** : Exemples détaillés
- **[Documentation Fonctionnelle](docs/feature_prediction_nutriscore.md)** : Description de la feature
- **[Use Cases](docs/use_case_prediction.md)** : Scénarios d'utilisation
- **[Guide des Améliorations](docs/guide_ameliorations.md)** : Améliorations implémentées
- **[Rapport Qualité](docs/quality_report.md)** : Tests et couverture
- **[CHANGELOG](docs/CHANGELOG.md)** : Historique des modifications

### Notebooks
- `02_model_training.ipynb` : Entraînement sur dataset original
- `03_model_training_clean2.ipynb` : Entraînement sur clean_2
- `04_compare_models.ipynb` : Comparaison des modèles

---

## 📁 Structure du Projet

```
Openfoodfact/
├── data/                           # Datasets
│   ├── openfoodfacts_clean.csv
│   └── openfoodfacts_clean_2.csv
├── src/                            # Code source
│   ├── __init__.py
│   ├── data_loader.py             # Chargement des données
│   ├── preprocessing.py           # Preprocessing
│   ├── feature_engineering.py     # Feature engineering
│   ├── models.py                  # Modèles ML
│   ├── training.py                # Entraînement
│   ├── evaluation.py              # Évaluation
│   └── prediction.py              # Prédictions
├── scripts/                        # Scripts d'exécution
│   ├── train_model_clean2.py
│   ├── compare_datasets.py
│   └── process_data.py
├── notebooks/                      # Notebooks Jupyter
│   ├── 02_model_training.ipynb
│   ├── 03_model_training_clean2.ipynb
│   └── 04_compare_models.ipynb
├── tests/                          # Tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_end_to_end.py
├── docs/                           # Documentation
│   ├── CHANGELOG.md
│   ├── feature_prediction_nutriscore.md
│   ├── use_case_prediction.md
│   ├── quality_report.md
│   ├── exemple_utilisation.md
│   └── guide_ameliorations.md
├── models/                         # Modèles sauvegardés
├── pytest.ini                      # Configuration pytest
├── .coveragerc                     # Configuration couverture
├── requirements.txt                # Dépendances
├── pyproject.toml                  # Configuration projet
└── README.md                       # Ce fichier
```

---

## 🧪 Tests

### Exécuter les Tests

```bash
# Tous les tests
pytest

# Tests avec verbosité
pytest -v

# Tests avec couverture
pytest --cov=src --cov-report=html

# Tests spécifiques
pytest tests/test_end_to_end.py
pytest tests/test_models.py
```

### Couverture de Code

Objectif : **≥ 80%**

```bash
# Générer le rapport de couverture
pytest --cov=src --cov-report=html

# Ouvrir le rapport
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html # Windows
```

---

## 📊 Performance

### Métriques

| Métrique | Objectif | Actuel | Statut |
|----------|----------|--------|--------|
| **Accuracy Test** | ≥ 85% | 88.23% | ✅ |
| **F1-Score Macro** | ≥ 0.80 | 0.8445 | ✅ |
| **Accuracy ±1 grade** | ≥ 95% | 97.2% | ✅ |
| **Temps/1000 produits** | < 100ms | 45ms | ✅ |

### Modèles

**Random Forest** (Meilleur)
- Accuracy Train : 95.12%
- Accuracy Test : 88.23%
- F1-Score : 0.8445
- Temps : 67.8s

**XGBoost**
- Accuracy Train : 92.45%
- Accuracy Test : 87.56%
- F1-Score : 0.8312
- Temps : 45.2s

---

## 🛠️ Utilisation Avancée

### Personnaliser les Features

```python
from src.feature_engineering import create_engineered_features

def create_custom_features(X):
    X_custom = create_engineered_features(X)
    
    # Ajouter vos features
    X_custom['healthy_score'] = (
        X_custom['proteins_100g'] + X_custom['fiber_100g']
    ) / (X_custom['sugars_100g'] + X_custom['salt_100g'] + 1)
    
    return X_custom
```

### Optimiser les Hyperparamètres

```python
from sklearn.model_selection import GridSearchCV
from src.models import create_random_forest_model

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = create_random_forest_model()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

print(f"Meilleurs paramètres : {grid_search.best_params_}")
```

---

## 🤝 Contribuer

Les contributions sont les bienvenues !

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout amélioration'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

### Guidelines
- Suivre PEP 8
- Ajouter des tests
- Mettre à jour la documentation
- Vérifier que tous les tests passent

---

## 📝 License

Ce projet est sous license MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📞 Contact

- **Projet** : [Openfoodfact](https://github.com/mhsylla/Openfoodfact)
- **Issues** : [GitHub Issues](https://github.com/mhsylla/Openfoodfact/issues)
- **Documentation** : `docs/`

---

## 🙏 Remerciements

- [OpenFoodFacts](https://world.openfoodfacts.org/) pour les données
- [Santé Publique France](https://www.santepubliquefrance.fr/) pour le Nutri-Score
- Communauté Python et scikit-learn

---

## 📈 Roadmap

### Court Terme
- [ ] Optimisation des hyperparamètres
- [ ] Validation croisée
- [ ] Features catégorielles (catégories, marques, pays)

### Moyen Terme
- [ ] API REST
- [ ] Interface web
- [ ] Déploiement cloud

### Long Terme
- [ ] Monitoring en production
- [ ] Réentraînement automatique
- [ ] Modèles spécialisés par catégorie

---

**Made with ❤️ for better nutrition**