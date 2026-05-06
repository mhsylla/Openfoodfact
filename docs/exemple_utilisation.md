# Guide d'Utilisation - Prédiction du Nutri-Score

## Installation

### Prérequis
- Python 3.8+
- pip ou uv

### Installation des dépendances
```bash
# Avec pip
pip install -r requirements.txt

# Avec uv
uv sync
```

---

## 🚀 Démarrage Rapide

### 1. Prédiction avec un Modèle Pré-entraîné

```python
from src.prediction import load_model, predict_with_confidence
import pandas as pd

# Charger le modèle
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

---

## 📊 Entraîner un Nouveau Modèle

### Option 1 : Avec un Script Python

```bash
python scripts/train_model_clean2.py
```

### Option 2 : Avec un Notebook Jupyter

```bash
jupyter notebook notebooks/03_model_training_clean2.ipynb
```

### Option 3 : Programmatiquement

```python
from src.data_loader import load_data, get_features_and_target
from src.preprocessing import NutriscorePreprocessor
from src.feature_engineering import create_engineered_features
from src.models import create_random_forest_model
from src.training import train_model
from src.prediction import save_model

# 1. Charger les données
df = load_data('data/openfoodfacts_clean_2.csv')
X, y = get_features_and_target(df)

# 2. Feature engineering
X_eng = create_engineered_features(X)

# 3. Preprocessing
preprocessor = NutriscorePreprocessor()
X_train, X_test, y_train, y_test = preprocessor.fit_transform(X_eng, y)

# 4. Entraîner le modèle
model = create_random_forest_model()
model, metrics = train_model(model, X_train, y_train, X_test, y_test)

# 5. Sauvegarder
save_model(model, preprocessor, preprocessor.label_encoder, 
           'models/mon_modele.pkl')
```

---

## 🔍 Exemples d'Utilisation

### Exemple 1 : Prédiction Unitaire

```python
from src.prediction import load_model, predict_nutriscore
import pandas as pd

# Charger le modèle
model_data = load_model('models/nutriscore_model_clean2.pkl')

# Un seul produit
produit = pd.DataFrame([{
    'energy_100g': 2000,
    'fat_100g': 20.0,
    'saturated-fat_100g': 10.0,
    'carbohydrates_100g': 60.0,
    'sugars_100g': 30.0,
    'fiber_100g': 2.0,
    'proteins_100g': 5.0,
    'salt_100g': 2.0
}])

# Prédire
grade, probas = predict_nutriscore(model_data, produit)
print(f"Grade prédit : {grade[0]}")
print(f"Probabilités : {probas[0]}")
```

---

### Exemple 2 : Prédiction en Batch

```python
from src.prediction import load_model, predict_with_confidence
import pandas as pd

# Charger le modèle
model_data = load_model('models/nutriscore_model_clean2.pkl')

# Plusieurs produits
produits = pd.DataFrame([
    {'energy_100g': 1500, 'fat_100g': 10, 'saturated-fat_100g': 3, 
     'carbohydrates_100g': 50, 'sugars_100g': 5, 'fiber_100g': 3, 
     'proteins_100g': 8, 'salt_100g': 1},
    {'energy_100g': 2500, 'fat_100g': 25, 'saturated-fat_100g': 15, 
     'carbohydrates_100g': 70, 'sugars_100g': 40, 'fiber_100g': 1, 
     'proteins_100g': 3, 'salt_100g': 3},
    {'energy_100g': 500, 'fat_100g': 2, 'saturated-fat_100g': 0.5, 
     'carbohydrates_100g': 10, 'sugars_100g': 2, 'fiber_100g': 5, 
     'proteins_100g': 12, 'salt_100g': 0.5}
])

# Prédire
results = predict_with_confidence(model_data, produits)
print(results)
```

---

### Exemple 3 : Analyse de Reformulation

```python
from src.prediction import load_model, predict_nutriscore
import pandas as pd

model_data = load_model('models/nutriscore_model_clean2.pkl')

# Produit actuel
produit_actuel = pd.DataFrame([{
    'energy_100g': 2000,
    'fat_100g': 20.0,
    'saturated-fat_100g': 10.0,
    'carbohydrates_100g': 60.0,
    'sugars_100g': 30.0,
    'fiber_100g': 2.0,
    'proteins_100g': 5.0,
    'salt_100g': 2.0
}])

# Produit reformulé (réduction sel et sucre)
produit_reforme = produit_actuel.copy()
produit_reforme['salt_100g'] = 1.0
produit_reforme['sugars_100g'] = 15.0

# Comparer
grade_actuel, _ = predict_nutriscore(model_data, produit_actuel)
grade_reforme, _ = predict_nutriscore(model_data, produit_reforme)

print(f"Grade actuel : {grade_actuel[0]}")
print(f"Grade reformulé : {grade_reforme[0]}")
```

---

### Exemple 4 : Validation de Données

```python
from src.prediction import load_model, predict_nutriscore
import pandas as pd

model_data = load_model('models/nutriscore_model_clean2.pkl')

# Charger des produits avec Nutri-Score déclaré
df = pd.read_csv('data/openfoodfacts_clean_2.csv')
sample = df.sample(100)

# Prédire
features = sample[['energy_100g', 'fat_100g', 'saturated-fat_100g',
                   'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
                   'proteins_100g', 'salt_100g']]
grades_pred, _ = predict_nutriscore(model_data, features)

# Comparer
sample['nutriscore_predicted'] = grades_pred
sample['match'] = sample['nutriscore_grade'] == sample['nutriscore_predicted']

print(f"Taux de correspondance : {sample['match'].mean()*100:.1f}%")
print(f"\nIncohérences détectées :")
print(sample[~sample['match']][['nutriscore_grade', 'nutriscore_predicted']])
```

---

## 📈 Évaluation d'un Modèle

```python
from src.evaluation import evaluate_model
from src.data_loader import load_data, get_features_and_target
from src.preprocessing import NutriscorePreprocessor
from src.feature_engineering import create_engineered_features
from src.prediction import load_model

# Charger le modèle
model_data = load_model('models/nutriscore_model_clean2.pkl')

# Charger les données de test
df = load_data('data/openfoodfacts_clean_2.csv')
X, y = get_features_and_target(df)
X_eng = create_engineered_features(X)

# Preprocessing
preprocessor = model_data['preprocessor']
label_encoder = model_data['label_encoder']
X_scaled = preprocessor.transform(X_eng)
y_encoded = label_encoder.transform(y)

# Évaluer
results = evaluate_model(
    model_data['model'], 
    X_scaled, 
    y_encoded,
    label_encoder,
    save_path='models/evaluation'
)

print(f"Accuracy : {results['accuracy']:.4f}")
print(f"F1-Score : {results['f1_macro']:.4f}")
```

---

## 🔧 Personnalisation

### Créer des Features Personnalisées

```python
from src.feature_engineering import create_engineered_features
import pandas as pd

def create_custom_features(X):
    X_custom = create_engineered_features(X)
    
    # Ajouter vos propres features
    X_custom['healthy_score'] = (
        X_custom['proteins_100g'] + X_custom['fiber_100g']
    ) / (X_custom['sugars_100g'] + X_custom['salt_100g'] + 1)
    
    return X_custom
```

### Utiliser un Autre Modèle

```python
from sklearn.svm import SVC
from src.training import train_model

# Créer un SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# Entraîner
svm_model, metrics = train_model(
    svm_model, X_train, y_train, X_test, y_test
)
```

---

## ⚠️ Notes Importantes

### Données Requises
- **Toutes les 8 features doivent être présentes**
- Les valeurs doivent être positives
- Cohérence : sucres ≤ glucides, acides gras saturés ≤ lipides

### Performance
- Temps de prédiction : ~45ms pour 1000 produits
- Mémoire : ~200MB pour le modèle chargé

### Limitations
- Le modèle est entraîné sur des produits français/européens
- Certaines catégories peuvent avoir des performances variables
- Le modèle est basé sur la formule Nutri-Score actuelle

---

## 📚 Ressources

- **Documentation complète** : `docs/`
- **Notebooks d'exemple** : `notebooks/`
- **Tests** : `tests/`
- **Scripts** : `scripts/`

---

## 🆘 Aide

### Problèmes Courants

**Erreur : "Module not found"**
```bash
# Vérifier l'installation
pip list | grep xgboost
pip list | grep scikit-learn

# Réinstaller
pip install -r requirements.txt
```

**Erreur : "Model file not found"**
```bash
# Vérifier le chemin
ls models/

# Entraîner un nouveau modèle
python scripts/train_model_clean2.py
```

**Performance faible**
- Vérifier la qualité des données
- Réentraîner avec plus de données
- Optimiser les hyperparamètres

---

## 📞 Support

Pour toute question :
- **Documentation** : `docs/`
- **Issues** : GitHub Issues
- **Email** : support@project.com
