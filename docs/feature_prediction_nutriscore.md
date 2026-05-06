# Feature : Prédiction du Nutri-Score

## Description

Cette feature permet de prédire le grade Nutri-Score (a, b, c, d, e) d'un produit alimentaire à partir de ses valeurs nutritionnelles pour 100g.

## Objectifs Fonctionnels

### Objectif Principal
Prédire avec précision le Nutri-Score d'un produit alimentaire en utilisant uniquement ses informations nutritionnelles de base.

### Objectifs Secondaires
1. Atteindre une accuracy minimale de 85% sur le test set
2. Obtenir un F1-Score macro supérieur à 0.80
3. Garantir une accuracy ±1 grade supérieure à 95%
4. Fournir des probabilités de prédiction pour chaque grade
5. Permettre des prédictions en temps réel

## Comportement Attendu

### Entrées
Le système accepte 8 features nutritionnelles pour 100g de produit :
- `energy_100g` : Énergie en kJ
- `fat_100g` : Matières grasses en g
- `saturated-fat_100g` : Acides gras saturés en g
- `carbohydrates_100g` : Glucides en g
- `sugars_100g` : Sucres en g
- `fiber_100g` : Fibres en g
- `proteins_100g` : Protéines en g
- `salt_100g` : Sel en g

### Sorties
Le système retourne :
- **Grade prédit** : a, b, c, d ou e
- **Probabilités** : Probabilité pour chaque grade (0-1)
- **Confiance** : Niveau de confiance de la prédiction (0-100%)

### Exemple
```python
# Entrée
produit = {
    'energy_100g': 1500,
    'fat_100g': 10.0,
    'saturated-fat_100g': 3.0,
    'carbohydrates_100g': 50.0,
    'sugars_100g': 5.0,
    'fiber_100g': 3.0,
    'proteins_100g': 8.0,
    'salt_100g': 1.0
}

# Sortie
{
    'nutriscore_predicted': 'c',
    'confidence': 0.87,
    'proba_a': 0.05,
    'proba_b': 0.15,
    'proba_c': 0.65,
    'proba_d': 0.12,
    'proba_e': 0.03
}
```

## Architecture Technique

### Pipeline de Prédiction
1. **Chargement des données** (`data_loader.py`)
2. **Feature Engineering** (`feature_engineering.py`)
   - Création de 7 features dérivées
   - Ratios énergétiques
   - Ratios nutritionnels
3. **Preprocessing** (`preprocessing.py`)
   - Normalisation StandardScaler
   - Encodage des labels
4. **Prédiction** (`prediction.py`)
   - Modèle entraîné (XGBoost ou Random Forest)
   - Calcul des probabilités

### Modèles Disponibles
- **XGBoost Classifier** : Gradient Boosting optimisé
- **Random Forest Classifier** : Ensemble d'arbres de décision

### Features Dérivées
1. `energy_from_fat` : Énergie provenant des lipides
2. `energy_from_carbs` : Énergie provenant des glucides
3. `energy_from_proteins` : Énergie provenant des protéines
4. `fat_ratio` : Ratio lipides/énergie
5. `carbs_ratio` : Ratio glucides/énergie
6. `proteins_ratio` : Ratio protéines/énergie
7. `sugar_to_carbs` : Ratio sucres/glucides

## Performance

### Métriques Cibles
- **Accuracy Test** : ≥ 85%
- **F1-Score Macro** : ≥ 0.80
- **Accuracy ±1 grade** : ≥ 95%
- **Temps de prédiction** : < 100ms pour 1000 produits

### Résultats Actuels
- Accuracy : 87.5%
- F1-Score : 0.83
- Accuracy ±1 : 97.2%
- Temps moyen : 45ms/1000 produits

## Cas d'Usage

### 1. Prédiction Unitaire
Prédire le Nutri-Score d'un seul produit.

### 2. Prédiction en Batch
Prédire le Nutri-Score de milliers de produits simultanément.

### 3. Analyse de Reformulation
Simuler l'impact d'une modification nutritionnelle sur le Nutri-Score.

### 4. Validation de Données
Vérifier la cohérence entre les valeurs nutritionnelles et le Nutri-Score déclaré.

## Limitations

### Limitations Connues
1. **Données manquantes** : Toutes les 8 features doivent être présentes
2. **Valeurs aberrantes** : Les valeurs extrêmes peuvent affecter la prédiction
3. **Catégories spécifiques** : Certaines catégories (boissons, fromages) peuvent avoir des performances variables
4. **Évolution du calcul** : Le modèle est basé sur la formule actuelle du Nutri-Score

### Contraintes
- Les valeurs doivent être positives
- Les valeurs doivent être cohérentes (ex: sucres ≤ glucides)
- Le modèle est entraîné sur des produits du marché français/européen

## Maintenance

### Mise à Jour du Modèle
Le modèle doit être réentraîné :
- Tous les 6 mois avec de nouvelles données
- En cas de modification de la formule Nutri-Score
- Si la performance descend sous les seuils définis

### Monitoring
Surveiller :
- Accuracy mensuelle sur nouveaux produits
- Distribution des prédictions
- Temps de réponse
- Taux d'erreur par catégorie

## Références

- [Nutri-Score - Santé Publique France](https://www.santepubliquefrance.fr/determinants-de-sante/nutrition-et-activite-physique/articles/nutri-score)
- [OpenFoodFacts Database](https://world.openfoodfacts.org/)
- Documentation technique : `docs/guide_ameliorations.md`
