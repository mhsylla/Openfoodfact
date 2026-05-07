# Use Case : Prédiction du Nutri-Score

## Objectif Global
Permettre la prédiction automatique du Nutri-Score d'un produit alimentaire à partir de ses valeurs nutritionnelles.

---

## Checklist de Validation

### ✅ Fonctionnalités Core

- [ ] **UC-001** : Le système charge les données nutritionnelles d'un produit
- [ ] **UC-002** : Le système crée automatiquement les features dérivées
- [ ] **UC-003** : Le système normalise les données avec StandardScaler
- [ ] **UC-004** : Le système prédit le grade Nutri-Score (a, b, c, d, e)
- [ ] **UC-005** : Le système retourne les probabilités pour chaque grade
- [ ] **UC-006** : Le système calcule le niveau de confiance de la prédiction

### ✅ Performance

- [ ] **UC-007** : L'accuracy sur le test set est ≥ 85%
- [ ] **UC-008** : Le F1-Score macro est ≥ 0.80
- [ ] **UC-009** : L'accuracy ±1 grade est ≥ 95%
- [ ] **UC-010** : Le temps de prédiction est < 100ms pour 1000 produits

### ✅ Gestion des Données

- [ ] **UC-011** : Le système détecte les valeurs manquantes
- [ ] **UC-012** : Le système valide la cohérence des valeurs (sucres ≤ glucides)
- [ ] **UC-013** : Le système gère les valeurs aberrantes
- [ ] **UC-014** : Le système accepte les données en DataFrame ou dictionnaire

### ✅ Modèles

- [ ] **UC-015** : Le système peut utiliser XGBoost
- [ ] **UC-016** : Le système peut utiliser Random Forest
- [ ] **UC-017** : Le système sélectionne automatiquement le meilleur modèle
- [ ] **UC-018** : Le système sauvegarde le modèle entraîné au format .pkl

### ✅ Prédictions

- [ ] **UC-019** : Le système prédit un seul produit (prédiction unitaire)
- [ ] **UC-020** : Le système prédit plusieurs produits en batch
- [ ] **UC-021** : Le système retourne les résultats au format DataFrame
- [ ] **UC-022** : Le système charge un modèle pré-entraîné

### ✅ Visualisations

- [ ] **UC-023** : Le système génère la matrice de confusion
- [ ] **UC-024** : Le système affiche l'importance des features
- [ ] **UC-025** : Le système sauvegarde les graphiques en PNG
- [ ] **UC-026** : Le système affiche le rapport de classification

### ✅ Documentation

- [ ] **UC-027** : La documentation fonctionnelle est complète
- [ ] **UC-028** : Le CHANGELOG est à jour
- [ ] **UC-029** : Les exemples d'utilisation sont fournis
- [ ] **UC-030** : Le guide des améliorations est disponible

### ✅ Tests

- [ ] **UC-031** : Les tests end-to-end passent tous
- [ ] **UC-032** : La couverture de code est ≥ 80%
- [ ] **UC-033** : Les tests unitaires couvrent tous les modules
- [ ] **UC-034** : Les tests de performance sont validés

### ✅ Qualité du Code

- [ ] **UC-035** : Le code respecte PEP 8
- [ ] **UC-036** : Toutes les fonctions ont des docstrings
- [ ] **UC-037** : Les imports sont organisés
- [ ] **UC-038** : Pas de code dupliqué

---

## Scénarios d'Utilisation

### Scénario 1 : Prédiction Simple
**Acteur** : Data Scientist  
**Objectif** : Prédire le Nutri-Score d'un nouveau produit

**Étapes** :
1. Charger le modèle pré-entraîné
2. Préparer les données du produit (8 features)
3. Appeler la fonction de prédiction
4. Obtenir le grade et les probabilités

**Résultat attendu** : Grade prédit avec confiance > 80%

---

### Scénario 2 : Entraînement d'un Nouveau Modèle
**Acteur** : ML Engineer  
**Objectif** : Entraîner un modèle sur de nouvelles données

**Étapes** :
1. Charger le dataset (openfoodfacts_clean.csv ou clean_2.csv)
2. Séparer features et target
3. Appliquer le feature engineering
4. Entraîner XGBoost et Random Forest
5. Comparer les performances
6. Sauvegarder le meilleur modèle

**Résultat attendu** : Modèle avec accuracy > 85%

---

### Scénario 3 : Prédiction en Batch
**Acteur** : Application Backend  
**Objectif** : Prédire le Nutri-Score de 10 000 produits

**Étapes** :
1. Charger le modèle
2. Charger le fichier CSV des produits
3. Appliquer les transformations
4. Prédire en batch
5. Sauvegarder les résultats

**Résultat attendu** : Prédictions en < 5 secondes

---

### Scénario 4 : Analyse de Reformulation
**Acteur** : Nutritionniste  
**Objectif** : Simuler l'impact d'une réduction de sel

**Étapes** :
1. Charger les données du produit actuel
2. Modifier la valeur de `salt_100g`
3. Prédire le nouveau Nutri-Score
4. Comparer avec le grade actuel

**Résultat attendu** : Visualisation de l'impact de la modification

---

## Critères de Succès Global

La feature est considérée comme **fonctionnelle** si :

✅ **TOUTES** les cases de la checklist sont cochées  
✅ Les 4 scénarios d'utilisation fonctionnent sans erreur  
✅ La documentation est complète et à jour  
✅ Les tests passent à 100%  
✅ La couverture de code est ≥ 80%

---

## Validation Finale

**Date de validation** : _________  
**Validé par** : _________  
**Statut** : ⬜ Validé | ⬜ En cours | ⬜ À corriger

**Commentaires** :
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
