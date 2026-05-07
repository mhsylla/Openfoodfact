import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import NutriscorePreprocessor


def test_preprocessor_fit_transform():
    """Test du preprocessing complet."""
    X = pd.DataFrame({
        'energy_100g': [1000, 1500, 2000, 2500, 3000],
        'fat_100g': [5, 10, 15, 20, 25],
        'saturated-fat_100g': [2, 4, 6, 8, 10],
        'carbohydrates_100g': [20, 30, 40, 50, 60],
        'sugars_100g': [5, 10, 15, 20, 25],
        'fiber_100g': [2, 3, 4, 5, 6],
        'proteins_100g': [8, 10, 12, 14, 16],
        'salt_100g': [0.5, 1.0, 1.5, 2.0, 2.5]
    })
    
    y = pd.Series(['a', 'b', 'c', 'd', 'e'])
    
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y, test_size=0.4, random_state=42)
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert X_train.shape[1] == 8
    
    assert np.allclose(X_train.mean(), 0, atol=0.1)
    assert np.allclose(X_train.std(), 1, atol=0.2)


def test_preprocessor_transform():
    """Test de la transformation de nouvelles données."""
    X_train = pd.DataFrame({
        'energy_100g': [1000, 1500, 2000],
        'fat_100g': [5, 10, 15],
        'saturated-fat_100g': [2, 4, 6],
        'carbohydrates_100g': [20, 30, 40],
        'sugars_100g': [5, 10, 15],
        'fiber_100g': [2, 3, 4],
        'proteins_100g': [8, 10, 12],
        'salt_100g': [0.5, 1.0, 1.5]
    })
    
    y_train = pd.Series(['a', 'b', 'c'])
    
    preprocessor = NutriscorePreprocessor()
    preprocessor.fit_transform(X_train, y_train, test_size=0.0, random_state=42)
    
    X_new = pd.DataFrame({
        'energy_100g': [1200],
        'fat_100g': [7],
        'saturated-fat_100g': [3],
        'carbohydrates_100g': [25],
        'sugars_100g': [7],
        'fiber_100g': [2.5],
        'proteins_100g': [9],
        'salt_100g': [0.7]
    })
    
    X_scaled = preprocessor.transform(X_new)
    
    assert isinstance(X_scaled, pd.DataFrame)
    assert len(X_scaled) == 1
    assert X_scaled.shape[1] == 8


def test_label_encoder():
    """Test de l'encodage des labels."""
    X = pd.DataFrame({
        'energy_100g': [1000, 1500, 2000, 2500, 3000],
        'fat_100g': [5, 10, 15, 20, 25],
        'saturated-fat_100g': [2, 4, 6, 8, 10],
        'carbohydrates_100g': [20, 30, 40, 50, 60],
        'sugars_100g': [5, 10, 15, 20, 25],
        'fiber_100g': [2, 3, 4, 5, 6],
        'proteins_100g': [8, 10, 12, 14, 16],
        'salt_100g': [0.5, 1.0, 1.5, 2.0, 2.5]
    })
    
    y = pd.Series(['a', 'b', 'c', 'd', 'e'])
    
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y, test_size=0.4, random_state=42)
    
    assert all(isinstance(label, (int, np.integer)) for label in y_train)
    assert all(isinstance(label, (int, np.integer)) for label in y_test)
    
    assert set(preprocessor.label_encoder.classes_) == {'a', 'b', 'c', 'd', 'e'}
