import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import create_engineered_features, create_ratio_features


def test_create_engineered_features():
    """Test de la création des features dérivées."""
    X = pd.DataFrame({
        'energy_100g': [1000, 1500, 2000],
        'fat_100g': [5, 10, 15],
        'saturated-fat_100g': [2, 4, 6],
        'carbohydrates_100g': [20, 30, 40],
        'sugars_100g': [5, 10, 15],
        'fiber_100g': [2, 3, 4],
        'proteins_100g': [8, 10, 12],
        'salt_100g': [0.5, 1.0, 1.5]
    })
    
    X_eng = create_engineered_features(X)
    
    assert isinstance(X_eng, pd.DataFrame)
    assert X_eng.shape[1] == 15
    
    assert 'energy_from_fat' in X_eng.columns
    assert 'energy_from_carbs' in X_eng.columns
    assert 'energy_from_proteins' in X_eng.columns
    assert 'fat_ratio' in X_eng.columns
    assert 'carbs_ratio' in X_eng.columns
    assert 'proteins_ratio' in X_eng.columns
    assert 'sugar_to_carbs' in X_eng.columns


def test_energy_calculations():
    """Test des calculs d'énergie."""
    X = pd.DataFrame({
        'energy_100g': [1000],
        'fat_100g': [10],
        'saturated-fat_100g': [4],
        'carbohydrates_100g': [30],
        'sugars_100g': [10],
        'fiber_100g': [3],
        'proteins_100g': [8],
        'salt_100g': [1.0]
    })
    
    X_eng = create_engineered_features(X)
    
    assert X_eng['energy_from_fat'].iloc[0] == 10 * 9
    assert X_eng['energy_from_carbs'].iloc[0] == 30 * 4
    assert X_eng['energy_from_proteins'].iloc[0] == 8 * 4


def test_ratio_calculations():
    """Test des calculs de ratios."""
    X = pd.DataFrame({
        'energy_100g': [1000],
        'fat_100g': [10],
        'saturated-fat_100g': [4],
        'carbohydrates_100g': [30],
        'sugars_100g': [10],
        'fiber_100g': [3],
        'proteins_100g': [8],
        'salt_100g': [1.0]
    })
    
    X_eng = create_engineered_features(X)
    
    assert X_eng['fat_ratio'].iloc[0] == pytest.approx(10 / 1001, rel=1e-3)
    assert X_eng['sugar_to_carbs'].iloc[0] == pytest.approx(10 / 31, rel=1e-3)


def test_create_ratio_features():
    """Test de la création des ratios supplémentaires."""
    X = pd.DataFrame({
        'energy_100g': [1000],
        'fat_100g': [10],
        'saturated-fat_100g': [4],
        'carbohydrates_100g': [30],
        'sugars_100g': [10],
        'fiber_100g': [3],
        'proteins_100g': [8],
        'salt_100g': [1.0]
    })
    
    X_ratio = create_ratio_features(X)
    
    assert 'fat_to_protein' in X_ratio.columns
    assert 'sugar_to_fiber' in X_ratio.columns
    assert X_ratio['fat_to_protein'].iloc[0] == pytest.approx(10 / 9, rel=1e-3)
