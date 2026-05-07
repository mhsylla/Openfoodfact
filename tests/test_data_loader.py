import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, get_features_and_target, save_data


def test_load_data():
    """Test du chargement des données."""
    data_path = Path(__file__).parent.parent / 'data' / 'openfoodfacts_clean_2.csv'
    
    if not data_path.exists():
        pytest.skip("Dataset clean_2 non disponible")
    
    df = load_data(str(data_path))
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert 'nutriscore_grade' in df.columns


def test_get_features_and_target():
    """Test de la séparation features/target."""
    df = pd.DataFrame({
        'energy_100g': [1000, 1500, 2000],
        'fat_100g': [5, 10, 15],
        'saturated-fat_100g': [2, 4, 6],
        'carbohydrates_100g': [20, 30, 40],
        'sugars_100g': [5, 10, 15],
        'fiber_100g': [2, 3, 4],
        'proteins_100g': [8, 10, 12],
        'salt_100g': [0.5, 1.0, 1.5],
        'nutriscore_grade': ['a', 'b', 'c']
    })
    
    X, y = get_features_and_target(df)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 3
    assert len(y) == 3
    assert X.shape[1] == 8
    assert 'nutriscore_grade' not in X.columns


def test_save_data(tmp_path):
    """Test de la sauvegarde des données."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    output_path = tmp_path / 'test_output.csv'
    save_data(df, str(output_path))
    
    assert output_path.exists()
    
    df_loaded = pd.read_csv(output_path)
    assert len(df_loaded) == 3
    assert list(df_loaded.columns) == ['col1', 'col2']
