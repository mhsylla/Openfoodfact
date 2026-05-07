import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models import (
    create_xgboost_model,
    create_random_forest_model,
    compute_sample_weights
)


def test_create_xgboost_model():
    """Test de la création du modèle XGBoost."""
    model = create_xgboost_model()
    
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert model.n_estimators == 100
    assert model.max_depth == 6


def test_create_random_forest_model():
    """Test de la création du modèle Random Forest."""
    model = create_random_forest_model()
    
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    assert model.n_estimators == 100
    assert model.class_weight == 'balanced'


def test_compute_sample_weights():
    """Test du calcul des poids d'échantillons."""
    y_train = np.array([0, 0, 0, 1, 1, 2])
    
    weights = compute_sample_weights(y_train)
    
    assert len(weights) == len(y_train)
    assert all(w > 0 for w in weights)
    
    assert weights[3] > weights[0]
    assert weights[5] > weights[0]


def test_xgboost_parameters():
    """Test des paramètres personnalisés XGBoost."""
    model = create_xgboost_model(n_estimators=50, max_depth=4, learning_rate=0.05)
    
    assert model.n_estimators == 50
    assert model.max_depth == 4
    assert model.learning_rate == 0.05


def test_random_forest_parameters():
    """Test des paramètres personnalisés Random Forest."""
    model = create_random_forest_model(n_estimators=50, max_depth=10)
    
    assert model.n_estimators == 50
    assert model.max_depth == 10
