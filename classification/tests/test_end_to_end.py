import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, get_features_and_target
from src.preprocessing import NutriscorePreprocessor
from src.feature_engineering import create_engineered_features
from src.models import create_random_forest_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.prediction import save_model, load_model, predict_nutriscore


def test_full_pipeline(tmp_path):
    """Test du pipeline complet de bout en bout."""
    data_path = Path(__file__).parent.parent / 'data' / 'openfoodfacts_clean_2.csv'
    
    if not data_path.exists():
        pytest.skip("Dataset clean_2 non disponible")
    
    df = load_data(str(data_path))
    sample = df.sample(n=min(1000, len(df)), random_state=42)
    
    X, y = get_features_and_target(sample)
    
    X_eng = create_engineered_features(X)
    assert X_eng.shape[1] == 15
    
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X_eng, y, test_size=0.3, random_state=42
    )
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    
    model = create_random_forest_model(n_estimators=10)
    model, metrics = train_model(model, X_train, y_train, X_test, y_test)
    
    assert metrics['test_accuracy'] > 0.5
    assert metrics['test_f1'] > 0.3
    assert metrics['training_time'] > 0
    
    model_path = tmp_path / 'test_model.pkl'
    save_model(model, preprocessor, preprocessor.label_encoder, str(model_path))
    assert model_path.exists()
    
    model_data = load_model(str(model_path))
    assert 'model' in model_data
    assert 'preprocessor' in model_data
    assert 'label_encoder' in model_data
    
    X_pred_sample = X_eng.sample(n=5, random_state=42)
    predictions, probabilities = predict_nutriscore(model_data, X_pred_sample)
    
    assert len(predictions) == 5
    assert len(probabilities) == 5
    assert all(grade in ['a', 'b', 'c', 'd', 'e'] for grade in predictions)


def test_pipeline_with_small_dataset():
    """Test du pipeline avec un petit dataset synthétique."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'energy_100g': np.random.randint(500, 3000, 100),
        'fat_100g': np.random.uniform(0, 30, 100),
        'saturated-fat_100g': np.random.uniform(0, 15, 100),
        'carbohydrates_100g': np.random.uniform(0, 80, 100),
        'sugars_100g': np.random.uniform(0, 40, 100),
        'fiber_100g': np.random.uniform(0, 10, 100),
        'proteins_100g': np.random.uniform(0, 20, 100),
        'salt_100g': np.random.uniform(0, 3, 100),
        'nutriscore_grade': np.random.choice(['a', 'b', 'c', 'd', 'e'], 100)
    })
    
    X, y = get_features_and_target(df)
    X_eng = create_engineered_features(X)
    
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X_eng, y, test_size=0.3, random_state=42
    )
    
    model = create_random_forest_model(n_estimators=10)
    model, metrics = train_model(model, X_train, y_train, X_test, y_test)
    
    assert metrics['test_accuracy'] >= 0
    assert metrics['test_accuracy'] <= 1
    assert metrics['test_f1'] >= 0
    assert metrics['test_f1'] <= 1


def test_prediction_consistency(tmp_path):
    """Test de la cohérence des prédictions."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'energy_100g': np.random.randint(500, 3000, 100),
        'fat_100g': np.random.uniform(0, 30, 100),
        'saturated-fat_100g': np.random.uniform(0, 15, 100),
        'carbohydrates_100g': np.random.uniform(0, 80, 100),
        'sugars_100g': np.random.uniform(0, 40, 100),
        'fiber_100g': np.random.uniform(0, 10, 100),
        'proteins_100g': np.random.uniform(0, 20, 100),
        'salt_100g': np.random.uniform(0, 3, 100),
        'nutriscore_grade': np.random.choice(['a', 'b', 'c', 'd', 'e'], 100)
    })
    
    X, y = get_features_and_target(df)
    X_eng = create_engineered_features(X)
    
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X_eng, y, test_size=0.3, random_state=42
    )
    
    model = create_random_forest_model(n_estimators=10, random_state=42)
    model, _ = train_model(model, X_train, y_train, X_test, y_test)
    
    model_path = tmp_path / 'consistency_model.pkl'
    save_model(model, preprocessor, preprocessor.label_encoder, str(model_path))
    
    model_data = load_model(str(model_path))
    
    X_sample = X_eng.iloc[:5]
    pred1, prob1 = predict_nutriscore(model_data, X_sample)
    pred2, prob2 = predict_nutriscore(model_data, X_sample)
    
    assert np.array_equal(pred1, pred2)
    assert np.allclose(prob1, prob2)
