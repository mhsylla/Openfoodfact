from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def create_xgboost_model(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
    """
    Crée un modèle XGBoost Classifier.
    
    Args:
        n_estimators (int): Nombre d'arbres
        max_depth (int): Profondeur maximale
        learning_rate (float): Taux d'apprentissage
        random_state (int): Seed
        
    Returns:
        XGBClassifier: Modèle XGBoost
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='mlogloss'
    )
    return model


def create_random_forest_model(n_estimators=100, max_depth=None, random_state=42):
    """
    Crée un modèle Random Forest Classifier.
    
    Args:
        n_estimators (int): Nombre d'arbres
        max_depth (int): Profondeur maximale
        random_state (int): Seed
        
    Returns:
        RandomForestClassifier: Modèle Random Forest
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced'
    )
    return model


def compute_sample_weights(y_train):
    """
    Calcule les poids des échantillons pour gérer le déséquilibre des classes.
    
    Args:
        y_train (array): Labels d'entraînement
        
    Returns:
        array: Poids des échantillons
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([class_weights[y] for y in y_train])
    return sample_weights
