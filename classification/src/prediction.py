import joblib
import pandas as pd
import numpy as np


def save_model(model, preprocessor, label_encoder, file_path):
    """
    Sauvegarde le modèle et les transformateurs.
    
    Args:
        model: Modèle entraîné
        preprocessor: Preprocesseur
        label_encoder: Encodeur de labels
        file_path (str): Chemin de sauvegarde
    """
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'label_encoder': label_encoder
    }
    joblib.dump(model_data, file_path)
    print(f"Modèle sauvegardé: {file_path}")


def load_model(file_path):
    """
    Charge un modèle sauvegardé.
    
    Args:
        file_path (str): Chemin du modèle
        
    Returns:
        dict: Dictionnaire contenant model, preprocessor, label_encoder
    """
    model_data = joblib.load(file_path)
    print(f"Modèle chargé: {file_path}")
    return model_data


def predict_nutriscore(model_data, X):
    """
    Prédit le Nutri-Score pour de nouvelles données.
    
    Args:
        model_data (dict): Données du modèle chargé
        X (pd.DataFrame): Features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    label_encoder = model_data['label_encoder']
    
    X_scaled = preprocessor.transform(X)
    
    y_pred_encoded = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    y_proba = model.predict_proba(X_scaled)
    
    return y_pred, y_proba


def predict_with_confidence(model_data, X):
    """
    Prédit avec niveau de confiance.
    
    Args:
        model_data (dict): Données du modèle
        X (pd.DataFrame): Features
        
    Returns:
        pd.DataFrame: Prédictions avec confiance
    """
    predictions, probabilities = predict_nutriscore(model_data, X)
    
    confidence = np.max(probabilities, axis=1)
    
    results = pd.DataFrame({
        'nutriscore_predicted': predictions,
        'confidence': confidence
    })
    
    label_encoder = model_data['label_encoder']
    for i, grade in enumerate(label_encoder.classes_):
        results[f'proba_{grade}'] = probabilities[:, i]
    
    return results
