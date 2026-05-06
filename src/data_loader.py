import pandas as pd
from pathlib import Path


def load_data(file_path):
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
    """
    df = pd.read_csv(file_path)
    return df


def get_features_and_target(df, target_col='nutriscore_grade'):
    """
    Sépare les features et la cible.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        target_col (str): Nom de la colonne cible
        
    Returns:
        tuple: (X, y) features et cible
    """
    feature_cols = [
        'energy_100g', 'fat_100g', 'saturated-fat_100g',
        'carbohydrates_100g', 'sugars_100g', 'fiber_100g',
        'proteins_100g', 'salt_100g'
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    return X, y


def save_data(df, file_path):
    """
    Sauvegarde un DataFrame dans un fichier CSV.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
        file_path (str): Chemin de destination
    """
    df.to_csv(file_path, index=False)
    print(f"Données sauvegardées dans {file_path}")
