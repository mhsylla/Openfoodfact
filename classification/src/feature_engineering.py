import pandas as pd
import numpy as np


def create_engineered_features(X):
    """
    Crée des features dérivées à partir des features originales.
    
    Args:
        X (pd.DataFrame): Features originales
        
    Returns:
        pd.DataFrame: Features avec features dérivées
    """
    X_eng = X.copy()
    
    X_eng['energy_from_fat'] = X_eng['fat_100g'] * 9
    X_eng['energy_from_carbs'] = X_eng['carbohydrates_100g'] * 4
    X_eng['energy_from_proteins'] = X_eng['proteins_100g'] * 4
    
    X_eng['fat_ratio'] = X_eng['fat_100g'] / (X_eng['energy_100g'] + 1)
    X_eng['carbs_ratio'] = X_eng['carbohydrates_100g'] / (X_eng['energy_100g'] + 1)
    X_eng['proteins_ratio'] = X_eng['proteins_100g'] / (X_eng['energy_100g'] + 1)
    
    X_eng['sugar_to_carbs'] = X_eng['sugars_100g'] / (X_eng['carbohydrates_100g'] + 1)
    
    return X_eng


def create_ratio_features(X):
    """
    Crée des ratios entre les différentes features.
    
    Args:
        X (pd.DataFrame): Features
        
    Returns:
        pd.DataFrame: Features avec ratios
    """
    X_ratio = X.copy()
    
    X_ratio['fat_to_protein'] = X['fat_100g'] / (X['proteins_100g'] + 1)
    X_ratio['sugar_to_fiber'] = X['sugars_100g'] / (X['fiber_100g'] + 1)
    
    return X_ratio
