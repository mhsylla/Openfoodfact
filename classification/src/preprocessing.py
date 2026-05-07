import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class NutriscorePreprocessor:
    """
    Classe pour le preprocessing des données Nutri-Score.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def fit_transform(self, X, y, test_size=0.3, random_state=42):
        """
        Applique le preprocessing et split les données.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Cible
            test_size (float): Proportion du test set
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def transform(self, X):
        """
        Applique le scaling sur de nouvelles données.
        
        Args:
            X (pd.DataFrame): Features à transformer
            
        Returns:
            pd.DataFrame: Features transformées
        """
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled
