import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)


def evaluate_model(model, X_test, y_test, label_encoder, save_path=None):
    """
    Évalue un modèle et génère des visualisations.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
        label_encoder: Encodeur de labels
        save_path (str): Chemin pour sauvegarder les visualisations
        
    Returns:
        dict: Métriques d'évaluation
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    accuracy_tolerance = np.mean(np.abs(y_test - y_pred) <= 1)
    
    print(f"\n{'='*60}")
    print("ÉVALUATION DU MODÈLE")
    print(f"{'='*60}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (macro):   {f1_macro:.4f}")
    print(f"F1-Score (weighted):{f1_weighted:.4f}")
    print(f"Accuracy ±1 grade:  {accuracy_tolerance:.4f} ({accuracy_tolerance*100:.2f}%)")
    
    target_names = label_encoder.classes_
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")
    
    if save_path:
        plot_confusion_matrix(y_test, y_pred, target_names, save_path)
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, X_test.columns, save_path)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy_tolerance': accuracy_tolerance
    }


def plot_confusion_matrix(y_true, y_pred, target_names, save_path):
    """
    Génère et sauvegarde la matrice de confusion.
    
    Args:
        y_true: Vraies labels
        y_pred: Labels prédites
        target_names: Noms des classes
        save_path (str): Chemin de sauvegarde
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(f"{save_path}_confusion_matrix.png", dpi=300)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {save_path}_confusion_matrix.png")


def plot_feature_importance(model, feature_names, save_path):
    """
    Génère et sauvegarde l'importance des features.
    
    Args:
        model: Modèle avec feature_importances_
        feature_names: Noms des features
        save_path (str): Chemin de sauvegarde
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 15 Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{save_path}_feature_importance.png", dpi=300)
    plt.close()
    print(f"Importance des features sauvegardée: {save_path}_feature_importance.png")
