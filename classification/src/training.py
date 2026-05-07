import time
from sklearn.metrics import accuracy_score, f1_score
from .models import compute_sample_weights


def train_model(model, X_train, y_train, X_test, y_test, use_sample_weights=False):
    """
    Entraîne un modèle et retourne les métriques.
    
    Args:
        model: Modèle à entraîner
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_test: Features de test
        y_test: Labels de test
        use_sample_weights (bool): Utiliser les poids d'échantillons
        
    Returns:
        tuple: (model, metrics)
    """
    start_time = time.time()
    
    if use_sample_weights:
        sample_weights = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='macro'),
        'test_f1': f1_score(y_test, y_test_pred, average='macro'),
        'training_time': training_time
    }
    
    print(f"Accuracy Train: {metrics['train_accuracy']:.4f}")
    print(f"Accuracy Test:  {metrics['test_accuracy']:.4f}")
    print(f"F1-Score Train: {metrics['train_f1']:.4f}")
    print(f"F1-Score Test:  {metrics['test_f1']:.4f}")
    print(f"Temps:          {metrics['training_time']:.2f}s")
    
    return model, metrics
