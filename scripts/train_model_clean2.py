import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, get_features_and_target
from src.preprocessing import NutriscorePreprocessor
from src.feature_engineering import create_engineered_features
from src.models import create_xgboost_model, create_random_forest_model
from src.training import train_model
from src.evaluation import evaluate_model
from src.prediction import save_model


def main():
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE NUTRI-SCORE - DATASET CLEAN_2")
    print("="*60)
    
    data_path = Path(__file__).parent.parent / 'data' / 'openfoodfacts_clean_2.csv'
    
    print(f"\nChargement des données depuis: {data_path}")
    df = load_data(str(data_path))
    print(f"Dataset chargé: {df.shape[0]:,} produits, {df.shape[1]} colonnes")
    
    print("\n" + "-"*60)
    print("EXTRACTION DES FEATURES ET CIBLE")
    print("-"*60)
    X, y = get_features_and_target(df)
    print(f"Features: {X.shape[1]} colonnes")
    print(f"Distribution du Nutri-Score:")
    print(y.value_counts().sort_index())
    
    print("\n" + "-"*60)
    print("FEATURE ENGINEERING")
    print("-"*60)
    X_engineered = create_engineered_features(X)
    print(f"Features après engineering: {X_engineered.shape[1]} colonnes")
    
    print("\n" + "-"*60)
    print("PREPROCESSING")
    print("-"*60)
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X_engineered, y, test_size=0.3, random_state=42
    )
    print(f"Train set: {X_train.shape[0]:,} échantillons")
    print(f"Test set: {X_test.shape[0]:,} échantillons")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT XGBOOST")
    print("="*60)
    xgb_model = create_xgboost_model()
    xgb_model, xgb_metrics = train_model(
        xgb_model, X_train, y_train, X_test, y_test, use_sample_weights=True
    )
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT RANDOM FOREST")
    print("="*60)
    rf_model = create_random_forest_model()
    rf_model, rf_metrics = train_model(
        rf_model, X_train, y_train, X_test, y_test, use_sample_weights=False
    )
    
    print("\n" + "="*60)
    print("COMPARAISON DES MODÈLES")
    print("="*60)
    
    print(f"\n{'Métrique':<25} {'XGBoost':<15} {'Random Forest':<15}")
    print("-"*60)
    print(f"{'Accuracy Train':<25} {xgb_metrics['train_accuracy']:<15.4f} {rf_metrics['train_accuracy']:<15.4f}")
    print(f"{'Accuracy Test':<25} {xgb_metrics['test_accuracy']:<15.4f} {rf_metrics['test_accuracy']:<15.4f}")
    print(f"{'F1-Score Train':<25} {xgb_metrics['train_f1']:<15.4f} {rf_metrics['train_f1']:<15.4f}")
    print(f"{'F1-Score Test':<25} {xgb_metrics['test_f1']:<15.4f} {rf_metrics['test_f1']:<15.4f}")
    print(f"{'Temps (secondes)':<25} {xgb_metrics['training_time']:<15.2f} {rf_metrics['training_time']:<15.2f}")
    
    best_model = rf_model if rf_metrics['test_f1'] > xgb_metrics['test_f1'] else xgb_model
    best_name = 'Random Forest' if rf_metrics['test_f1'] > xgb_metrics['test_f1'] else 'XGBoost'
    best_metrics = rf_metrics if rf_metrics['test_f1'] > xgb_metrics['test_f1'] else xgb_metrics
    
    print(f"\n🏆 Meilleur modèle: {best_name}")
    print(f"   F1-Score: {best_metrics['test_f1']:.4f}")
    
    print("\n" + "="*60)
    print(f"ÉVALUATION DÉTAILLÉE - {best_name}")
    print("="*60)
    
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_prefix = 'clean2_' + best_name.lower().replace(' ', '_')
    results = evaluate_model(
        best_model, X_test, y_test,
        preprocessor.label_encoder,
        save_path=str(models_dir / model_prefix)
    )
    
    print("\n" + "="*60)
    print("SAUVEGARDE DU MODÈLE")
    print("="*60)
    model_path = models_dir / 'nutriscore_model_clean2.pkl'
    save_model(best_model, preprocessor, preprocessor.label_encoder, str(model_path))
    
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"\n📊 Dataset: openfoodfacts_clean_2.csv")
    print(f"📈 Produits: {df.shape[0]:,}")
    print(f"🔧 Features: {X_engineered.shape[1]}")
    print(f"🏆 Modèle: {best_name}")
    print(f"📈 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"📈 F1-Score (macro): {results['f1_macro']:.4f}")
    print(f"📈 Accuracy ±1 grade: {results['accuracy_tolerance']:.4f} ({results['accuracy_tolerance']*100:.2f}%)")
    print(f"💾 Modèle sauvegardé: {model_path}")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)


if __name__ == "__main__":
    main()
