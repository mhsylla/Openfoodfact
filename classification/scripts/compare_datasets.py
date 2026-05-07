import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, get_features_and_target
from src.preprocessing import NutriscorePreprocessor
from src.feature_engineering import create_engineered_features
from src.models import create_random_forest_model
from src.training import train_model


def analyze_dataset(data_path, dataset_name):
    print(f"\n{'='*60}")
    print(f"ANALYSE - {dataset_name}")
    print(f"{'='*60}")
    
    df = load_data(str(data_path))
    print(f"\n📊 Taille: {df.shape[0]:,} produits, {df.shape[1]} colonnes")
    
    X, y = get_features_and_target(df)
    
    print(f"\n📈 Distribution du Nutri-Score:")
    dist = y.value_counts().sort_index()
    for grade, count in dist.items():
        pct = count / len(y) * 100
        print(f"   Grade {grade}: {count:,} ({pct:.2f}%)")
    
    print(f"\n📊 Statistiques des features:")
    print(X.describe().loc[['mean', 'std', 'min', 'max']].round(2))
    
    X_eng = create_engineered_features(X)
    preprocessor = NutriscorePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X_eng, y, test_size=0.3, random_state=42
    )
    
    print(f"\n🤖 Entraînement Random Forest...")
    rf_model = create_random_forest_model()
    rf_model, metrics = train_model(
        rf_model, X_train, y_train, X_test, y_test, use_sample_weights=False
    )
    
    print(f"\n✅ Résultats:")
    print(f"   Accuracy Train: {metrics['train_accuracy']:.4f}")
    print(f"   Accuracy Test:  {metrics['test_accuracy']:.4f}")
    print(f"   F1-Score Train: {metrics['train_f1']:.4f}")
    print(f"   F1-Score Test:  {metrics['test_f1']:.4f}")
    print(f"   Temps:          {metrics['training_time']:.2f}s")
    
    return {
        'name': dataset_name,
        'n_products': df.shape[0],
        'n_features': X.shape[1],
        'distribution': dist,
        'stats': X.describe(),
        'accuracy_train': metrics['train_accuracy'],
        'accuracy_test': metrics['test_accuracy'],
        'f1_train': metrics['train_f1'],
        'f1_test': metrics['test_f1'],
        'training_time': metrics['training_time'],
        'X': X,
        'y': y
    }


def compare_datasets(results1, results2):
    print(f"\n{'='*60}")
    print("COMPARAISON DES DATASETS")
    print(f"{'='*60}")
    
    comparison = pd.DataFrame({
        'Métrique': [
            'Nombre de produits',
            'Nombre de features',
            'Accuracy Train',
            'Accuracy Test',
            'F1-Score Train',
            'F1-Score Test',
            'Temps (s)'
        ],
        results1['name']: [
            f"{results1['n_products']:,}",
            results1['n_features'],
            f"{results1['accuracy_train']:.4f}",
            f"{results1['accuracy_test']:.4f}",
            f"{results1['f1_train']:.4f}",
            f"{results1['f1_test']:.4f}",
            f"{results1['training_time']:.2f}"
        ],
        results2['name']: [
            f"{results2['n_products']:,}",
            results2['n_features'],
            f"{results2['accuracy_train']:.4f}",
            f"{results2['accuracy_test']:.4f}",
            f"{results2['f1_train']:.4f}",
            f"{results2['f1_test']:.4f}",
            f"{results2['training_time']:.2f}"
        ]
    })
    
    print(f"\n{comparison.to_string(index=False)}")
    
    print(f"\n{'='*60}")
    print("DIFFÉRENCES")
    print(f"{'='*60}")
    
    diff_products = results2['n_products'] - results1['n_products']
    diff_acc = results2['accuracy_test'] - results1['accuracy_test']
    diff_f1 = results2['f1_test'] - results1['f1_test']
    
    print(f"\n📊 Produits: {diff_products:+,} ({diff_products/results1['n_products']*100:+.1f}%)")
    print(f"📈 Accuracy Test: {diff_acc:+.4f} ({diff_acc*100:+.2f}%)")
    print(f"📈 F1-Score Test: {diff_f1:+.4f} ({diff_f1*100:+.2f}%)")
    
    if results2['accuracy_test'] > results1['accuracy_test']:
        print(f"\n✅ {results2['name']} est MEILLEUR en accuracy (+{diff_acc*100:.2f}%)")
    elif results2['accuracy_test'] < results1['accuracy_test']:
        print(f"\n⚠️ {results1['name']} est MEILLEUR en accuracy ({diff_acc*100:.2f}%)")
    else:
        print(f"\n➡️ Performance IDENTIQUE")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    grades = ['a', 'b', 'c', 'd', 'e']
    dist1 = [results1['distribution'].get(g, 0) for g in grades]
    dist2 = [results2['distribution'].get(g, 0) for g in grades]
    
    x = np.arange(len(grades))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, dist1, width, label=results1['name'], color='steelblue')
    axes[0, 0].bar(x + width/2, dist2, width, label=results2['name'], color='forestgreen')
    axes[0, 0].set_xlabel('Grade Nutri-Score')
    axes[0, 0].set_ylabel('Nombre de produits')
    axes[0, 0].set_title('Distribution du Nutri-Score', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(grades)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    metrics = ['Accuracy Test', 'F1-Score Test']
    values1 = [results1['accuracy_test'], results1['f1_test']]
    values2 = [results2['accuracy_test'], results2['f1_test']]
    
    x = np.arange(len(metrics))
    axes[0, 1].bar(x - width/2, values1, width, label=results1['name'], color='steelblue')
    axes[0, 1].bar(x + width/2, values2, width, label=results2['name'], color='forestgreen')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Performance des Modèles', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    features = ['energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']
    means1 = [results1['X'][f].mean() for f in features]
    means2 = [results2['X'][f].mean() for f in features]
    
    x = np.arange(len(features))
    axes[1, 0].bar(x - width/2, means1, width, label=results1['name'], color='steelblue')
    axes[1, 0].bar(x + width/2, means2, width, label=results2['name'], color='forestgreen')
    axes[1, 0].set_ylabel('Valeur moyenne')
    axes[1, 0].set_title('Moyennes des Features Principales', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(features, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    axes[1, 1].bar([results1['name'], results2['name']], 
                   [results1['training_time'], results2['training_time']],
                   color=['steelblue', 'forestgreen'])
    axes[1, 1].set_ylabel('Temps (secondes)')
    axes[1, 1].set_title('Temps d\'Entraînement', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / 'models'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'comparison_datasets.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graphiques sauvegardés: {output_path}")
    
    plt.show()


def main():
    print("="*60)
    print("COMPARAISON DES DATASETS")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / 'data'
    
    path1 = data_dir / 'openfoodfacts_clean.csv'
    path2 = data_dir / 'openfoodfacts_clean_2.csv'
    
    if not path1.exists():
        print(f"❌ Fichier non trouvé: {path1}")
        return
    
    if not path2.exists():
        print(f"❌ Fichier non trouvé: {path2}")
        return
    
    results1 = analyze_dataset(path1, 'Clean Original')
    results2 = analyze_dataset(path2, 'Clean_2')
    
    compare_datasets(results1, results2)
    
    print("\n" + "="*60)
    print("COMPARAISON TERMINÉE")
    print("="*60)


if __name__ == "__main__":
    main()
