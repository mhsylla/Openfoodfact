import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid')

print("="*60)
print("COMPARAISON DES DATASETS")
print("="*60)

data_dir = Path(__file__).parent / 'data'
path1 = data_dir / 'openfoodfacts_clean.csv'
path2 = data_dir / 'openfoodfacts_clean_2.csv'

print("\n📂 Chargement des datasets...")

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

print(f"\n✅ Clean Original: {df1.shape[0]:,} produits, {df1.shape[1]} colonnes")
print(f"✅ Clean_2:        {df2.shape[0]:,} produits, {df2.shape[1]} colonnes")

print("\n" + "="*60)
print("COMPARAISON GÉNÉRALE")
print("="*60)

comparison_general = pd.DataFrame({
    'Métrique': [
        'Nombre de produits',
        'Nombre de colonnes',
        'Taille mémoire (MB)'
    ],
    'Clean Original': [
        f"{df1.shape[0]:,}",
        df1.shape[1],
        f"{df1.memory_usage(deep=True).sum() / 1024**2:.2f}"
    ],
    'Clean_2': [
        f"{df2.shape[0]:,}",
        df2.shape[1],
        f"{df2.memory_usage(deep=True).sum() / 1024**2:.2f}"
    ]
})

print(f"\n{comparison_general.to_string(index=False)}")

diff_products = df2.shape[0] - df1.shape[0]
diff_pct = diff_products / df1.shape[0] * 100

print(f"\n📊 Différence produits: {diff_products:+,} ({diff_pct:+.1f}%)")

print("\n" + "="*60)
print("DISTRIBUTION DU NUTRI-SCORE")
print("="*60)

dist1 = df1['nutriscore_grade'].value_counts().sort_index()
dist2 = df2['nutriscore_grade'].value_counts().sort_index()

print(f"\n{'Grade':<10} {'Clean Original':<20} {'Clean_2':<20} {'Différence':<15}")
print("-"*65)

for grade in ['a', 'b', 'c', 'd', 'e']:
    count1 = dist1.get(grade, 0)
    count2 = dist2.get(grade, 0)
    pct1 = count1 / len(df1) * 100
    pct2 = count2 / len(df2) * 100
    diff = count2 - count1
    
    print(f"{grade:<10} {count1:>8,} ({pct1:>5.2f}%)   {count2:>8,} ({pct2:>5.2f}%)   {diff:>+8,}")

print("\n" + "="*60)
print("STATISTIQUES DES FEATURES")
print("="*60)

features = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 
            'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g']

print(f"\n{'Feature':<25} {'Clean Original':<25} {'Clean_2':<25}")
print("-"*75)

for feature in features:
    if feature in df1.columns and feature in df2.columns:
        mean1 = df1[feature].mean()
        mean2 = df2[feature].mean()
        std1 = df1[feature].std()
        std2 = df2[feature].std()
        
        print(f"{feature:<25} {mean1:>8.2f} (±{std1:>6.2f})     {mean2:>8.2f} (±{std2:>6.2f})")

print("\n" + "="*60)
print("VALEURS MANQUANTES")
print("="*60)

print(f"\n{'Feature':<25} {'Clean Original':<20} {'Clean_2':<20}")
print("-"*65)

for feature in features:
    if feature in df1.columns and feature in df2.columns:
        na1 = df1[feature].isna().sum()
        na2 = df2[feature].isna().sum()
        pct1 = na1 / len(df1) * 100
        pct2 = na2 / len(df2) * 100
        
        print(f"{feature:<25} {na1:>8,} ({pct1:>5.2f}%)   {na2:>8,} ({pct2:>5.2f}%)")

print("\n" + "="*60)
print("VISUALISATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

grades = ['a', 'b', 'c', 'd', 'e']
counts1 = [dist1.get(g, 0) for g in grades]
counts2 = [dist2.get(g, 0) for g in grades]

x = np.arange(len(grades))
width = 0.35

axes[0, 0].bar(x - width/2, counts1, width, label='Clean Original', color='steelblue')
axes[0, 0].bar(x + width/2, counts2, width, label='Clean_2', color='forestgreen')
axes[0, 0].set_xlabel('Grade Nutri-Score')
axes[0, 0].set_ylabel('Nombre de produits')
axes[0, 0].set_title('Distribution du Nutri-Score', fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(grades)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

pcts1 = [dist1.get(g, 0) / len(df1) * 100 for g in grades]
pcts2 = [dist2.get(g, 0) / len(df2) * 100 for g in grades]

axes[0, 1].bar(x - width/2, pcts1, width, label='Clean Original', color='steelblue')
axes[0, 1].bar(x + width/2, pcts2, width, label='Clean_2', color='forestgreen')
axes[0, 1].set_xlabel('Grade Nutri-Score')
axes[0, 1].set_ylabel('Pourcentage (%)')
axes[0, 1].set_title('Distribution en Pourcentage', fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(grades)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

main_features = ['energy_100g', 'fat_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']
means1 = [df1[f].mean() for f in main_features]
means2 = [df2[f].mean() for f in main_features]

x = np.arange(len(main_features))
axes[1, 0].bar(x - width/2, means1, width, label='Clean Original', color='steelblue')
axes[1, 0].bar(x + width/2, means2, width, label='Clean_2', color='forestgreen')
axes[1, 0].set_ylabel('Valeur moyenne')
axes[1, 0].set_title('Moyennes des Features Principales', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(main_features, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

sizes = [df1.shape[0], df2.shape[0]]
labels = ['Clean Original\n' + f'{df1.shape[0]:,} produits', 
          'Clean_2\n' + f'{df2.shape[0]:,} produits']
colors = ['steelblue', 'forestgreen']

axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Répartition des Produits', fontweight='bold')

plt.tight_layout()

output_path = Path(__file__).parent / 'comparison_datasets.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n📊 Graphiques sauvegardés: {output_path}")

plt.show()

print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)

print(f"\n📊 Clean Original:")
print(f"   - {df1.shape[0]:,} produits")
print(f"   - Distribution: a={dist1.get('a', 0):,}, b={dist1.get('b', 0):,}, c={dist1.get('c', 0):,}, d={dist1.get('d', 0):,}, e={dist1.get('e', 0):,}")

print(f"\n📊 Clean_2:")
print(f"   - {df2.shape[0]:,} produits ({diff_pct:+.1f}%)")
print(f"   - Distribution: a={dist2.get('a', 0):,}, b={dist2.get('b', 0):,}, c={dist2.get('c', 0):,}, d={dist2.get('d', 0):,}, e={dist2.get('e', 0):,}")

if 'nutriscore_score' in df2.columns and 'nutriscore_score' not in df1.columns:
    print(f"\n✅ Clean_2 contient la colonne 'nutriscore_score' (absente dans Clean Original)")

print("\n" + "="*60)
print("COMPARAISON TERMINÉE")
print("="*60)
