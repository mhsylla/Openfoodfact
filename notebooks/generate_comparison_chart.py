import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Données
data = {
    'Modèle': ['XGBoost', 'Random Forest'],
    'Accuracy Train': [0.847127, 0.990679],
    'Accuracy Test': [0.837557, 0.869270],
    'F1-Score Train': [0.832055, 0.989137],
    'F1-Score Test': [0.822693, 0.854442],
    'Temps (s)': [24.274573, 140.236301]
}

df = pd.DataFrame(data)

# Créer la figure avec 2 graphiques
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Performance (Accuracy et F1-Score Test)
metrics = ['Accuracy Test', 'F1-Score Test']
xgb_scores = [df.loc[0, 'Accuracy Test'], df.loc[0, 'F1-Score Test']]
rf_scores = [df.loc[1, 'Accuracy Test'], df.loc[1, 'F1-Score Test']]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[0].bar(x - width/2, xgb_scores, width, label='XGBoost', 
                     color='steelblue', edgecolor='black', linewidth=1.2)
bars2 = axes[0].bar(x + width/2, rf_scores, width, label='Random Forest', 
                     color='forestgreen', edgecolor='black', linewidth=1.2)

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_ylabel('Score', fontsize=13, fontweight='bold')
axes[0].set_title('Performance des Modèles - Test Set', fontsize=15, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics, fontsize=12)
axes[0].legend(fontsize=11, loc='lower right')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0].set_ylim([0, 1.0])

# Graphique 2 : Temps d'entraînement
temps = [df.loc[0, 'Temps (s)'], df.loc[1, 'Temps (s)']]
colors = ['steelblue', 'forestgreen']

bars = axes[1].bar(['XGBoost', 'Random Forest'], temps, 
                    color=colors, edgecolor='black', linewidth=1.2)

# Ajouter les valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].set_ylabel('Temps (secondes)', fontsize=13, fontweight='bold')
axes[1].set_title('Temps d\'Entraînement', fontsize=15, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1].set_xticklabels(['XGBoost', 'Random Forest'], fontsize=12)

plt.tight_layout()
plt.savefig('comparison_models_chart.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé : comparison_models_chart.png")
plt.show()

# Créer aussi un tableau visuel
fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('tight')
ax.axis('off')

# Préparer les données du tableau
table_data = [
    ['Modèle', 'Accuracy\nTrain', 'Accuracy\nTest', 'F1-Score\nTrain', 'F1-Score\nTest', 'Temps (s)'],
    ['XGBoost', '84.71%', '83.76%', '83.21%', '82.27%', '24.3'],
    ['Random Forest', '99.07%', '86.93%', '98.91%', '85.44%', '140.2']
]

# Couleurs pour les cellules
cell_colors = [
    ['#4472C4'] * 6,  # Header
    ['#D9E1F2'] * 6,  # XGBoost
    ['#E2EFDA'] * 6   # Random Forest
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                cellColours=cell_colors)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style pour l'en-tête
for i in range(6):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white')
    cell.set_facecolor('#4472C4')

# Style pour les lignes de données
for i in range(1, 3):
    for j in range(6):
        cell = table[(i, j)]
        cell.set_text_props(weight='bold' if j == 0 else 'normal')
        
# Mettre en évidence les meilleures valeurs
# Accuracy Test : Random Forest
table[(2, 2)].set_facecolor('#92D050')
table[(2, 2)].set_text_props(weight='bold')

# F1-Score Test : Random Forest
table[(2, 4)].set_facecolor('#92D050')
table[(2, 4)].set_text_props(weight='bold')

# Temps : XGBoost
table[(1, 5)].set_facecolor('#92D050')
table[(1, 5)].set_text_props(weight='bold')

plt.title('Comparaison des Modèles - Métriques Détaillées', 
          fontsize=16, fontweight='bold', pad=20)
plt.savefig('comparison_models_table.png', dpi=300, bbox_inches='tight')
print("✅ Tableau sauvegardé : comparison_models_table.png")
plt.show()

print("\n" + "="*70)
print("RÉSUMÉ DE LA COMPARAISON")
print("="*70)
print(f"\n🏆 Meilleur modèle (Accuracy Test) : Random Forest (86.93%)")
print(f"🏆 Meilleur modèle (F1-Score Test) : Random Forest (85.44%)")
print(f"⚡ Modèle le plus rapide : XGBoost (24.3s vs 140.2s)")
print(f"\n📊 Différence de performance :")
print(f"   - Accuracy Test : +3.17% pour Random Forest")
print(f"   - F1-Score Test : +3.17% pour Random Forest")
print(f"\n⏱️  Différence de temps : Random Forest est 5.8x plus lent")
print("\n✅ CONCLUSION : Random Forest offre les meilleures performances")
print("   malgré un temps d'entraînement plus long.")
print("="*70)
