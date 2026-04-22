import pandas as pd

INPUT_PATH = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
OUTPUT_PATH = '../data/openfoodfacts_regression_clean.csv'

# Colonnes pour les features et la cible numérique
nutriscore_cols = [
    'energy_100g', 'fat_100g', 'saturated-fat_100g', 
    'sugars_100g', 'fiber_100g', 'proteins_100g', 'carbohydrates_100g',
    'salt_100g'
]

# Changement : on vise le SCORE (numérique)
target = 'nutriscore_score' 
identity_cols = ['code', 'product_name', 'nutriscore_grade'] # On garde le grade juste pour info

cols = nutriscore_cols + [target] + identity_cols

def process_data_regression(file_path, cols, chunk_size=20000):
    reader = pd.read_csv(
        file_path, compression='gzip', sep='\t',
        on_bad_lines='skip', chunksize=chunk_size,
        low_memory=False, usecols=cols
    )

    clean_chunks = []
    print("Début du traitement des blocs...")

    for i, chunk in enumerate(reader):
        temp_chunk = chunk.copy()

        # 1. Nettoyage de la Cible (Target)
        # On supprime les lignes où le score numérique manque
        temp_chunk = temp_chunk.dropna(subset=[target])
        
        # Filtre de réalisme pour le score (Officiel : -15 à +40)
        temp_chunk = temp_chunk[(temp_chunk[target] >= -15) & (temp_chunk[target] <= 40)]

        # 2. Conversion numérique des nutriments
        for col in nutriscore_cols:
            temp_chunk[col] = pd.to_numeric(temp_chunk[col], errors='coerce')
        
        # Remplissage des nutriments manquants par 0 (Imputation)
        temp_chunk[nutriscore_cols] = temp_chunk[nutriscore_cols].fillna(0)

        # 3. Filtres Outliers (0-100g et Energie)
        for col in nutriscore_cols:
            if col != 'energy_100g':
                temp_chunk = temp_chunk[(temp_chunk[col] >= 0) & (temp_chunk[col] <= 100)]
        
        temp_chunk = temp_chunk[(temp_chunk['energy_100g'] >= 0) & (temp_chunk['energy_100g'] < 4000)]

        # 4. Gestion de l'identité
        temp_chunk['product_name'] = temp_chunk['product_name'].fillna('Unknown Product')
        temp_chunk = temp_chunk.drop_duplicates(subset=['code'])

        # 5. Stockage du bloc propre
        if not temp_chunk.empty:
            clean_chunks.append(temp_chunk)
        
        if i % 10 == 0:
            print(f"Bloc {i} traité...")

    # 6. Assemblage final (HORS de la boucle pour la performance)
    print("Fusion des blocs en cours...")
    df_final = pd.concat(clean_chunks, ignore_index=True)
    
    df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Traitement terminé ! {len(df_final)} lignes sauvegardées dans {OUTPUT_PATH}")

    return df_final

# Lancement
df = process_data_regression(INPUT_PATH, cols)