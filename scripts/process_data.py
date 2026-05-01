import pandas as pd

def clean_df(df_raw, cols, target='nutriscore_score'):
    df = df_raw.copy()
    
    # 1. Conversion de la TARGET en INT
    # On convertit d'abord en float pour gérer les décimales éventuelles du CSV,
    # puis on supprime les NaNs, et enfin on convertit en int.
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)

    # Filtre de réalisme pour le score cible (-15 à +40)[cite: 2]
    df = df[(df[target] >= -15) & (df[target] <= 40)]
    
    # 2. Conversion des NUTRI_COLS en FLOAT
    
    for col in cols:
        if col in df.columns:
            # Conversion en float pour supporter les valeurs décimales (ex: 0.5g de sel)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Suppression des lignes où le nutriment est invalide ou hors limites (0-100g)[cite: 2]
            df = df.dropna(subset=[col])
            if col != 'energy_100g':
                df = df[(df[col] >= 0) & (df[col] <= 100)]
    
    # 3. Filtre spécifique sur l'énergie (max 4000 kJ)[cite: 2]
    if 'energy_100g' in df.columns:
        df = df[(df['energy_100g'] >= 0) & (df['energy_100g'] < 4000)]
        
    
    return df


# total_rows = 0
# for chunk in reader:
#     total_rows += len(chunk)

# print(f"Taille totale du dataset : {total_rows} lignes")
# # Taille totale du dataset : 4465020 lignes