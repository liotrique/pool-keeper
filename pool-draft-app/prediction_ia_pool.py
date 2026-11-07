import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Configuration
SEASONS = ["20212022", "20222023", "20232024", "20242025", "20252026"]
CURRENT_YEAR = 2026
CSV_FILENAME = "predictions_keeper_20262027.csv"
GRAPH_FILENAME = "ia_feature_importances.png"

def calculate_pool_points(goals, assists, position):
    if position in ['C', 'L', 'R']:
        return goals * 1.0 + assists * 1.0
    elif position == 'D':
        return goals * 2.0 + assists * 1.0
    return 0.0

def fetch_season_data(season):
    print(f"📡 Fetching data for {season}...")
    url = f"https://api.nhle.com/stats/rest/en/skater/summary?cayenneExp=seasonId={season}%20and%20gameTypeId=2&limit=-1"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json().get('data', [])
    
    df = pd.DataFrame(data)
    cols_to_keep = ['playerId', 'skaterFullName', 'positionCode', 'gamesPlayed', 'goals', 'assists', 'timeOnIcePerGame', 'shots', 'shootingPct', 'ppPoints']
    for c in cols_to_keep:
        if c not in df.columns:
            df[c] = 0.0
            
    df = df[cols_to_keep].copy()
    df['Points_Pool'] = df.apply(lambda x: calculate_pool_points(x['goals'], x['assists'], x['positionCode']), axis=1)
    
    year_suffix = season[:4] # ex: 2021, 2022, 2023, 2024, 2025
    df = df.rename(columns={
        'gamesPlayed': f'GP_{year_suffix}',
        'Points_Pool': f'Pts_{year_suffix}',
        'timeOnIcePerGame': f'TOI_{year_suffix}',
        'shots': f'Shots_{year_suffix}',
        'shootingPct': f'ShPct_{year_suffix}',
        'ppPoints': f'PPP_{year_suffix}'
    })
    return df

def fetch_bios(season="20252026"):
    print("📡 Fetching bios for Age calculation...")
    url = f"https://api.nhle.com/stats/rest/en/skater/bios?cayenneExp=seasonId={season}%20and%20gameTypeId=2&limit=-1"
    res = requests.get(url)
    data = res.json().get('data', [])
    df = pd.DataFrame(data)[['playerId', 'birthDate']]
    
    ref_date = pd.to_datetime(f"{CURRENT_YEAR}-09-01")
    df['birthDate'] = pd.to_datetime(df['birthDate'], errors='coerce')
    df['Age'] = (ref_date - df['birthDate']).dt.days / 365.25
    return df[['playerId', 'Age']]

def main():
    print("🤖 Démarrage de l'algorithme d'IA prédictif pour le pool...")
    
    # 1. Fetch des données
    dfs = {season[:4]: fetch_season_data(season) for season in SEASONS}
    df_bios = fetch_bios()
    
    # 2. Fusion (Merge) des données par joueur
    all_players = pd.concat([df[['playerId', 'skaterFullName', 'positionCode']] for df in dfs.values()]).drop_duplicates(subset=['playerId'])
    
    df_master = all_players.merge(df_bios, on='playerId', how='left')
    df_master['Age'] = df_master['Age'].fillna(25.0)
    df_master['Is_Def'] = (df_master['positionCode'] == 'D').astype(int)
    
    # Merge des stats pour chaque année
    for year, df in dfs.items():
        df_master = df_master.merge(df.drop(columns=['skaterFullName', 'positionCode', 'goals', 'assists']), on='playerId', how='left')
    
    df_master.fillna(0, inplace=True)
    
    # 3. Création du Dataset d'entraînement
    print("🧠 Structuration du dataset pour l'apprentissage machine (5 dernières années)...")
    
    features_cols = [
        'Age', 'Is_Def', 
        'GP_t1', 'Pts_t1', 'TOI_t1', 'Shots_t1', 'ShPct_t1', 'PPP_t1',
        'GP_t2', 'Pts_t2',
        'GP_t3', 'Pts_t3',
        'GP_t4', 'Pts_t4'
    ]
    
    # Entraînement: Target = 2025, t1=24, t2=23, t3=22, t4=21
    train_df = df_master[df_master['GP_2025'] > 10].copy()
    train_X = train_df[['Age', 'Is_Def']].copy()
    train_X['Age'] = train_X['Age'] - 1 
    
    train_X['GP_t1'] = train_df['GP_2024']
    train_X['Pts_t1'] = train_df['Pts_2024']
    train_X['TOI_t1'] = train_df['TOI_2024']
    train_X['Shots_t1'] = train_df['Shots_2024']
    train_X['ShPct_t1'] = train_df['ShPct_2024']
    train_X['PPP_t1'] = train_df['PPP_2024']
    
    train_X['GP_t2'] = train_df['GP_2023']
    train_X['Pts_t2'] = train_df['Pts_2023']
    
    train_X['GP_t3'] = train_df['GP_2022']
    train_X['Pts_t3'] = train_df['Pts_2022']
    
    train_X['GP_t4'] = train_df['GP_2021']
    train_X['Pts_t4'] = train_df['Pts_2021']
    
    train_y = train_df['Pts_2025']
    
    # 4. Évaluation
    print("🔬 Tests d'algorithmes (Random Forest vs Gradient Boosting)...")
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    
    print(f"Erreur moyenne (MAE) - Random Forest: {rf_mae:.2f} points")
    print(f"Erreur moyenne (MAE) - Gradient Boosting: {gb_mae:.2f} points")
    
    best_model = gb if gb_mae < rf_mae else rf
    best_model_name = "Gradient Boosting" if gb_mae < rf_mae else "Random Forest"
    print(f"✅ Modèle optimal sélectionné: {best_model_name}")
    
    # 5. Graphique d'importance
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': features_cols, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='purple')
    plt.title("Importance des attributs (Fields) dans la prédiction des points")
    plt.xlabel("Impact sur le modèle")
    plt.tight_layout()
    plt.savefig(GRAPH_FILENAME)
    print(f"📊 Graphique d'importance généré: {GRAPH_FILENAME}")
    
    # 6. Prédiction pour 2026-2027
    best_model.fit(train_X, train_y)
    
    # Target = 2026, t1=25, t2=24, t3=23, t4=22
    pred_df = df_master.copy()
    pred_X = pred_df[['Age', 'Is_Def']].copy()
    
    pred_X['GP_t1'] = pred_df['GP_2025']
    pred_X['Pts_t1'] = pred_df['Pts_2025']
    pred_X['TOI_t1'] = pred_df['TOI_2025']
    pred_X['Shots_t1'] = pred_df['Shots_2025']
    pred_X['ShPct_t1'] = pred_df['ShPct_2025']
    pred_X['PPP_t1'] = pred_df['PPP_2025']
    
    pred_X['GP_t2'] = pred_df['GP_2024']
    pred_X['Pts_t2'] = pred_df['Pts_2024']
    
    pred_X['GP_t3'] = pred_df['GP_2023']
    pred_X['Pts_t3'] = pred_df['Pts_2023']
    
    pred_X['GP_t4'] = pred_df['GP_2022']
    pred_X['Pts_t4'] = pred_df['Pts_2022']
    
    predictions_raw = best_model.predict(pred_X)
    pred_df['Points_Predits_26_27'] = np.round(predictions_raw, 1)
    
    # 7. Bonus de longévité (Keeper)
    def calcul_facteur_longevite(age):
        if age < 24:
            return 1.10
        elif age < 28:
            return 1.05
        elif age <= 31:
            return 1.00
        else:
            return max(0.85, 1.0 - ((age - 31) * 0.02))
            
    pred_df['Facteur_Longevite'] = pred_df['Age'].apply(calcul_facteur_longevite)
    pred_df['Score_Keeper_Optimal'] = np.round(pred_df['Points_Predits_26_27'] * pred_df['Facteur_Longevite'], 1)
    
    final_df = pred_df[['skaterFullName', 'positionCode', 'Age', 'Pts_2024', 'Pts_2025', 'Points_Predits_26_27', 'Score_Keeper_Optimal']]
    final_df.columns = ['Nom', 'Position', 'Âge', 'Points Pool 2024-25', 'Points Pool 2025-26', 'Prédiction Brute 2026-27', 'Score Valeur Dynastie']
    
    final_df = final_df[final_df['Prédiction Brute 2026-27'] > 20]
    final_df = final_df.sort_values(by='Score Valeur Dynastie', ascending=False)
    
    final_df.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
    print(f"🏆 Fichier de repêchage optimal généré: {CSV_FILENAME}")

if __name__ == "__main__":
    main()
