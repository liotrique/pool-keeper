import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# Configuration de la saison (ex: 20252026 pour la saison la plus récente disponible, ou 20232024, etc.)
SEASON = "20252026"
CSV_FILENAME = "meilleurs_joueurs_pool.csv"
GRAPH_FILENAME = "top_joueurs_pool_nhl.png"

def main():
    print(f"🚀 Récupération des statistiques LNH pour la saison {SEASON}...")

    # 1. Fetch des patineurs (Attaquants et Défenseurs)
    skaters_url = f"https://api.nhle.com/stats/rest/en/skater/summary?cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2&limit=-1"
    try:
        skaters_res = requests.get(skaters_url)
        skaters_res.raise_for_status()
        skaters_data = skaters_res.json().get('data', [])
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des patineurs: {e}")
        return

    # 2. Fetch des gardiens
    goalies_url = f"https://api.nhle.com/stats/rest/en/goalie/summary?cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2&limit=-1"
    try:
        goalies_res = requests.get(goalies_url)
        goalies_res.raise_for_status()
        goalies_data = goalies_res.json().get('data', [])
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des gardiens: {e}")
        return

    players = []

    # --- Traitement des Patineurs ---
    print(f"🏒 Traitement de {len(skaters_data)} patineurs...")
    for skater in skaters_data:
        pos = skater.get('positionCode', '')
        goals = skater.get('goals', 0)
        assists = skater.get('assists', 0)
        name = skater.get('skaterFullName', '')
        gp = skater.get('gamesPlayed', 0)
        team = skater.get('teamAbbrevs', '')
        
        if gp == 0:
            continue
            
        if pos in ['C', 'L', 'R']:
            position = 'Attaquant'
            # Règle: But = 1 pt, Passe = 1 pt
            points_pool = (goals * 1) + (assists * 1)
        elif pos == 'D':
            position = 'Défenseur'
            # Règle: But = 2 pts, Passe = 1 pt
            points_pool = (goals * 2) + (assists * 1)
        else:
            continue
            
        players.append({
            'Nom': name,
            'Équipe': team,
            'Position': position,
            'Matchs Joués': gp,
            'Buts (Patin.)': goals,
            'Passes (Patin.)': assists,
            'Victoires (Gard.)': 0,
            'Arrêts (Gard.)': 0,
            'Buts Accordés (Gard.)': 0,
            'Points Pool': float(points_pool)
        })

    # --- Traitement des Gardiens ---
    print(f"🥅 Traitement de {len(goalies_data)} gardiens...")
    print("⏳ Cela peut prendre quelques dizaines de secondes pour interroger l'historique match par match de chaque gardien (règle du plancher)...")
    
    for goalie in goalies_data:
        player_id = goalie.get('playerId')
        name = goalie.get('goalieFullName', '')
        gp = goalie.get('gamesPlayed', 0)
        team = goalie.get('teamAbbrevs', '')
        
        if gp == 0:
            continue
            
        # Appel API pour le Game Log du gardien afin de calculer le plancher par match
        log_url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/{SEASON}/2"
        try:
            log_res = requests.get(log_url)
            log_res.raise_for_status()
            game_logs = log_res.json().get('gameLog', [])
        except Exception:
            # Si le game log échoue, on ignore ou on approxime, on choisit d'ignorer pour la précision
            continue
        
        total_points = 0.0
        total_wins = 0
        total_saves = 0
        total_ga = 0
        
        for game in game_logs:
            decision = game.get('decision', '')
            shots_against = game.get('shotsAgainst', 0)
            goals_against = game.get('goalsAgainst', 0)
            
            saves = max(0, shots_against - goals_against)
            
            total_saves += saves
            total_ga += goals_against
            
            # Règle: Victoire = 1, Arrêt = 0.1, But accordé = -1
            win_points = 1 if decision == 'W' else 0
            save_points = saves * 0.1
            ga_points = goals_against * -1
            
            game_points = win_points + save_points + ga_points
            
            # Règle du plancher (Floor): score ramené à 0 si négatif
            if game_points < 0:
                game_points = 0
                
            total_points += game_points
            if decision == 'W':
                total_wins += 1
                
        players.append({
            'Nom': name,
            'Équipe': team,
            'Position': 'Gardien',
            'Matchs Joués': len(game_logs),
            'Buts (Patin.)': 0,
            'Passes (Patin.)': 0,
            'Victoires (Gard.)': total_wins,
            'Arrêts (Gard.)': total_saves,
            'Buts Accordés (Gard.)': total_ga,
            'Points Pool': round(total_points, 1)
        })
        
        time.sleep(0.05) # Pause pour éviter le rate-limiting de l'API LNH

    # --- Création du CSV ---
    df = pd.DataFrame(players)
    
    # Tri du meilleur pointeur au pire
    df = df.sort_values(by='Points Pool', ascending=False)
    
    df.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
    print(f"✅ Liste complète générée et triée dans : {CSV_FILENAME}")

    # --- Génération des graphiques ---
    print(f"📊 Génération des graphiques en cours...")
    plt.style.use('ggplot')
    
    attaquants = df[df['Position'] == 'Attaquant'].head(15)
    defenseurs = df[df['Position'] == 'Défenseur'].head(15)
    gardiens = df[df['Position'] == 'Gardien'].head(15)

    # Création d'une figure avec 3 sous-graphiques
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # 1. Attaquants
    axes[0].barh(attaquants['Nom'], attaquants['Points Pool'], color='skyblue', edgecolor='black')
    axes[0].set_title(f"Top 15 - Attaquants (Points Pool) - Saison {SEASON}")
    axes[0].set_xlabel("Points Pool")
    axes[0].invert_yaxis()  # Pour avoir le 1er en haut
    for i, v in enumerate(attaquants['Points Pool']):
        axes[0].text(v + 0.5, i, str(v), va='center', fontweight='bold')

    # 2. Défenseurs
    axes[1].barh(defenseurs['Nom'], defenseurs['Points Pool'], color='lightgreen', edgecolor='black')
    axes[1].set_title(f"Top 15 - Défenseurs (Points Pool) - Saison {SEASON}")
    axes[1].set_xlabel("Points Pool")
    axes[1].invert_yaxis()
    for i, v in enumerate(defenseurs['Points Pool']):
        axes[1].text(v + 0.5, i, str(v), va='center', fontweight='bold')

    # 3. Gardiens
    axes[2].barh(gardiens['Nom'], gardiens['Points Pool'], color='salmon', edgecolor='black')
    axes[2].set_title(f"Top 15 - Gardiens (Points Pool) - Saison {SEASON}")
    axes[2].set_xlabel("Points Pool")
    axes[2].invert_yaxis()
    for i, v in enumerate(gardiens['Points Pool']):
        axes[2].text(v + 0.5, i, str(v), va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(GRAPH_FILENAME, dpi=300)
    print(f"✅ Graphiques sauvegardés dans : {GRAPH_FILENAME}")
    print("🎉 Terminé !")

if __name__ == "__main__":
    main()
