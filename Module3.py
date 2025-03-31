import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


#Read in the csv
stats = pd.read_csv("/Users/devinreed/Downloads/INST414/Module3/premier-player-23-24.csv")

print(stats.columns)

#Removes Goalkeepers and players under 250 minutes played
filtered = stats[(stats['Pos'] != 'GK') & (stats['Min'] >=250)].copy()


#Stats that will be used for comparing players
metrics = ['Gls', 'Ast', 'G+A_90', 'xG_90', 'xAG_90', 'npxG_90', 'PrgP', 'PrgC', 
           'Ast_90', 'Gls_90']

filtered = filtered.dropna(subset = metrics)

scaler = StandardScaler()
metrics_sc = scaler.fit_transform(filtered[metrics])


matrix = cosine_similarity(metrics_sc)


sim_df = pd.DataFrame(matrix, index=filtered['Player'], columns=filtered['Player'])

#Top 10 similar players
def top_similar_players(player_name, top_n = 10):
    if player_name not in sim_df.columns:
        return f"{player_name} not found"
    sim_players = sim_df[player_name].sort_values(ascending = False)[1:top_n+1]
    return sim_players

players = ['Phil Foden', 'Erling Haaland', 'Rodri', 'Virgil van Dijk']

for player in players:
    print(f"\nPlayers similar to {player}:\n")
    print(top_similar_players(player))




