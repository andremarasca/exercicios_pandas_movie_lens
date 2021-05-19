import pandas as pd

data_header = "user_id|item_id|rating|timestamp"
data_header = data_header.split("|")
df_data = pd.read_csv("ml-100k/u.data", sep="\t", names=data_header)

estatisticas_data = df_data.describe()

#%% 

rating = df_data["rating"]
mean = estatisticas_data["rating"]["mean"]
std = estatisticas_data["rating"]["std"]
mini = estatisticas_data["rating"]["min"]
maxi = estatisticas_data["rating"]["max"]

minimax_norm = (rating - mini) / (maxi - mini)
mean_norm = (rating - mean) / (maxi - mini)
z_score_norm = (rating - mean) / std

#%% 

df_data["minimax norm rating"] = minimax_norm
df_data["mean norm rating"] = mean_norm
df_data["z score rating"] = z_score_norm