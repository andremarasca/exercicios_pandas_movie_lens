import pandas as pd

data_header = "user_id|item_id|rating|timestamp"
data_header = data_header.split("|")
df_data = pd.read_csv("ml-100k/u.data", sep="\t", names=data_header)

#%% Cálculo da média, desvio padrão e variância para o dataset de avaliações completo;

estatisticas_data = df_data.describe()

print(estatisticas_data)

#%% Cálculo de média, desvio padrão e variância para cada usuário (armazenar esses valores em novas colunas do dataset);

group_user = df_data[["user_id", "rating"]].groupby(["user_id"])

estatisticas_user = group_user.agg(["mean", "std", "var"])

print(estatisticas_user)

#%% Encontrar indivíduos que avaliam filmes de forma mais uniforme, i.e., avaliações estão próximo ao valor da média do indivíduo;

mean_user = estatisticas_user[('rating',  'mean')]

mean = estatisticas_data["rating"]["mean"]
std = estatisticas_data["rating"]["std"]

limite_inferior = mean - std
limite_superior = mean + std

decisao = (limite_inferior < mean_user) & (mean_user < limite_superior)

usuarios_uniformes = estatisticas_user[:][decisao]

#%% Encontrar indivíduos cujas avaliações são mais diversas, i.e., valores de avaliação muito negativos e positivos;

# Ordenar pela variancia
estatistica_user_sorted = estatisticas_user.sort_values(by=[('rating',  'var')], ascending=False)

indice = estatistica_user_sorted.index
variancia = estatistica_user_sorted[('rating',  'var')]

df_variancia = pd.DataFrame({"user_id":list(indice), "var": list(variancia)})

print("Muita variancia\n", df_variancia.iloc[0:5,:])

#%% Encontrar indivíduos que avaliam filmes de forma excessivamente positiva, i.e., aqueles cuja média de avaliações está bem acima da média global;

decisao = mean_user > limite_superior

usuarios_acima = estatisticas_user[:][decisao]

#%% Encontrar indivíduos que avaliam filmes de forma excessivamente negativa

decisao = mean_user < limite_inferior

usuarios_abaixo = estatisticas_user[:][decisao]