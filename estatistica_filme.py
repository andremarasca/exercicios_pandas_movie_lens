import pandas as pd

item_header = "movie_id|movie_title|release_date|video_release_date|IMDb_URL|unknown|Action|Adventure|Animation|Children's|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western"
item_header = item_header.split("|")
df_movie = pd.read_csv("ml-100k/u.item", sep="|", names=item_header, encoding="latin-1")

generos = df_movie.loc[:, "unknown":]

#%% Identificar qual gênero de filme possui o maior número de exemplos;

series_soma = generos.sum()

print(series_soma)
print("MAXIMO:", series_soma.idxmax())

#%% Verificar se existem dados faltando

print(df_movie.isnull().sum())

#%% Criar novo DataFrame que condense informações sobre o gênero do filme:

df_movie2 = df_movie["movie_id|movie_title|release_date|video_release_date|IMDb_URL".split("|")]
df_movie2.set_index("movie_id", inplace=True, drop=False)

data_header = "user_id|item_id|rating|timestamp"
data_header = data_header.split("|")
df_data = pd.read_csv("ml-100k/u.data", sep="\t", names=data_header)
group_movie = df_data[["item_id", "rating"]].groupby(["item_id"])


#%% Adicionar colunas que armazenem dados para o total de avaliações, a soma das avaliações, média, valor máximo (e mínimo), desvio padrão e variância;

estatisticas_movie = group_movie.agg(["count", "sum", "mean", "std", "var", "max", "min"])
estatisticas_movie.columns=estatisticas_movie.columns.droplevel()
juncao = df_movie2.join(estatisticas_movie)

#%% Mostrar filmes com maior (e menor) número de avaliações;

idxmax_count = juncao["count"].idxmax()
idxmin_count = juncao["count"].idxmin()

contagem_extremos = juncao.iloc[[idxmax_count, idxmin_count],:]

print(contagem_extremos)