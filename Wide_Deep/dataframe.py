import pandas as pd
import numpy as np

df_ratings = pd.read_csv("./ml-1m/ratings.dat", sep="::", names=['userId', 'movieId', 'rating', 'timestamp'])
df_movies = pd.read_table("./ml-1m/movies.dat", sep="::", names=["movieId","movie_name", "genre"], encoding = "latin-1")
df_users = pd.read_table("./ml-1m/users.dat", sep="::", names=["userId", "gender", "age", "occupation", "zipcode"])

df_ratings = df_ratings.drop('timestamp', axis=1)
df_movies['genre'] = df_movies.apply(lambda row : row['genre'].split("|")[0],axis=1)
df_movies['movie_year'] = df_movies.apply(lambda row : int(row['movie_name'].split("(")[-1][:-1]),axis=1)
df_movies.drop(['movie_name'],axis=1,inplace=True)
rating_movie = pd.merge(df_ratings,df_movies,how='left',on="movieId")

# df_users['gender'].replace({'F':0,'M':1},inplace=True)
# df_users['age'].replace({1:0,18:1, 25:2, 35:3, 45:4, 50:5, 56:6 },inplace=True)
df_users.drop(['zipcode'],axis=1,inplace=True)
final_df = pd.merge(rating_movie, df_users,how='left',on='userId')

final_df
