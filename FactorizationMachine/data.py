import pandas as pd
import numpy as np

def create_dataset(df_train):
    
    num_users = df_train["user_id"].max()
    num_movies = df_train["movie_id"].max()
    
    # Build user one-hot dataset
    user = np.zeros((len(df_train), num_users))
    for i in range(len(df_train)):
        user[i, df_train["user_id"][i] - 1] = 1
        
    # Build movie one_hot dataset
    movie = np.zeros((len(df_train), num_movies))
    for i in range(len(df_train)):
        movie[i, df_train["movie_id"][i] - 1] = 1
        
    rating_mat = np.zeros((num_users, num_movies))
    for row in df_train.itertuples():
        rating_mat[row[1]-1, row[2]-1] = row[3]
    rating_mat = np.array(np.vectorize(lambda x: 0 if x==0 else 1)(rating_mat))
    
    
    # Build movie rated dataset
    movie_rated = np.zeros((len(df_train), num_movies))
    for i in range(len(df_train)):
        movie_rated[i] = rating_mat[df_train["user_id"][i]-1]

    movie_rated = movie_rated / movie_rated.sum(axis = 1)[:, np.newaxis]
    
    time = np.zeros((len(df_train), 1))
    for i in range(len(df_train)):
        time[i] = df_train["unix_timestamp"][i]
        
    train_dataset = np.concatenate([user, movie, movie_rated], axis = 1)
    
    train_y = np.zeros((len(df_train), 1))
    for i in range(len(df_train)):
        train_y[i] = df_train['rating'][i]
        
    return train_dataset, train_y