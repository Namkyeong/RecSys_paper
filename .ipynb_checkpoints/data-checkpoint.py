import numpy as np
import pandas as pd

# pass in column names for each CSV
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols)
train_df = pd.read_csv('ml-100k/u1.base', sep='\t', names=r_cols)
test_df = pd.read_csv('ml-100k/u1.test', sep='\t', names=r_cols)

n_users = df.user_id.unique().shape[0]
n_movies = df.movie_id.unique().shape[0]

ml_100k = np.zeros((n_users, n_movies))
train = np.zeros((n_users, n_movies))
test = np.zeros((n_users, n_movies))

for row in df.itertuples():
    ml_100k[row[1]-1, row[2]-1] = row[3]

for row in train_df.itertuples():
    train[row[1]-1, row[2]-1] = row[3]
    
for row in test_df.itertuples():
    test[row[1]-1, row[2]-1] = row[3]