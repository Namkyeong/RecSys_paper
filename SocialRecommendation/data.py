import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ratings_data = pd.read_csv("ratings_data.txt", names=["userId", "itemId", "ratings"], sep=" ")
trust_data = pd.read_csv("trust_data.txt", names=["fromId", "toId", "trust"], sep = " ")

train_data, test_data = train_test_split(ratings_data, test_size = 0.2, random_state = 315)

n_users = ratings_data.userId.unique().shape[0]
n_items = ratings_data.itemId.unique().shape[0]

epinions = np.zeros((49289, n_items))
train = np.zeros((49289, n_items))
test = np.zeros((49289, n_items))

for row in ratings_data.itertuples():
    epinions[row[1]-1, row[2]-1] = row[3]

for row in train_data.itertuples():
    train[row[1]-1, row[2]-1] = row[3]
    
for row in test_data.itertuples():
    test[row[1]-1, row[2]-1] = row[3]