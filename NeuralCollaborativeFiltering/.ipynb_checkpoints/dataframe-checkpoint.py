import pandas as pd
import numpy as np
import random
from copy import deepcopy

ml1m_dir = 'ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))

ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')

ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

# convert to binary data
ratings = deepcopy(ml1m_rating)
ratings['rating'][ratings['rating'] > 0] = 1.0