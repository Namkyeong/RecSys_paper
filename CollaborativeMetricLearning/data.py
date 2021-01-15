import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import random
random.seed(315)


# Custom Dataset
# https://wikidocs.net/57165
class RatingDataset(Dataset):
    """
    torch.utils.data.Dataset 상속
    """
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        
    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    

class SampleGenerator():
    
    def __init__(self, data):
        
        self.data = data
        self.user_pool = set(self.data["userId"].unique())
        self.item_pool = set(self.data["itemId"].unique())
        self.negatives = self._sample_negative(data)
        self.train, self.test = self._split(self.data)
    
    
    def _sample_negative(self, data):
        # we uniformly sample negative instances from unobserved interactions in each iteration
        # and control the sampling ratio w.r.t. the number of observed interations
        interact_status = data.groupby("userId")["itemId"].apply(set).reset_index().rename(columns = {"itemId" : "interacted_items"})
        interact_status["negative_items"] = interact_status["interacted_items"].apply(lambda x: self.item_pool - x)

        return interact_status[["userId", "negative_items"]]
    
    
    def instance_a_train_loader(self, num_negative, batch_size):
        """
        instance train loader for a training epoch
        """
        users, items, implicit = [], [], []
        train_loader = pd.merge(self.train, self.negatives, on="userId")
        train_loader["negatives"] = train_loader["negative_items"].apply(lambda x: random.sample(x, num_negative))
        for row in train_loader.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            implicit.append(float(row.rating))
            for i in range(num_negative):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                implicit.append(float(0))
        
        dataset = RatingDataset(user_tensor = torch.LongTensor(users),
                               item_tensor = torch.LongTensor(items),
                               target_tensor = torch.FloatTensor(implicit))
        
        return DataLoader(dataset, batch_size = batch_size, shuffle=True)
    
    @property
    def evaluate_data(self):
        """
        create test data
        """
        test_data = pd.merge(self.test, self.negatives, on= "userId")
        test_data["negative_samples"] = test_data["negative_items"].apply(lambda x: random.sample(x, 99))
        test_users, test_items, negative_items = [], [], []
        
        for row in test_data.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            negative_items.append(row.negative_samples)
        
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_items)]
        