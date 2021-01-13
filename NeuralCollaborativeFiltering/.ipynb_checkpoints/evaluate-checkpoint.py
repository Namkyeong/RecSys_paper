import numpy as np
import torch


def cal_metrics(model, test_data):
    
    hit = 0
    ndcg = 0
    
    for i in range(6040):
        test_pred = torch.FloatTensor(model(test_data[0][i].cuda(), test_data[1][i].cuda()).cpu()).view(-1, 1)
        neg_pred = torch.FloatTensor(model(test_data[0][i].expand(99).cuda(), test_data[2][i].cuda()).cpu())
        concat = torch.cat([test_pred, neg_pred]).view(-1)
            
        _, indices = torch.topk(concat, 10)
        indices = indices.numpy().tolist()
        if 0 in indices:
            hit += 1
            index = indices.index(0)
            ndcg += np.reciprocal(np.log2(index+2))
            
    hit_ratio = hit / 6040
    ndcg = ndcg / 6040
    
    return hit_ratio, ndcg