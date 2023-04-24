import numpy as np
import scipy
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


data = fetch_movielens(min_rating = 4.0)
print(repr(data['train']))
print(repr(data['test']))


model = LightFM(loss = 'warp')
model.fit(data['train'], epochs = 30, num_threads = 2)


def recommend(model, data , uids):
    n_users,n_items = data['train'].shape
    
    for uid in uids:
        
        known_positives = data['item_labels'][data['train'].tocsr()[uid].indices]
        scores = model.predict(uid , np.arange(n_items))
        top_items = data["item_labels"][np.argsort(scores)]
        print(f'User {uid} Movies')
        print('='*20)
        print ('    Known Positives')
        for x in known_positives:
            print(x)
            
        print('\n===========================')
        print('    Recommended')
        for x in top_items[:3]:
            print(x)
            
            
            

recommend (model, data , [4])
            
        
