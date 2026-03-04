import sys
import os
import time

sys.path.insert(0, os.path.abspath('c:/Projects/LM-Lab'))
from api.services.inference import list_mlp_grid_configurations, mlp_grid_training_timeline, mlp_grid_predict

configs = list_mlp_grid_configurations()
print("Found configs:", len(configs))
one_hot_configs = [c for c in configs if c.get('embedding_dim') == 0]
print("One hot configs:", len(one_hot_configs))

for c in one_hot_configs:
    print(c)
    print("Timeline:", mlp_grid_training_timeline(c['embedding_dim'], c['hidden_size'], c['learning_rate']).keys())
    print("Predict:", mlp_grid_predict("the", c['embedding_dim'], c['hidden_size'], c['learning_rate']).keys())
