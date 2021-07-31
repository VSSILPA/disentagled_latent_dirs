import json
import os
import numpy as np

## Load JSON Code

attributes = ['pose', 'eyeglasses', 'male', 'smiling', 'young']
topk = 1

with open(os.path.join('../results/Closed-Form-Analysis-results', 'Attribute_variation_dictionary.json'), 'r') as f:
  data = json.load(f)

direction_scores = np.array(list(data.values()))
directions = {}
for attribute_idx, attribute_name in enumerate(attributes):
    best_direction = np.argsort(direction_scores[:,attribute_idx])[::-1][:topk]
    directions[attribute_name] = best_direction

attribute_directions = np.array(list(directions.values()))
rescore_metric = direction_scores[attribute_directions,:]

print(data)