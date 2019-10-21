import numpy as np
import os
import torch
import pandas as pd

#open train test splits. 
train_images = open('./lists/train.txt')
train_images = train_images.readlines()
len(train_images)

train_images[0]
species = train_images[0].split('/')[0]
id, name = int(species.split('.')[0]), species.split('.')[1]

#create train images array
train_set = list()
for i in range(len(train_images)):
    _train_set = train_images[i].split("\n")[0]
    species = train_images[0].split('/')[0]
    id, name = int(species.split('.')[0])-1, species.split('.')[1]
    _train_set = ['./images/' + _train_set, id, name]
    train_set.append(_train_set)
print(len(train_set))

train_set[0]