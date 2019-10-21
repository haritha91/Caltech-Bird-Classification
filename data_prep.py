import numpy as np
import os
import torch
import pandas as pd

#open train test splits. 
train_set = open('./lists/train.txt')
a = train_set.readlines()
len(a)


train = list()
for i in range(len(a)):
    _t = a[i].split("\n")[0]
    _t = './images/' + _t
    train.append(_t)
print(len(train))

a[0]
b = a[0].split("\n")[0]
b
