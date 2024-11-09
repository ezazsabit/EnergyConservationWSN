from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as utils
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, maxabs_scale, MaxAbsScaler
import time
import random



def balance_data(df):
    # *************************
    ##Create a balanced Data
    # *************************
    for i in range(1, 8):
        print(i)
        #         print(df.index)
        cond = df.Cover_Type == i
        subset = df[cond].dropna()
        #         print(subset)
        if i == 4:
            subset = subset.sample(n=2747)
        else:
            subset = subset.sample(n=2747)
        if i == 1:
            balanced_data = subset
        else:
            balanced_data = balanced_data._append(subset)
            # frames = [balanced_data, subset]
 
            # balanced_data = pd.concat(frames)
    return balanced_data

def getData(filename, balance=False):
    chunksize = 1200000
    flag =1
    for data in pd.read_csv(filename, sep=",", chunksize=chunksize):
        while flag<2 and chunksize < 1200000:
            print(data)
        flag+=1
#     print(data.head())
    print(list(data.columns.values))
    print(type(data))
    if balance:
        data = balance_data(data)
    return data

def normalize_data(rawData):
    # training
    norm_tcolumns = rawData[rawData.columns[:10]]  # only the first ten columns need normalization, the rest is binary
    #     scaler = MinMaxScaler(copy=True, feature_range=(0, 1)).fit(norm_tcolumns.values)
    scaler = MaxAbsScaler(copy=True).fit(norm_tcolumns.values)
    scaledf = scaler.transform(norm_tcolumns.values)
    training_examples = pd.DataFrame(scaledf, index=norm_tcolumns.index,
                                     columns=norm_tcolumns.columns)  # scaledf is converted from array to dataframe
    rawData.update(training_examples)

    return rawData

filename = 'Dataset/covtype.csv'
# dataset = getData(filename, balance=True)
# print(type(dataset))

dataset=getData(filename, balance=False)#.values.tolist()
# random.shuffle(dataset)
# dataset=pd.DataFrame(dataset) #from dataset_list to dataset_dataframe after suffling

# print(dataset)

# print(dataset_shuffled)
nNAN_dataset=dataset.iloc[0:581012,:]

normalized_dataset=normalize_data(nNAN_dataset)
print(normalized_dataset)

result = normalized_dataset.columns[normalized_dataset.isna().any()].tolist()
print(result)
normalized_dataset.to_csv('Cover_type_whole_genuine_dataset_normalization.csv', index=False)