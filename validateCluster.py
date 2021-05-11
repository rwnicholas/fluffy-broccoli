#!/usr/bin/python3
import pandas as pd
import numpy as np
import gtinFixer as gtf

# k = 8273
k = 88241 # K Total

all_acc = 0


for i in range(k):
    dataTmp = pd.read_csv("clusters/cluster" + str(i) +".csv")

    uniques_gtin, count_gtin = np.unique(dataTmp['gtin'].values, return_counts=True)
    all_acc += (np.max(count_gtin)/np.sum(count_gtin))

all_acc/=k
print(all_acc)
