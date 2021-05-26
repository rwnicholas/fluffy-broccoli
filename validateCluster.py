#!/usr/bin/python3
import pandas as pd
import numpy as np

def accClusters(k):
    all_acc = 0
    tmp_k = k

    for i in range(k):
        dataTmp = pd.read_csv("clusters/cluster" + str(i) +".csv")

        if dataTmp['gtin'].count() == 1:
            tmp_k-=1
            continue

        uniques_gtin, count_gtin = np.unique(dataTmp['gtin'].values, return_counts=True)
        all_acc += (np.max(count_gtin)/np.sum(count_gtin))

    all_acc/=tmp_k
    print("Valor de K:", tmp_k)
    return all_acc