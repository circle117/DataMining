import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

if __name__=="__main__":
    df = pd.read_csv("heros_new.csv", encoding="gbk")
    df = df.drop(['英雄','次要定位 '], axis=1)
    df = df[df['法力成长']!=0]
    cols = df.columns.tolist()[1:19]
    cols_new = []
    # normalization, PCA
    for i in range(0,len(cols),3):
        arr = df.loc[:,[cols[i+1],cols[i+2]]].values
        print(df.loc[:,[cols[i+1],cols[i+2]]].head(2))
        min_max_scaler = MinMaxScaler()
        pca = PCA(n_components=1)
        arr = min_max_scaler.fit_transform(arr)
        arr = pca.fit_transform(arr)

        df[cols[i][-2:]]=arr
        df = df.drop([cols[i],cols[i+1],cols[i+2]], axis=1)
        cols_new.append(cols[i][-2:])
    print(df.head())

    # trisection
    for i in range(len(cols_new)):
        df[cols_new[i]] = pd.qcut(df[cols_new[i]], q=3, labels=[cols_new[i]+'小', cols_new[i]+'中',cols_new[i]+'大'])
    
    # Apriori
    te = TransactionEncoder()
    te_ary = te.fit(df.values).transform(df.values)
    df_new = pd.DataFrame(te_ary, columns=te.columns_)
    freq = apriori(df_new, min_support=0.3, use_colnames=True)
    print(freq)

    # Select
    attack = set(df['攻击范围'].values)
    position = set(df['主要定位'].values)
    speed = set(df['最大攻速'].values)
    for i in range(freq.shape[0]):
        if len(freq.iloc[i,1])>1:
            for elem in freq.iloc[i,1]:
                if elem in attack:
                    print(freq.iloc[i,0], tuple(freq.iloc[i,1]))
                    break
                elif elem in position:
                    print(freq.iloc[i,0], tuple(freq.iloc[i,1]))
                    break
                elif elem in speed:
                    print(freq.iloc[i,0], tuple(freq.iloc[i,1]))
                    break