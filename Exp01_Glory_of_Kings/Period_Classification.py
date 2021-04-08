import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
import graphviz
sns.set(font='SimHei')                      # 解决中文无法显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示问题

if __name__=="__main__":
    cols_1 = ['初始物防', '物防成长', '初始物攻', '物攻成长', '初始法力', '法力成长']
    df = pd.read_csv("heros_new.csv", encoding='gbk')
    df_new = df[cols_1]
    print(df['时期'].value_counts())

    # 数据处理
    # 归一化，PCA
    arr = df_new.values
    min_max_scaler = MinMaxScaler()
    arr_min_max = min_max_scaler.fit_transform(arr)

    max_accu = 0
    max_f1 = 0
    temp = [0,0,0,0]
    cols = ['1', '2', '3']
    for m in range(3,5):
        pca = PCA(n_components=m)
        arr_pca = pca.fit_transform(arr_min_max)
        X_train, X_test, Y_train, Y_test = train_test_split(arr_pca, df['时期'], test_size=0.2)
        for j in range(5,10):
            for k in range(2,7):
                for i in range(5,7):
                    clas = DecisionTreeClassifier(min_samples_leaf=i, min_samples_split=j, max_depth=k)
                    clas.fit(X_train, Y_train)
                    Y_pred = clas.predict(X_test)
                    # print(i, j, k)
                    if metrics.accuracy_score(Y_test, Y_pred)>max_accu:
                        temp[0] = i
                        temp[1] = j
                        temp[2] = k
                        temp[3] = m
                        max_accu = metrics.accuracy_score(Y_test, Y_pred)
                        max_f1 = metrics.f1_score(Y_test, Y_pred, average="macro")
                        dot_data = tree.export_graphviz(clas
                                        ,feature_names= cols
                                        ,class_names=["前期", "中期", "后期"]
                                        ,filled=True
                                        ,rounded=True
                                        )
                        graph = graphviz.Source(dot_data)
                    # print(metrics.f1_score(Y_test, Y_pred, average="macro"))
                    # print(metrics.accuracy_score(Y_test, Y_pred))
    print(temp, max_accu, max_f1)
    graph
    # graph.render(view=True, format="pdf", filename="decisiontree_pdf")