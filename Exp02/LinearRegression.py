import numpy as np
import pandas as pd
import random
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt

def data_preprocessing(df, lowest_month, lowest_mean):
    # create new dataframe
    cols = []
    if lowest_mean:
        cols.append('lowest')
    else:
        for i in range(lowest_month):
            cols.append('lowest_'+str(i+1))
            # cols.append('e_'+str(i+1))
    cols.append('total_number_change')
    # cols.append('applicant/total number')
    cols.append('applicant_tendency')
    cols.append('lowest_tendency')
    cols.append('random')
    cols.append('y')
    df_new = pd.DataFrame(columns=cols)

    # fill in the dataframe
    for i in range(lowest_month, df.shape[0]):
        new_row = {}
        row = i-lowest_month
        if lowest_mean:
            sum_lowest = 0
            for j in range(lowest_month):
                sum_lowest += int(df.iloc[row,2]) # *(1+(j+1)*0.004)
                row += 1
            new_row['lowest'] = round(sum_lowest/lowest_month)
            lowest_month = 1
        else:
            for elem in cols[:end]:
                if elem[:6]=="lowest":
                    new_row[elem] = int(df.iloc[row, 2])# *random.gauss(0, 0.5)
                    row += 1
                else:
                    new_row[elem] = random.gauss(0,1)
        # new_row['applicant/total number'] = int(df.iloc[row-2,3])/int(df.iloc[row-1,1])
        new_row['total_number_change'] =  int(df.iloc[row,1])/int(df.iloc[row-1,1]) # + random.gauss(0, 1)
        new_row['applicant_tendency'] = int(df.iloc[row,3])/int(df.iloc[row-1,3]) # + random.gauss(0, 1)
        new_row['lowest_tendency'] = int(df.iloc[row-1,2])/int(df.iloc[row-2,2]) # + random.gauss(0, 1)
        new_row['random'] = random.gauss(0, 1)
        new_row['y'] = int(df.iloc[row, 2])
        df_new = df_new.append(new_row, ignore_index=True)
    return df_new

if __name__=="__main__":
    # 读取数据，删除异常值
    df = pd.read_csv("上海车牌价格.csv")
    df = df.drop(['avg price'], axis=1)
    # df = df.drop(index=(df.loc[(df['Date']=='2-Dec')].index))
    # df = df.drop(index=(df.loc[(df['Date']=='8-Jan')].index))
    # df = df.drop(index=(df.loc[(df['Date']=='10-Dec')].index))
    df = df.drop(list(range(143)))
    df = df.drop(index=(df.loc[(df['Date']=='13-Oct')].index))


    # 数据预处理：发行量归一化（没有实际使用）
    arr = df['Total number of license issued'].values.reshape(-1,1)
    arr = MinMaxScaler().fit_transform(arr)
    df['Total number of license issued_pro'] = arr
    print(df.head())
    lowest_month = 5                 # 选择前几月的最低价
    df_new = data_preprocessing(df, lowest_month, lowest_mean=True)
    print(df_new.tail())

    # 分隔测试集和训练集
    data = df_new.values
    X = data[:,:-1]
    y = data[:,-1]
    print(X.shape)
    # poly_features = PolynomialFeatures(degree=1, include_bias=True)
    # poly_X = poly_features.fit_transform(X)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=True)

    # 线性回归
    reg = LinearRegression()
    scores = cross_val_score(reg, train_X, train_y, cv=6, scoring='neg_mean_absolute_error')
    print('LinearRegression_Validation_Set_MAE:', -scores.mean())

    # SVC
    # svc = SVR(kernel='rbf')
    # scores = cross_val_score(svc, train_X, train_y, cv=6, scoring='neg_mean_absolute_error')
    # print('SVR:', -scores.mean())

    # Adaboost Regressor
    # ada_reg = AdaBoostRegressor(reg)
    # scores = cross_val_score(ada_reg, train_X, train_y, cv=6, scoring='neg_mean_absolute_error')
    # print('Adaboost Regressor:', -scores.mean())

    # Bagging Regressor
    # bag_reg = BaggingRegressor(reg)
    # scores = cross_val_score(bag_reg, train_X, train_y, cv=6, scoring='neg_mean_absolute_error')
    # print('Bagging Regressor:', -scores.mean())

    # 训练+测试
    train_reg = reg.fit(train_X, train_y)
    y_pre = reg.predict(test_X)
    print(min(y_pre))
    print(max(y_pre))
    print(reg.coef_)
    mae = MAE(y_pre, test_y)
    print('LinearRegression_Test_MAE', mae)
    for i in range(y_pre.shape[0]):
        print(y_pre[i], test_y[i], y_pre[i]-test_y[i])


    # 绘制预测折线图
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    df = df.drop(['Date','Total number of license issued',
                'Total number of applicants', 'Total number of license issued_pro'],axis=1)
    # print(df.head())
    df = df.drop(list(range(143, 143+lowest_month)))
    y_pre = reg.predict(X)
    if df.shape[0]!= y_pre.shape[0]:
        print("!")
    df['pre'] = np.nan
    for i in range(df.shape[0]):
        df.iloc[i, 1] = y_pre[i]
    # print(df.head())
    sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    # plt.show()
    # sns.lineplot(x=range(df.shape[0]),y='difference', data=df)
    plt.savefig('predict_3.png')