import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
sns.set(style="whitegrid")

if __name__=="__main__":
    df = pd.read_csv("上海车牌价格_v2.csv")
    # df = df.drop(['avg price'], axis=1)
    df = df.drop(['Date'], axis=1)
    # df = df.drop(list(range(206, 230)))
    # df['difference'] = df['avg price'] - df['lowest price ']
    # df = df.drop(index=(df.loc[(df['Date']=='2-Dec')].index))
    # df = df.drop(index=(df.loc[(df['Date']=='8-Jan')].index))
    # df = df.drop(index=(df.loc[(df['Date']=='10-Dec')].index))
    # df = df.drop(list(range(143)))

    # Plot the responses for different events and regions
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    # df = df.drop(list(range(143)))
    sns.lineplot(data=df, palette="tab10", linewidth=2.5)
    # sns.lineplot(x=range(df.shape[0]),y='difference', data=df)
    plt.savefig('general_with_avg.jpg')

    # cols = ['date', 'key', 'value']
    # df_new = pd.DataFrame(columns=cols)
    # new_row = {}
    # for i in range(df.shape[0]):
    #     new_row['date'] = i
    #     new_row['key'] = 'Total number of license issued'
    #     new_row['value'] = int(df.iloc[i,1])
    #     df_new = df_new.append(new_row, ignore_index=True)
    #     new_row = {}
    #     new_row['date'] = i
    #     new_row['key'] = 'lowest price'
    #     new_row['value'] = int(df.iloc[i,2])
    #     df_new = df_new.append(new_row, ignore_index=True)
    #     new_row = {}
    #     new_row['date'] = i
    #     new_row['key'] = 'Total number of applicants'
    #     new_row['value'] = int(df.iloc[i,3])
    #     df_new = df_new.append(new_row, ignore_index=True)

    # sns.relplot(x='date', y="value",
    #             hue="key", # style="choice",
    #             hue_norm=LogNorm(),
    #             kind="line",
    #             data=df_new);
    # df = df.drop(index=(df.loc[(df['Date']=='13-Oct')].index))
    # sns.lineplot(x=range(df.shape[0]), y="lowest price ",
    #             data=df)
    # plt.show()