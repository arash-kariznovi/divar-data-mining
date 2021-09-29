import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from apyori import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    df_main = pd.read_csv('divar_posts_dataset.csv')
    df_digikala = pd.read_csv('orders.csv')

    #############################################
    ##                Phase 2                  ##
    #############################################
    #Some preprocessing:

    #Drop ids and Desc as addition data

    to_drop = ['id','desc']
    df_main.drop(to_drop,inplace=True,axis=1)

    #Change -1 prices with mean of all prices

    price_mean = df_main[["price"]].mean()
    df_main = df_main.replace([-1], price_mean)
    # print(df_main.head(10))

    df = df_main[['city', 'cat1', 'cat2', 'cat3','platform']]

    # Removing rows with Nan cat3
    df = df.dropna(subset=['cat3'],inplace = False)


    # cities
    print(len(df.city.unique()),'Cities:',df['city'].unique())


    # Total number of each Category
    print(f'Total number of each Category: \n cat1:{len(df.cat1.unique())}\tcat2:{len(df.cat2.unique())}\t cat3:{len(df.cat3.unique())}')


    # Total number of each Category in each city
    for i in df['city'].unique():
        print(i)
        print(len(df.loc[df['city']==i].cat1.unique()),
              len(df.loc[df['city']==i].cat2.unique()),
              len(df.loc[df['city']==i].cat3.unique()))


    # Question1 : frequent pattern of sale in different cities:

    print('############################# Frequent Pattern of SALE in different cities #############################################')

    df_cat1_cat2 = []
    for city in df['city'].unique():
        df_cat1_cat2.append(df.loc[df['city'] == city].cat2.unique())

    te = TransactionEncoder()
    te_ary = te.fit(df_cat1_cat2).transform(df_cat1_cat2)
    df_cat1_cat2_1 = pd.DataFrame(te_ary, columns=te.columns_)


    print(df_cat1_cat2_1)

    fq_1 = fpgrowth(df_cat1_cat2_1, min_support=0.5, use_colnames=True)

    fq_1['length'] = fq_1['itemsets'].apply(lambda x: len(x))
    # print(fq_1[(fq_1['length'] >= 2) &
    #                       (fq_1['support'] >= 0.05)])

    rules = association_rules(fq_1, metric="lift", min_threshold=1)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    print(rules.head())
    print('#######################################################################')
   #
   #  Question 2:
   #
    print(len(df_digikala.ID_Order.unique()))
    print(len(df_digikala.ID_Customer.unique()))
    print(len(df_digikala.ID_Item.unique()))
    print(df_digikala.head())

    df_digikala.dropna(axis=0, subset=['ID_Order'], inplace=True)
    df_digikala['ID_Order'] = df_digikala['ID_Order'].astype('str')

    # print(len(df_digikala.ID_Order.unique()))

    basket = (df_digikala[df_digikala['city_name_fa'] == "گرگان"]
              .groupby(['ID_Customer', 'ID_Item'])['Quantity_item']
              .sum().unstack().reset_index().fillna(0)
              .set_index('ID_Customer'))


    def encode_units(x):
        if x <= 0:
            return 0
        if x >= 1:
            return 1


    basket_sets = basket.applymap(encode_units)
   # print(basket_sets)

    frequent_itemsets = fpgrowth(basket_sets, min_support=0.001, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print(rules.head(30))


    # Question 4:

    # df_3 = df_main[["created_at", "platform"]]
    #
    #
    # df_sale_platform = []
    # for cat in df_3['platform'].unique():
    #     df_sale_platform.append(df_3.loc[df_3['platform'] == cat].created_at.unique())
    #
    # print(df_sale_platform)
    #
    # te = TransactionEncoder()
    # te_ary = te.fit(df_sale_platform).transform(df_sale_platform)
    # df = pd.DataFrame(te_ary, columns=te.columns_)
    # rule1 = fpgrowth(df,min_support=.1)
    # # rules1 = apriori(df_sale_platform, min_support=0.1, min_confidence=0.5)
    # print(rules1)
    # association_results1 = list(rules1)
    #
    # for item in rules1:
    #
    #     pair = item[0]
    #     items = [x for x in pair]
    #     print("Rule: " + items[0] + " -> " + items[1])
    #     print("Support: " + str(item[1]))
    #     print("Confidence: " + str(item[2][0][2]))
    #     print("Lift: " + str(item[2][0][3]))
    #     print("=====================================")
    #
    # #Question 5:
    #
    #
    # df_2 = df[["cat2","platform"]]
    #
    # df_sale_platform = []
    # for cat in df_2['platform'].unique():
    #     df_sale_platform.append(df_2.loc[df_2['platform']==cat].cat2.unique())
    #
    # rules = apriori(df_sale_platform, min_support=0.9, min_confidence=1)
    #
    # association_results = list(rules)
    # print(rules[0])
    # for item in rules:
    #     pair = item[0]
    #     items = [x for x in pair]
    #     print("Rule: " + items[0] + " -> " + items[1])
    #     print("Support: " + str(item[1]))
    #     print("Confidence: " + str(item[2][0][2]))
    #     print("Lift: " + str(item[2][0][3]))
    #     print("=====================================")


    

