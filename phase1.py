import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    df_main = pd.read_csv('divar_posts_dataset.csv')
    df_digikala = pd.read_csv('orders.csv')


    #############################################
    ##                Phase 1                  ##
    #############################################
  # Uncomment to see the result of every Question:
# Question 1:
#     del df['id']
#     dataframe = df.iloc[:,1:]
    # print("\n****Mode****\n")
    # print(dataframe.mode(), '\n')
    # print("\n****Maximum****\n")
    # print(dataframe.max(numeric_only=True))
    # print("\n****Minimum****\n")
    # print(dataframe.min(numeric_only=True))
    # print("\n****Mean****\n")
    # print(dataframe.mean(numeric_only=True))
    # print("\n****Median****\n")
    # print(dataframe.median(numeric_only=True))
    #
    # df.boxplot(column=["image_count"])
    # plt.show()
    # df.boxplot(column=["mileage"])
    # plt.show()
    # plt.ylim(-1000000,4000000)
    # df.boxplot(column=["price"],grid=5000)
    # plt.show()
    # df.plot.bar(x="C", y="Visits", rot=70, title="Number of tourist visits - Year 2018");

# Question 4:
#     print(df.shape)
    # df.info()
    # print(df.count()/len(df))
    # is_Null = df[df.isnull().any(axis=1)]
    # measure1 = (1-(len(is_Null)/len(df)))*100
    # print(f'Completeness percentage 1 : {measure1}%')
    #
    # measure2 = (1-(df.isna().sum().sum()/ (len(df)*len(df.columns))))*100
    # print(f'Completeness percentage 2 : {measure2}%')

# Question 5:
    # sums = df.groupby(df["platform"])["price"].sum()
    # plt.pie(sums, labels=sums.index,autopct='%1.1f%%')
    # plt.title(" % Market value measured by Platform")
    # plt.show()
    # #
    # sums = df.groupby(df["city"])["price"].sum()
    # plt.pie(sums, labels=sums.index, autopct='%1.1f%%')
    # plt.title(" % Market value of cities")
    # plt.show()
    # #
    # plt.title("Effect of mileage to price")
    # plt.ylabel("Price")
    # plt.xlabel("mileage")
    # plt.scatter(df['mileage'],df['price'])
    # plt.show()
    #
    # sums = df.groupby(df["year"])["price"].sum()
    # plt.pie(sums, labels=sums.index,autopct='%1.1f%%')
    # plt.title(" % Market value per year")
    # plt.show()
    #
    #
    # plt.title("Distribution of Sales")
    # plt.ylabel("Value of Market")
    # df.groupby('type').price.sum().plot.bar()
    # plt.show()

    #############################################
    ##                Phase 2                  ##
    #############################################

    #Drop ids and Desc as addition data

    # to_drop = ['id','desc']
    # df_main.drop(to_drop,inplace=True,axis=1)
    #
    # #Chanege -1 prices with mean of all prices
    #
    # price_mean = df_main[["price"]].mean()
    # df_main = df_main.replace([-1], price_mean)
    # print(df_main.head(10))
    #
    # #Question1:
    #
    # df = df_main[['city', 'cat1', 'cat2', 'cat3']]
    #
    #
    # # Removing rows with Nan cat3
    # df = df.dropna(subset=['cat3'],inplace = False)
    #
    #
    # # cities
    # print(len(df.city.unique()),'Cities:',df['city'].unique())
    #
    #
    # # Total number of each Category
    # print(f'Total number of each Category: \n cat1:{len(df.cat1.unique())}\tcat2:{len(df.cat2.unique())}\t cat3:{len(df.cat3.unique())}')


    # Total number of each Category in each city
    # for i in df['city'].unique():
    #     print(len(df.loc[df['city']==i].cat1.unique()),
    #           len(df.loc[df['city']==i].cat2.unique()),
    #           len(df.loc[df['city']==i].cat3.unique()))




    # Question1 : frequent pattern of sale in different cities:

    # df_groupby_city = df.groupby(['city'])
    #
    # print('############################# Frequent Pattern of SALE in different cities #############################################')
    # for city in df_groupby_city:
    #     print(f'Frequent Pattern for {city[0]}:')
    #     df_cat1_cat2 = []
    #     for cat in city[1]['cat1'].unique():
    #         df_cat1_cat2.append(city[1].loc[city[1]['cat1'] == cat].cat2.unique())
    #
    #     te = TransactionEncoder()
    #     te_ary = te.fit(df_cat1_cat2).transform(df_cat1_cat2)
    #     df_cat1_cat2_1 = pd.DataFrame(te_ary, columns=te.columns_)
    #
    #
    #     # print(df_cat1_cat2_1)
    #
    #     fq_1 = fpgrowth(df_cat1_cat2_1, min_support=0.15, use_colnames=True)
    #     fq_1['length'] = fq_1['itemsets'].apply(lambda x: len(x))
    #
    #     # print(fq_1[(fq_1['length'] >= 2) &
    #     #                  (fq_1['support'] >= 0.1)])
    #
    #     rules = association_rules(fq_1, metric="lift", min_threshold=0.5)
    #     rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    #     print(rules.head())
    #     print('#######################################################################')

    #Question 2:

    # print(len(df_digikala.ID_Order.unique()))
    # print(len(df_digikala.ID_Customer.unique()))
    # print(len(df_digikala.ID_Item.unique()))
    # print(df_digikala.head())
   #
   #  df_digikala.dropna(axis=0, subset=['ID_Order'], inplace=True)
   #  df_digikala['ID_Order'] = df_digikala['ID_Order'].astype('str')
   #
   #  # print(len(df_digikala.ID_Order.unique()))
   #
   #  basket = (df_digikala[df_digikala['city_name_fa'] == "گرگان"]
   #            .groupby(['ID_Customer', 'ID_Item'])['Quantity_item']
   #            .sum().unstack().reset_index().fillna(0)
   #            .set_index('ID_Customer'))
   #
   #
   #  def encode_units(x):
   #      if x <= 0:
   #          return 0
   #      if x >= 1:
   #          return 1
   #
   #
   #  basket_sets = basket.applymap(encode_units)
   # # print(basket_sets)
   #
   #  frequent_itemsets = fpgrowth(basket_sets, min_support=0.001, use_colnames=True)
   #
   #  rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
   #  print(rules.head(30))


    # Question 4:

    # df_3 = df_main[["created_at", "platform"]]
    #
    # # Each array is plaform of one cat
    # df_sale_platform = []
    # for cat in df_3['platform'].unique():
    #     df_sale_platform.append(df_3.loc[df_3['platform'] == cat].created_at.unique())
    #
    # te = TransactionEncoder()
    # te_ary = te.fit(df_sale_platform).transform(df_sale_platform)
    # df_sale_platform_1 = pd.DataFrame(te_ary, columns=te.columns_)
    #
    # print(df_sale_platform_1)
    #
    # print(fpgrowth(df_sale_platform_1, min_support=0.1, use_colnames=True))

    #Question 5:


    # df_2 = df_main[["cat3","platform"]]
    #
    #
    # # Each array is plaform of one cat
    # df_sale_platform = []
    # for cat in df_2['cat3'].unique():
    #     df_sale_platform.append(df_2.loc[df_2['cat3']==cat].platform.unique())
    #
    #
    # te = TransactionEncoder()
    # te_ary = te.fit(df_sale_platform).transform(df_sale_platform)
    # df_sale_platform_1 = pd.DataFrame(te_ary, columns=te.columns_)
    #
    # print(df_sale_platform_1)
    #
    # fq_sale_platform_1 = fpgrowth(df_sale_platform_1, min_support=0.1 , use_colnames=True)
    #
    # rules_2 = association_rules(fq_sale_platform_1, metric="lift", min_threshold=0.5)
    # rules_2 = rules_2.sort_values(['confidence', 'lift'], ascending=[False, False])
    # print(rules_2.head(10))


    

