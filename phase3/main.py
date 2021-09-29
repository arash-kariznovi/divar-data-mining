import pandas as pd
import Cluster_me as CL

if __name__ == "__main__" :

# 1
    df = pd.read_csv('orders.csv')

    df.city_name_fa = pd.factorize(df.city_name_fa)[0]
    df = df[['city_name_fa', 'ID_Item']]

    digikala_cluster = CL.Clusters(df)
    digikala_cluster.make_clusters()
# 2
    df_divar = pd.read_csv('divar_posts_dataset.csv')

    df_divar.cat3 = pd.factorize(df_divar.cat3)[0]
    df_divar.city = pd.factorize(df_divar.city)[0]
    df_divar = df_divar[['city','cat3']]
    df_divar = df_divar.rename(columns={"city":"city_name_fa","cat3":"ID_Item"})
    #
    divar_cluster = CL.Clusters(df_divar)
    divar_cluster.make_clusters()
# # 4
    df3 = pd.read_csv('tarikhche kharid.csv')

    df3 = df3[['selling_price','product_id']]
    print(df3.head())
    df3 = df3.rename(columns={"selling_price":"city_name_fa","product_id":"ID_Item"})

    digikala_sale = CL.Clusters(df3)
    digikala_sale.make_clusters()