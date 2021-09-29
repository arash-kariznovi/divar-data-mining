from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering

from matplotlib import pyplot as plt
from numpy import unique
from numpy import where

class Clusters:

    def __init__(self,df):
        self.df = df.sample(n=1000)


    def make_clusters(self):
        fig, axes = plt.subplots(nrows=4, ncols=3)
        fig.tight_layout()
        #######################################################################################
        # KMeans
        model = KMeans(6)

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4, 3, 1)
        plt.title("KMeans")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter(self.df.loc[row_ix,'city_name_fa'], self.df.loc[row_ix,  'ID_Item'])

        ######################################################################################
        # DBScan
        model = DBSCAN(eps=0.3,min_samples=8)

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4, 3, 2)
        plt.title("DBSCAN")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            x2 = plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])

    #######################################################################################
        # AffinityPropagation
        model = AffinityPropagation(damping=0.9)

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4, 3, 3)
        plt.title("AffinityPropagation")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])

    #######################################################################################
        # AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=4)

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)


        plt.subplot(4, 3, 4)
        plt.title("AgglomerativeClustering")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])

    #######################################################################################
        #Birch
        model = Birch(threshold=0.01, n_clusters=4)

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4, 3, 5)
        plt.title("Birch")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])

    ######################################################################################
        # MiniBatchKMeans

        model = MiniBatchKMeans(n_clusters=3)

        y_predicted = model.fit_predict(self.df)

        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4,3,6)
        plt.title("MiniBatchKMeans")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter( self.df.loc[row_ix, 'city_name_fa'],self.df.loc[row_ix, 'ID_Item'])
        ######################################################################################

        # GaussianMixture

        model = GaussianMixture(n_components=8)

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4, 3, 7)
        plt.title("GaussianMixture")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])
        ######################################################################################
        # MeanShift

        model = MeanShift()

        y_predicted = model.fit_predict(self.df)
        clusters = unique(y_predicted)
        self.df = self.df.reset_index(drop=True)

        plt.subplot(4, 3, 8)
        plt.title("MeanShift")
        for cluster in clusters:
            row_ix = where(y_predicted == cluster)
            plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])
        #######################################################################################
    #     # OPTICS
    #
    #     model = OPTICS(eps=0.8, min_samples=10)
    #
    #     y_predicted = model.fit_predict(self.df)
    #     clusters = unique(y_predicted)
    #     self.df = self.df.reset_index(drop=True)
    #
    #     plt.subplot(4, 3, 9)
    #     plt.title("OPTICS")
    #     for cluster in clusters:
    #         row_ix = where(y_predicted == cluster)
    #         plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])
        #######################################################################################
            # SpectralClustering

            model = SpectralClustering(n_clusters=3)

            y_predicted = model.fit_predict(self.df)
            clusters = unique(y_predicted)
            self.df = self.df.reset_index(drop=True)

            plt.subplot(4, 3, 10)
            plt.title("SpectralClustering")
            for cluster in clusters:
                row_ix = where(y_predicted == cluster)
                plt.scatter(self.df.loc[row_ix, 'city_name_fa'], self.df.loc[row_ix, 'ID_Item'])
            #######################################################################################

        plt.subplot(4, 3, 12)
        plt.title("Primary")
        plt.scatter(self.df['city_name_fa'], self.df['ID_Item'])

        plt.show()









