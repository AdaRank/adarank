import random

import numpy as np
from src.quality_meter.quality_measure import QualityMeasure
from scipy.spatial import distance
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt

class Diversity(QualityMeasure):
    """
    Computes diversity as a quality measure
    Inherits from class QualityMeasure, has to implement abstract class compute_quality
    """
    def __init__(self, y, y_hat, max_val):
        super().__init__(y, y_hat, max_val)
        self.distance_matrix =[]

    @staticmethod
    def sparse_cos_dist(x, y):
        """Computes cosine distance on sparse vectors, i.e. vectors consisting of indices of the elements in a reference list

        :param x: vector x
        :param y: vector y
        :return: cosine distance
        """
        space = list(set(x + y))
        x_ = [x.count(i) for i in space]
        y_ = [y.count(i) for i in space]
        return distance.cosine(x_, y_)

    def compute_quality(self):
        """
        Creates distance matrix by computing the cosine distance of each given pair of vectors (cartesian product)

        :return: Dendrogram as a representative of the overall diversity
        """
        #sample = self.y[:1000]
        sample = random.sample(self.y, k=1000)
        c = list(combinations(sample, 2))
        #print(c)
        row = []
        number_of_columns = len(sample)
        
        current_row = 0
        it = 0
        i = 0
        for y in range(number_of_columns-1):
            #print("Computing row {}/{}".format(i,number_of_columns))
            i+=1
            for x in range(current_row+1):
                row.append(0.)
            #print(row)
            #print(len(row), number_of_columns)
            while len(row) < number_of_columns:
                #print(it)
                row.append(self.sparse_cos_dist(c[it][0], c[it][1]))
                it += 1
                #print(row)
            self.distance_matrix.append(row)
            row = []
            current_row += 1
        self.distance_matrix.append([0.]*number_of_columns)
        X = np.array(self.distance_matrix)
        X = X + X.T - np.diag(np.diag(X))
        #print(X)
        cluster_dist = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity="precomputed",
                                               linkage="average")
        cluster_dist.fit(X)
        self.plot_dendrogram(cluster_dist, truncate_mode='level', p=2)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        return plt

    @staticmethod
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def get_distance_matrix(self):
        return self.distance_matrix



