from sklearn.cluster import DBSCAN

from ny_felony_analysis.clusterization.clusterers.clusterer import Clusterer


class DensityBased(Clusterer, DBSCAN):
    name = "Density-Based Spatial Clustering of Applications with Noise"

    def __init__(self, data, args):
        super().__init__(data)

        try:
            eps = int(args[0])
            min_samples = float(args[1])

            self.eps = eps if eps > 0 else 17
            self.min_samples = min_samples if min_samples > 0 else 7
            self.model = DBSCAN(eps=self.eps,
                                min_samples=self.min_samples)

            print('Number of clusters: ', max(self.model.fit_predict(self.data)) + 1)
        except:
            self.model = DBSCAN()
