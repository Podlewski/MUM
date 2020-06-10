import pandas
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.common import factorize, correlate_sort, drop_infrequent, normalize
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

def optimalK(data, nrefs=3, maxClusters=15):
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            km = KMeans(k)
            km.fit(randomReference)
            refDisp = km.inertia_
            refDisps[i] = refDisp
        km = KMeans(k)
        km.fit(data)
        origDisp = km.inertia_
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        gaps[gap_index] = gap
        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)


def elbow(data):
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.clf()
    plt.close()
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig("elbow", dpi=200, bbox_inches='tight')


def kryteria(dataFrame):
    dawidekArray = []
    hrabaszczArray = []
    silchoutArray = []
    clusters = range(2, 11)
    for n_cluster in clusters:
        x = dataFrame
        kmeans = KMeans(n_clusters=n_cluster).fit(x)
        label = kmeans.labels_
        dawidek = davies_bouldin_score(x, label)
        hrabaszcz = calinski_harabasz_score(x, label)
        silchout = silhouette_score(x, label)
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, silchout))
        silchoutArray.append(silchout)
        print("For n_clusters={}, calinski_harabasz {}".format(n_cluster, hrabaszcz))
        hrabaszczArray.append(hrabaszcz)
        print("For n_clusters={}, davies_bouldin {}".format(n_cluster, dawidek))
        dawidekArray.append(dawidek)
    plt.clf()
    plt.close()
    plt.plot(clusters, silchoutArray, 'bx-')
    plt.xlabel('clusters')
    plt.ylabel('silhouette')
    plt.savefig("silhouette", dpi=200, bbox_inches='tight')

    plt.clf()
    plt.close()
    plt.plot(clusters, hrabaszczArray, 'bx-')
    plt.xlabel('clusters')
    plt.ylabel('calinski_harabasz')
    plt.savefig("calinski-harabasz", dpi=200, bbox_inches='tight')

    plt.clf()
    plt.close()
    plt.plot(clusters, dawidekArray, 'bx-')
    plt.xlabel('clusters')
    plt.ylabel('davies_bouldin')
    plt.savefig("davies_bouldin", dpi=200, bbox_inches='tight')

seaborn.set(color_codes=True)

data = pandas.read_csv('../../NYPD.csv')
data = data.drop(
    columns=['CMPLNT_TO_DT', 'CMPLNT_TO_TM'],
    axis=1
)

data = data.dropna()
data = drop_infrequent(data)
data = data.apply(factorize)
data = data.sample(n=6_000, random_state=666)

kryteria(data)
elbow(data)
optimalK(data)
x = data
k, gapdf = optimalK(x, nrefs=5, maxClusters=15)
print(f'Optymalny cluster:\t\t%0.4f' %k)
plt.clf()
plt.close()
plt.plot(gapdf.clusterCount, gapdf.gap,'bx-', linewidth=3,)
plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.savefig("GAP", dpi=200, bbox_inches='tight')
