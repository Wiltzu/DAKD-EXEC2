from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pylab as pl
import task1, task2



def k_means(data):
    data = np.array(data)
    for i in [3,5,8,10]:
        name = 'k=%d'%i
        km = KMeans(n_clusters = i).fit(data)
        figure = pl.figure(name, figsize=(8,6))
        pl.clf()
        ax = Axes3D(figure, rect=[0, 0, .95, 1], elev=48, azim=134)

        pl.cla()
        labels = km.labels_
        ax.scatter(data[:,3], data[:,0], data[:,2], c=labels.astype(float))

    pl.show()


def main():
    original_data = task1.get_full_data()
    data, classifier = task2.separate_classifiers(original_data)
    k_means(data)

if __name__ == '__main__':
    main()
