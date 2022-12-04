# %matplotlib inline
import matplotlib as matplotlib


from time import sleep
from IPython.display import display, clear_output
import matplotlib as mpl
import matplotlib.animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import lines
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs # Datasets
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split  # Cross validation library

# Additional functions for data visualization
from i2ai_utilities import show_scatter_3d, show_scatter, plotly_scatter_3d, show_risk_by_cluster, plot_dendrogram
from sklearn.datasets import load_sample_image
from PIL import Image

def moons():
    X, y = make_moons(n_samples=200, noise=0.05)
    # show_scatter(X)

    k = 2
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    print(kmeans.inertia_)
    show_scatter(X, y_pred, centers)

    km_list = list()

    for k in range(1, 10):
        km = KMeans(n_clusters=k)
        y_pred = km.fit(X)
        km_list.append(pd.Series({'clusters': k,
                                  'inertia': km.inertia_}))
    plot_data = (pd.concat(km_list, axis=1)
                 .T
                 [['clusters', 'inertia']]
                 .set_index('clusters'))

    ax = plot_data.plot(marker='o', ls='-')
    ax.set_xticks(range(0, 10, 1))
    ax.set_xlim(0, 10)
    ax.set(xlabel='Cluster', ylabel='Inertia')
    plt.show()
    ###########################################


    dbscan = DBSCAN(eps=0.3)
    y_pred = dbscan.fit_predict(X)
    print('Number of clusters:', len(set(y_pred)) - (1 if -1 in y_pred else 0))
    print('Number of outliers:', list(y_pred).count(-1))
    show_scatter(X, y_pred)
    #########################################

    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=3,
                                 affinity='euclidean', linkage='complete')
    y_pred = ac.fit_predict(X)
    print('Number of clusters:', len(set(y_pred)))
    plot_dendrogram(ac, truncate_mode='level', p=4)
    show_scatter(X, y_pred)

    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=3,
                                 affinity='l1', linkage='complete')
    y_pred = ac.fit_predict(X)
    print('Number of clusters:', len(set(y_pred)))
    plot_dendrogram(ac, truncate_mode='level', p=4)
    show_scatter(X, y_pred)

def circles():
    X, y = make_circles(n_samples=200, factor=0.5, noise=0.05)
    # show_scatter(X)

    k = 2
    kmeans = KMeans(n_clusters=k)
    y_pred = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    print(kmeans.inertia_)
    show_scatter(X, y_pred, centers)

    km_list = list()

    for k in range(1, 10):
        km = KMeans(n_clusters=k)
        y_pred = km.fit(X)
        km_list.append(pd.Series({'clusters': k,
                                  'inertia': km.inertia_}))
    plot_data = (pd.concat(km_list, axis=1)
                 .T
                 [['clusters', 'inertia']]
                 .set_index('clusters'))

    ax = plot_data.plot(marker='o', ls='-')
    ax.set_xticks(range(0, 10, 1))
    ax.set_xlim(0, 10)
    ax.set(xlabel='Cluster', ylabel='Inertia')
    plt.show()
    ###########################################


    dbscan = DBSCAN(eps=0.25)
    y_pred = dbscan.fit_predict(X)
    print('Number of clusters:', len(set(y_pred)) - (1 if -1 in y_pred else 0))
    print('Number of outliers:', list(y_pred).count(-1))
    show_scatter(X, y_pred)
    ########################################

    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=2.2,
                                 affinity='euclidean', linkage='complete')
    y_pred = ac.fit_predict(X)
    print('Number of clusters:', len(set(y_pred)))
    plot_dendrogram(ac, truncate_mode='level', p=4)
    show_scatter(X, y_pred)

    ########################################

    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=3,
                                 affinity='l1', linkage='complete')
    y_pred = ac.fit_predict(X)
    print('Number of clusters:', len(set(y_pred)))
    plot_dendrogram(ac, truncate_mode='level', p=4)
    show_scatter(X, y_pred)

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data

    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);

def photo():
    china = Image.open('NewYork.jpg')
    china = np.asarray(china)

    ax = plt.axes(xticks=[], yticks=[])
    ax.imshow(china)
    plt.show()


    print(china.shape)

    data = china / 255.0  # use 0...1 scale
    data = data.reshape(2368 * 4209, 3)
    print(data.shape)

    plot_pixels(data, title='Input color space: 16 million possible colors')
    plt.show()

    kmeans = KMeans(n_clusters=16)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

    plot_pixels(data, colors=new_colors,
                title="Reduced color space: 16 colors")

    china_recolored = new_colors.reshape(china.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(china)
    ax[0].set_title('Original Image - 16 million possible colors', size=16)
    ax[1].imshow(china_recolored)
    ax[1].set_title('16-color Image', size=16);
    plt.show()

    ########################################### 10 colors

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

    plot_pixels(data, colors=new_colors,
                title="Reduced color space: 10 colors")

    china_recolored = new_colors.reshape(china.shape)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(wspace=0.05)
    ax[0].imshow(china)
    ax[0].set_title('Original Image', size=16)
    ax[1].imshow(china_recolored)
    ax[1].set_title('10-color Image', size=16);
    plt.show()

if __name__ == '__main__':
    # moons()
    # circles()
    photo()



