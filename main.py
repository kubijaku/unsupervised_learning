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

X, y = make_moons(n_samples=200, noise=0.05)
show_scatter(X)
