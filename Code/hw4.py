import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from pylab import rcParams

import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


data = pd.read_csv('../Data/data.txt', sep="\t")
columns = np.array(data.columns)
data = np.array(data)
labels = data[:,0]
data = data[:,1:]

single = linkage(data,'single')
dendrogram(single, labels=labels)
plt.title('Single Link Agglomerative Clustering Dendrogram')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.show()

complete = linkage(data,'complete')
dendrogram(complete, labels=labels)
plt.title('Complete Link Agglomerative Clustering Dendrogram')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.show()

group_avg = linkage(data,'average')
dendrogram(group_avg, labels=labels)
plt.title('Group Average Link Agglomerative Clustering Dendrogram')
plt.xlabel('Country')
plt.ylabel('Distance')
plt.show()

def cluster_cost(k):
	cluster = KMeans(n_clusters=k).fit(data)
	prediction = cluster.predict(data)
	centers = cluster.cluster_centers_

	cost = 0
	for i in range(len(prediction)):
		cost += np.dot((data[i] - centers[prediction[i]]),(data[i] - centers[prediction[i]]))
	return cost 

k_max = 26
cost_list = np.zeros((k_max-1))
for i in range(2,k_max+1):
	cost_list[i-2] = cluster_cost(i)
cost_func = pd.DataFrame(cost_list, index=np.linspace(2,k_max,num=k_max-1))
knee = cost_func.plot(kind='line',title='Total Within-Cluster Sum of Square Distance with K Clusters',figsize=(10,8),fontsize=12,legend=False)
knee.set_xlabel('K',fontsize=12)
knee.set_ylabel('Total Within-Cluster Sum of Square Distance',fontsize=12)
plt.show()

