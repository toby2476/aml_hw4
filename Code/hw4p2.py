import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.vq import vq

from pylab import rcParams

import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score

import glob
import os

import collections


def import_data(folder):

	path = "../Data/HMP_Dataset/"+folder+"/"
	all_files = glob.glob(os.path.join(path, "*.txt")) #make list of paths
	content = []

	for file in all_files:
    		# Getting the file name without extension
    		file_name = os.path.splitext(os.path.basename(file))[0]
    		# Reading the file content to create a DataFrame
    		dfn = pd.read_csv(file, sep=" ")
    		content.append(dfn.as_matrix())
	
	return content



def import_all(): #Combines data from each folder - returns values in form train[folder][file] or test[folder][file] (as a list)
	folders = ["Brush_teeth","Climb_stairs","Comb_hair","Descend_stairs","Drink_glass","Eat_meat","Eat_soup","Getup_bed","Liedown_bed","Pour_water","Sitdown_chair","Standup_chair","Use_telephone","Walk"]
	train = []
	test = []
	for i in folders:
		data = import_data(i)
		size = len(data)
		train_size = int(0.8*size)
		train_data = data[0:train_size]
		test_data = data[train_size:]
		train.append(train_data)
		test.append(test_data)
	
	retval = [train,test]	
	return retval


def get_samples(data,segment_length):
	
	samples = [] #Will contain segments of data with fixed size
	start_val_all = [] #This list will contain the starting sample number denoting the beginning of each [folder][file]

	for b in range(len(data)):
		start_val = []
		for a in range(len(data[b])):
			start_val.append(len(samples))
			num_samples = len(data[b][a])
			num_segments = (int)(num_samples/(segment_length)) #Number of segments within file
			for i in range(num_segments):
				start_idx = i*(segment_length) #Starting index for segment
				end_idx = start_idx + segment_length #Ending index for segment
				sample = data[b][a][start_idx:end_idx]
				sample = sample.flatten('F') #Flatten segment to 2D (in column major order)
				samples.append(sample) #Add segment to list samples

			
		start_val_all.append(start_val)
		
	samples = np.array(samples) #Convert 'samples' to array
	retval = [samples,start_val_all]
	return retval
	

def k_means(samples,k):
	
	cluster = KMeans(n_clusters=k).fit(samples) #Do kmeans on all fixed length segments
	pred = cluster.predict(samples)
	centers = cluster.cluster_centers_
	retval = [pred, centers]
        ss = silhouette_score(samples,pred,metric='euclidean') #A metric to denote how well data is clustering
	#print(ss) 
	return retval;


def get_histogram(data,pred,centers,sample_size,startval): #Make histogram showing how many segments were assigned to each cluster for each file
	hist = []
	for i in range(len(data)):
		for j in range(len(data[i])):	
			
			num_centers = len(centers)
			signal_len = int(len(data[i][j])/sample_size)
			prediction = (pred[startval[i][j]:startval[i][j]+signal_len]).tolist() #Make list of cluster predictions for current file

			#if j==0 or j==1:
			#	plt.hist(prediction, bins=480)
			#	plt.show()

			
			values = []
			for x in range(num_centers):
				values.append(prediction.count(x)) #Count frequencies of cluster predictions in file
			values = [(float(x)/sum(values)) for x in values] #Normalize frequencies
			values.append(i) #Add labels as last column
			hist.append(values)
	hist = np.array(hist) #This list contains frequencies of cluster predictions for all files
	return hist


def classify_data(train_hist,test_hist): #Random forest classification
	rf = RandomForestClassifier(n_estimators=30,max_depth=20)
	train_labels = train_hist[:,-1]
	train_data = train_hist[:,:-1]
	test_labels = test_hist[:,-1]
	test_data = test_hist[:,:-1]
	rf.fit(train_data,train_labels)
	test_pred = rf.predict(test_data)
	cm = confusion_matrix(test_labels,test_pred) #Confusion Matrix
	print(cm)
	score = rf.score(test_data,test_labels) #Accuracy score
	return score
	
	

NUM_SAMPLES = 16 #Size of each segment
K = 480 #Number of clusters

[train,test] = import_all()
[train_samples, train_startval] = get_samples(train,NUM_SAMPLES)
print(train_samples.shape)
[test_samples, test_startval] = get_samples(test,NUM_SAMPLES)
[train_pred, train_centers] = k_means(train_samples,K)
[test_pred, test_centers] = k_means(test_samples,K)
train_hist = get_histogram(train,train_pred,train_centers,NUM_SAMPLES,train_startval)
test_hist = get_histogram(test,test_pred,test_centers,NUM_SAMPLES,test_startval)
score = classify_data(train_hist,test_hist)
print(score)







