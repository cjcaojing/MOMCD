#!/usr/bin/python
import sys
from numpy import random
from sklearn.cluster import KMeans
import os
import re
import random
import numpy as np
if __name__ == '__main__':
	feature_file = open(sys.argv[1])
	number_clusters = int(sys.argv[2])
	node_embedding = []
	node_list = []
	for line in feature_file:
		tmp = line.strip().split(' ')
		print(tmp)
		node_id = int(tmp[0])
		node_list.append(node_id)
		node_embedding.append([float(tmp[i]) for i in range(1,len(tmp))])
	for i in range(len(node_list)):
		print(node_list[i],":")
		print(node_embedding[i])
	kmeans = KMeans(n_clusters=10, random_state=0).fit(node_embedding)
	predict_label = kmeans.labels_
	clusters = [[] for i in range(number_clusters)]
	for i in range(len(predict_label)):
		clusters[predict_label[i]].append(node_list[i])
	out_clusters = open(sys.argv[3],'w')
	for i in clusters:
		out_clusters.write(' '.join(map(str,i))+'\n')
	print(predict_label)

