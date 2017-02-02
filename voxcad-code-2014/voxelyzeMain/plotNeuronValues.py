#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

inFile = open("neuronValues.txt")
totalTimesteps = 0
totalColumns = 0
for line in inFile:
	totalTimesteps += 1
	totalColumns = len(line.split())
inFile.close()
# totalTimesteps -= 1

data = np.zeros((totalTimesteps,totalColumns))

inFile = open("neuronValues.txt")
i = 0
for line in inFile:
	# print line
	splitLine = line.split()
	for j in range(totalColumns):
		data[i,j] = float(splitLine[j])
	i += 1
inFile.close()

# print data

labels = ["time","bias","pacemaker","proprio1","proprio2","hidden1L1","hidden2L1","hidden3L1","hidden1L2","hidden2L2","hidden3L2","output1","output2"]
colors = ["k","k","r","r","r","b","b","b","g","g","g","c","c"]
markers = ["","","^","o","s","^","o","s","^","o","s","^","o"]
for i in range(1,totalColumns):
	plt.plot(data[0:-1,0],data[0:-1,i],label=labels[i],color=colors[i],marker=markers[i],markersize=8)

plt.legend()

plt.show()