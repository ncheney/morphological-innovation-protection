#!/usr/bin/python

import hashlib
# import matplotlib.pyplot as plt
import random
import math
import numpy as np
# from Network import *
import networkx as nx
# import Image
import scipy as sp
import scipy.signal as signal
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import copy
import os
import time
import sys
import glob
# install scikit-image from http://scikit-image.org/docs/dev/install.html
# from skimage.morphology import skeletonize
# from Window import *
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import subprocess as sub
from pprint import *

#---------------------------------------------------------------------------------
# PARAMS
maxGenerations = 1000
popSize=25
mutationStd = 0.5

origSizeX = 10
origSizeY = 10
origSizeZ = 10
scalingFactor = 10

inputNodeNames = [
					['x','y','z','d','b'],
					['x','y','z','d','b']
					# ['x1','y1','x2','y2']
				 ]

outputNodeNames = [
					["materialPresent","materialHardOrSoft","materialMuscleOrTissue"],#,"materialMuscleType"], #,"materialHardOrSoft"],  <-- morphology network
					["phaseOffset","frequency"] # <-- controller network
					# ["materialPresent","materialMuscleOrTissue","materialHardOrSoft"],#,"materialMuscleType"], #,"materialHardOrSoft"],  <-- morphology network
					# ["materialMuscleType"] # <-- controller network
					# ['weight','presence']
				  ]

protectInnovatonAlong = -999 # -999 == no protection

nestedAgeInterval = -1 # -1 for no nesting

# substrateSize = [
# 					[10,10,10],
# 					[numTotalNeurons,numTotalNeurons,numTotalNeurons]
# 				]


# activationFunctionNames = ["sigmoid","sin","abs","square","sqrt","edge","gradient","erosion","dilation","opening","closing","coral-RD","worm-RD","spiral-RD","zebrafish-RD","fingerprint-RD","unstable-RD"]
# activationFunctionNames = ["sigmoid","sin","abs","square","sqrt","edge","gradient","erosion","dilation","opening","closing","coral-RD","spiral-RD","unstable-RD"]#,"worm-RD","zebrafish-RD","fingerprint-RD"]
# activationFunctionNames = ["sigmoid","sin","abs","edge","gradient","erosion","dilation","opening","closing"]
# activationFunctionNames = ["sigmoid","sin","abs","coralReaction"]
activationFunctionNames = ["sigmoid","sin","abs","nAbs","square","nSquare","sqrt","nSqrt"] # put in negative versions of abs, square, ...
#activationFunctionProbs = [0.3,0.3,0.2,0.1,0.1]
filterRadius = 5

# mutationsPerStep = 1

# resetNetworkProb = 0.01
# removeNodeProb = 0.02
# removeLinkProb = 0.05
# addNodeProb = 0.03
# addLinkProb = 0.10
# mutateWeightProb = 0.7
# mutFunctProb = 0.1

# equal chance of any mutation for testing 
# resetNetworkProb = 1.0
removeNodeProb = 1.0
removeLinkProb = 1.0
addNodeProb = 1.0
addLinkProb = 1.0
mutateWeightProb = 1.0
mutFunctProb = 1.0


# mutationChances = [resetNetworkProb,removeNodeProb,removeLinkProb,addNodeProb,addLinkProb,mutateWeightProb,mutFunctProb]
mutationChances = [removeNodeProb,removeLinkProb,addNodeProb,addLinkProb,mutateWeightProb,mutFunctProb]
mutationChances = [float(i)/sum(mutationChances) for i in mutationChances] # (normalized)
# mutationFunctions = [resetNetwork,removeNode,removeLink,addNode,addLink,mutateWeight,mutFunct]

proportionNewlyGenerated = 0.0 # 0.1
proportionFromCrossover = 0.0 # 0.45
proportionFromMutation = 1.0 # 0.45

# params for random network creation (todo: set via brute force search)
numRandomNodes = 5
numRandomLinkAdds = 10
numRandomWeightChanges = 100
numRandomLinkRemovals = 5
numRandomActivationFunction = 100

minimumFitness = 10**(-12)

# ------------------------------------------------------------------------
runName = "changeMe"
randomSeed = 1
maxID = 0
continuingRun = False

# -------------------------------------------------------------------------
# params for voxelyze
floorSlope = 0.0
inCage = 0
swarmClimb = 0
apertureProportion = 0.9

# fitnessEvaluationTime = 2.0
fitnessEvaluationCycles = 20
fitnessEvalInitTime = 0.5
if inCage:
	fitnessEvalInitTime = 0

# actuationsPerSecond = 10

minPercentFull = 0.10
minPercentMuscle = 0.05

saveVxaEvery = 100 # -1 to never save
saveAllIndividualData = True
saveCPPNs = True

SelfCollisionsEnabled = True

softestMaterial = 5 # default = 10, hardest is 2 orders of magnitude stiffer

# ----------------------------------------------------------------------
# run settings
# evolvingNeuralNets = True
mutateBodyProb = 0.5  # mutateBrainProb = 1-mutateBodyProb 


origFileForMutation = "none"
minVoxelPercentMutated  = 0.001
minVoxelDiffForAgeReset = 0.2
seedIndividual = None


#--------------------------------------------------------------------------
# command line interface tags:
# -r : random seed
# -n : name of run
# -s : size of softbot (cube)

for i in range(len(sys.argv[1:])):
	if sys.argv[i+1] == "-r":
		if sys.argv[i+2] == 'a':
			randomSeed = int(sub.check_output("pwd").split("/")[-1].split("_")[-1])
			print "autosetting random seed to",randomSeed,"based on directory"
			
		else:
			try:
				randomSeed = int(sys.argv[i+2])
			except:
				print "ERROR: random seed value must come after \"-r\""
				exit(0)

	if sys.argv[i+1] == "-n":
		if sys.argv[i+2] == 'a':
                        runName = sub.check_output("pwd").split("/")[-2]
                        print "autosetting runName seed to",runName,"based on directory"
                else:
			try:
				runName = sys.argv[i+2]
			except:
				print "ERROR: run name must come after \"-n\""
				exit(0)

	if sys.argv[i+1] == "-s":
		try:
			origSizeX = int(sys.argv[i+2])
			origSizeY = int(sys.argv[i+2])
			origSizeZ = int(sys.argv[i+2])
		except:
			print "ERROR: size of creature must come after \"-s\""
			exit(0)

	if sys.argv[i+1] == "-p":
		try:
			popSize = int(sys.argv[i+2])
		except:
			print "ERROR: population size of creature must come after \"-p\""
			exit(0)

	if sys.argv[i+1] == "--protect":
		try:
			protectInnovatonAlong = int(sys.argv[i+2])
			print "reset age of network", protectInnovatonAlong,"after each change"
		except:
			print "ERROR: Axis of innovation protection must come after \"--protect\""
			exit(0)

	if sys.argv[i+1] == "--nested":
		try:
			nestedAgeInterval = int(sys.argv[i+2])
			print "nested age interval set to:", nestedAgeInterval
		except:
			print "ERROR: Axis of innovation protection must come after \"--protect\""
			exit(0)

	if sys.argv[i+1] == "--mutateBodyProb":
		try:
			mutateBodyProb = sys.argv[i+2]
			if mutateBodyProb == "-a":
				# print sub.check_output("pwd")
				mutateBodyProb = float(sub.check_output("pwd").split("/")[-2].split("--mutBody_")[1])
			else:
				mutateBodyProb = float(sys.argv[i+2])
			# print "mutateBodyProb =", mutateBodyProb
		except:
			print "ERROR: mutateBodyProb must come after \"--mutateBodyProb\""
			exit(0)

	if sys.argv[i+1] == "--seedIndividual":
		try:
			seedIndividual = sys.argv[i+2]
		except:
			print "ERROR: mutateBodyProb must come after \"--mutateBodyProb\""
			exit(0)
		if not (len(glob.glob(seedIndividual[0:-5]+"0.txt")) == 1 and len(glob.glob(seedIndividual[0:-5]+"1.txt")) == 1):
			print "ERROR: seed individual not found!"
			exit(0)

	if sys.argv[i+1] == "--maxGens":
		try:
			maxGenerations = int(sys.argv[i+2])
		except:
			print "ERROR: maxGenerations must come after \"--maxGens\""
			exit(0)
		

if swarmClimb:
	origSizeY *= 2

# if "-f" in sys.argv[1:]:
# 	# origSizeX = 200
# 	# origSizeY = 200
# 	activationFunctionNames = ["sigmoid","sin","abs","square","sqrt"]#,"edge","gradient","erosion","dilation","opening","closing"]
	
# for arg in sys.argv[1:]:
# 	if "-m" in arg:
# 		mutationsPerStep = int(arg[2:])
# 		print "mutationsPerStep =",mutationsPerStep

for arg in sys.argv[1:]:
	if "-c" in arg:
		print "continueing from old run!"
		continuingRun = True

		try:
			numCPPNFolders = len(sub.check_output("ls -d cppn_gml/* | tail -2",shell=True).strip().split("\n"))
		except:
			numCPPNFolders = 0

		if numCPPNFolders == 0:
			print
			print "ERROR: must have seed population in cppn_gml/Gen_****.  exiting."
			exit(0)
		elif numCPPNFolders == 2:
			lastGenChecked = int(sub.check_output("ls -d cppn_gml/* | tail -1 | cut -d _ -f 3",shell=True)) - 1
			print "starting from second to last cppn_gml gen folder, Gen:",lastGenChecked
		elif numCPPNFolders == 1:
			lastGenChecked = int(sub.check_output("ls -d cppn_gml/* | tail -1 | cut -d _ -f 3",shell=True))
			print "only one cppn_gml folder avaliable."
			print "starting from cppn_gml gen folder, Gen:",lastGenChecked
			if lastGenChecked == 1:
				print "ERROR: Cannot restart at Gen 1.  exiting."

# --------------------------------------------------------------------------------------
# mutate old individual:
for i in range(len(sys.argv[1:])):
	if sys.argv[i+1] == "--mutate":
		origFileForMutation = sys.argv[i+2][0:-5]

				
#------------------------------------------------------------------------------------

# SET UP
blurMatrix = 1.0/16*np.array([[[1,2,1],
							   [2,4,2],
							   [1,2,1]]])

edgeMatrix = np.array([[[-1,-1,-1],
						[-1, 8,-1],
						[-1,-1,-1]]])

filterSize = (int(min(origSizeX,filterRadius)),int(min(origSizeY,filterRadius)),int(min(origSizeZ,filterRadius)))
#------------------------------------------------------------------------------------

def printName(var):
	print var+":",eval(var)

def mainNeuralNet():

	# mutationFunctions = [resetNetwork,removeNode,removeLink,addNode,addLink,mutateWeight,mutFunct]
	mutationFunctions = [removeNode,removeLink,addNode,addLink,mutateWeight,mutFunct]

	# ---------------------------------------------------------------------------
	# intialize record keeping
	startAll = time.time()
	random.seed(randomSeed)
	global maxID
	global maxGenerations
	global totalEvaluations
	totalEvaluations = 0
	global gen
	gen = 0
	global alreadyEvaluated
	alreadyEvaluated = {}
	global alreadyEvaluatedShape
	alreadyEvaluatedShape = {}
	global bestDistOnlySoFar
	bestDistOnlySoFar = -99999
	global bestEnergyOnlySoFar
	bestEnergyOnlySoFar = -99999
	global bestHeightOnlySoFar
	bestHeightOnlySoFar = -99999
	global bestFitOnlySoFar
	bestFitOnlySoFar = -99999
	global bestObj1SoFar
	bestObj1SoFar = -99999
	global bestHeightSoFar
	bestHeightSoFar = -99999


	# ----------------------------------------------------------------------------------------
	# PRINT RUN PARAMS:
	print "#####################################################################################"
	print "RUN PARAMETERS:"
	print 
	printName("maxGenerations")
	printName("popSize")
	printName("mutationStd")
	printName("origSizeX")
	printName("origSizeY")
	printName("origSizeZ")
	printName("activationFunctionNames")
	printName("scalingFactor")
	printName("filterRadius")
	printName("mutationChances")
	printName("proportionNewlyGenerated")
	printName("proportionFromCrossover")
	printName("proportionFromMutation")
	printName("numRandomNodes")
	printName("numRandomLinkAdds")
	printName("numRandomWeightChanges")
	printName("numRandomLinkRemovals")
	printName("numRandomActivationFunction")
	printName("minimumFitness")
	print
	printName("inputNodeNames")
	printName("outputNodeNames")
	printName("protectInnovatonAlong")
	printName("mutateBodyProb")	
	print
	printName("runName")
	printName("randomSeed")
	printName("continuingRun")
	printName("origFileForMutation")
	printName("minVoxelPercentMutated")
	printName("minVoxelDiffForAgeReset")
	print
	printName("floorSlope")
	printName("inCage")
	printName("swarmClimb")
	printName("apertureProportion")
	printName("fitnessEvaluationCycles")
	printName("fitnessEvalInitTime")
	# printName("actuationsPerSecond")
	printName("minPercentFull")
	printName("minPercentMuscle")
	printName("saveVxaEvery")
	printName("saveAllIndividualData")
	printName("saveCPPNs")
	printName("SelfCollisionsEnabled")
	printName("softestMaterial")
	print 
	print "#####################################################################################"

	# ----------------------------------------------------------------------------------------

	sub.call("mkdir voxelyzeFiles",shell=True)
	sub.call("rm -f voxelyzeFiles/*.vxa",shell=True)
	sub.call("mkdir fitnessFiles",shell=True)
	sub.call("mkdir bestSoFar",shell=True)
	sub.call("mkdir bestSoFar/distOnly",shell=True)
	sub.call("mkdir bestSoFar/energyOnly",shell=True)
	sub.call("mkdir bestSoFar/heightOnly",shell=True)
	sub.call("mkdir bestSoFar/fitOnly",shell=True)

	# --------------------------------------------------------------------------------------
	# mutate old individual:

	if origFileForMutation != "none":

		G0 = nx.read_gml(origFileForMutation+"0.txt")
		G0 = nx.relabel_nodes(G0,dict((n,d['label']) for n,d in G0.nodes(data=True)))#,copy=False)
		G1 = nx.read_gml(origFileForMutation+"1.txt")
		G1 = nx.relabel_nodes(G1,dict((n,d['label']) for n,d in G1.nodes(data=True)))#,copy=False)

		individual = [G0,G1]

		if len(glob.glob("mutantsOf_id_"+str(individual[0].graph["id"])+"/mutantOf--parentID_%04i--randSeed_%02i*"%(individual[0].graph["id"],randomSeed)))>0:
			print "random seed",randomSeed,"already evaluated.  SKIPPING."
			exit(0)

		createPhenotype(individual)	
		# evaluateAll([individual])
		# print sub.check_output("ls fitnessFiles")
		# print individual[0].graph["materialDistribution"]

		materialCounts = np.zeros((5))
		shapeMatrixOld = np.zeros((origSizeX,origSizeY,origSizeZ))
		makeOneShapeOnly(individual)
		for z in range(origSizeZ):
			for y in range(origSizeY):
				for x in range(origSizeX):
					if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
						shapeMatrixOld[x,y,z] = 0
						materialCounts[0] += 1
					elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
							shapeMatrixOld[x,y,z] = 2
							materialCounts[2] += 1
					elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
						# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
							shapeMatrixOld[x,y,z] = 3
							materialCounts[3] += 1
						# else:
						# 	shapeMatrixOld[x,y,z] = 4
						# 	materialCounts[4] += 1		
					else:
						shapeMatrixOld[x,y,z] = 1
						materialCounts[1] += 1

		
		oldMaterialDistribution = materialCounts	
		oldFitness = individual[0].graph["fitness"]
		oldDistance = individual[0].graph["distance"]
		oldHeight = individual[0].graph["height"]

		networkNum = 1*(random.random() > mutateBodyProb)
		print "mutating network",networkNum

		for outputNode in outputNodeNames[networkNum]:
			individual[networkNum].node[outputNode]["oldStatePostHocMutation"] = individual[networkNum].node[outputNode]["state"]
		done = False
		mutationCounter = 0

		oldIndividual = copy.deepcopy(individual)

		# print "networkNum:",networkNum

		while not done:
			mutationCounter += 1
			randomNum = random.random()
			randomProbSum = 0

			individual = copy.deepcopy(oldIndividual)

			for i in range(len(mutationChances)):
				randomProbSum += mutationChances[i]
				if randomNum < randomProbSum:
					variationDegree = mutationFunctions[i](individual[networkNum],networkNum)
					# print mutationFunctions[i].__name__,"on network",networkNum,"of id",individual[0].graph["id"]
					individual[0].graph["variationType"] = mutationFunctions[i].__name__ + variationDegree
					# print individual[0].graph["variationType"]
					break

			pruneNetwork(individual,networkNum)
			createPhenotype(individual)

			materialCounts = np.zeros((5))
			shapeMatrixNew = np.zeros((origSizeX,origSizeY,origSizeZ))
			makeOneShapeOnly(individual)
			for z in range(origSizeZ):
				for y in range(origSizeY):
					for x in range(origSizeX):
						if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
							shapeMatrixNew[x,y,z] = 0
							materialCounts[0] += 1
						elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
							shapeMatrixNew[x,y,z] = 2
							materialCounts[2] += 1
						elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
							# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
								shapeMatrixNew[x,y,z] = 3
								materialCounts[3] += 1
							# else:
							# 	shapeMatrixNew[x,y,z] = 4
							# 	materialCounts[4] += 1
						else:
							shapeMatrixNew[x,y,z] = 1
							materialCounts[1] += 1

			for outputNode in outputNodeNames[networkNum]:
				if outputNode == "phaseOffset":
					if individual[networkNum].node[outputNode]["oldStatePostHocMutation"][shapeMatrixNew>0] != individual[networkNum].node[outputNode]["state"][shapeMatrixNew>0]:
						done = True
				else:
					if np.sum(1*((shapeMatrixNew-shapeMatrixOld)!=0)) >= minVoxelPercentMutated*np.sum(oldMaterialDistribution[1:]) and np.sum(materialCounts[1:]) >= minPercentFull*origSizeX*origSizeY*origSizeZ and np.sum(materialCounts[3:]) >= minPercentMuscle*origSizeX*origSizeY*origSizeZ:
						done = True
				
		evaluateAll([individual])	

		shapeDiff = np.sum(np.abs((shapeMatrixOld>0)*1 - (shapeMatrixNew>0)*1))
		totalDiff = np.sum(((shapeMatrixOld - shapeMatrixNew) != 0)*1)

		newMaterialDistribution = individual[0].graph["materialDistribution"]
		newFitness = individual[0].graph["fitness"]
		newDistance = individual[0].graph["distance"]
		newHeight = individual[0].graph["height"]


		print
		print individual[0].graph["variationType"]
		print "fitness:",oldFitness,"->",newFitness
		print "distance:",oldDistance,"->",newDistance
		print "height:",oldHeight,"->",newHeight
		print "materialDistribution:",oldMaterialDistribution,"->",newMaterialDistribution
		print "shapeDiff:",shapeDiff
		print "totalDiff:",totalDiff

		sub.call("mkdir mutantsOf_id_"+str(individual[0].graph["id"]),shell=True)
		# sub.call("mv voxelyzeFiles/*")
		sub.call("mv voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" mutantsOf_id_"+str(individual[0].graph["id"])+"/pID_%04i--seed_%02i--shapeDiff_%04i--totalDiff_%04i--origSize_%04i--mutCount_%03i--fit_%.04f--oldFit_%.04f--dist_%.04f--height_%04f--mutNet_%01i--varType_"%(individual[0].graph["id"],randomSeed,shapeDiff,totalDiff,np.sum(oldMaterialDistribution[1:]),mutationCounter,individual[0].graph["fitness"],oldFitness,individual[0].graph["distance"],individual[0].graph["height"],networkNum)+individual[0].graph["variationType"]+".vxa",shell=True)

		# output cppn files:
		if saveCPPNs:
			tmpInd = copy.deepcopy(individual)
			for networkNum in range(2):

				# REMOVE STATE INFORMATOIN TO REDUCE FILE SIZE
				for nodeName in tmpInd[networkNum].nodes():
					tmpInd[networkNum].node[nodeName]["state"] = None
					tmpInd[networkNum].node[nodeName]["evaluated"] = 0
					if "oneShapeOnly" in tmpInd[networkNum].node[nodeName]:
						tmpInd[networkNum].node[nodeName]["oneShapeOnly"] = None
				if "dominatedBy" in tmpInd[networkNum].graph:
					tmpInd[networkNum].graph["dominatedBy"] = None
				if "materialDistribution" in tmpInd[networkNum].graph:
					tmpInd[networkNum].graph["materialDistribution"] = None
				
				nx.write_gml(tmpInd[networkNum],"mutantsOf_id_"+str(individual[0].graph["id"])+"/cppn_pID_%04i--seed_%02i--shapeDiff_%04i--totalDiff_%04i--origSize_%04i--mutCount_%03i--fit_%.04f--oldFit_%.04f--dist_%.04f--height_%04f--mutNet_%01i--varType_"%(individual[0].graph["id"],randomSeed,shapeDiff,totalDiff,np.sum(oldMaterialDistribution[1:]),mutationCounter,individual[0].graph["fitness"],oldFitness,individual[0].graph["distance"],individual[0].graph["height"],networkNum)+individual[0].graph["variationType"]+".txt")

		exit(0)	
	#------------------------------------------------------------------------------------


	if not continuingRun:
		sub.call("rm -f bestSoFar/*.vxa",shell=True)
		sub.call("rm -f bestSoFar/distOnly/*.vxa",shell=True)
		sub.call("rm -f bestSoFar/energyOnly/*.vxa",shell=True)
		sub.call("rm -f bestSoFar/heightOnly/*.vxa",shell=True)
		sub.call("rm -f bestSoFar/fitOnly/*.vxa",shell=True)
		sub.call("rm -rf Gen_*",shell=True)

		champFile = open("bestSoFar/bestOfGen.txt",'w')
		champFile.write("gen\t\tfitness\t\tdistance\t\theight\t\tage\n")
		champFile.write("-------------------------------------------------------------------------------\n")
		champFile.close()

		if saveAllIndividualData:
			sub.call("mkdir allIndividualsData",shell=True)
			sub.call("rm -f allIndividualsData/*",shell=True)

		if saveCPPNs:
			sub.call("mkdir cppn_gml",shell=True)
			sub.call("rm -rf cppn_gml/*",shell=True)


		# ---------------------------------------------------------------------------------
		# INIT POPULATION
		gen = 0
		
		totalEvaluations = 0
		alreadyEvaluated = {}

		if saveVxaEvery > 0:
			sub.call("mkdir Gen_%04i"%gen,shell=True)

		population = []
		counter = 0
		if seedIndividual == None:
			# for n in range(popSize):
			while len(population) < popSize:
				# tmpPop = []
				# for i in range(popSize-len(population)):
				individual = [createRandomNetwork(0),createRandomNetwork(1)]
				individual[0].graph["id"] = maxID
				individual[0].graph["parentID1"] = -1
				individual[0].graph["parentID2"] = -1
				individual[0].graph["parentMd5Shape1"] = "none"
				individual[0].graph["parentDist1"] = -1
				individual[0].graph["parentDist2"] = -1
				individual[0].graph["parentFit1"] = -1
				individual[0].graph["parentFit2"] = -1
				individual[0].graph["age"] = 0
				individual[0].graph["trueAge"] = 0
				individual[0].graph["networkNum"] = -1
				individual[0].graph["md5"] = None
				individual[0].graph["variationType"] = "newlyGenerated"
				maxID += 1

				createPhenotype(individual)
				
				shapeMatrixNew = np.zeros((origSizeX,origSizeY,origSizeZ))
				makeOneShapeOnly(individual)
				for z in range(origSizeZ):
					for y in range(origSizeY):
						for x in range(origSizeX):
							if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
								shapeMatrixNew[x,y,z] = 0
							elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
								shapeMatrixNew[x,y,z] = 2
							elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
								# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
									shapeMatrixNew[x,y,z] = 3
								# else:
								# 	shapeMatrixNew[x,y,z] = 4
							else:
								shapeMatrixNew[x,y,z] = 1

				if np.sum(shapeMatrixNew>0) >= minPercentFull*origSizeX*origSizeY*origSizeZ and np.sum(shapeMatrixNew>2) >= minPercentMuscle*origSizeX*origSizeY*origSizeZ:
					population.append(individual)
				counter += 1

			print popSize,"viable individuals found in",counter,"attempts"

			# ------------------------------------------------------------------------------------
			# if using random controllers or mophology, mutate the other x number of times for "complexification":

			# ------------------------------------------------------------------------------------

			evaluateAll(population)

			# for individual in tmpPop:
			# 	if individual[0].graph["fitness"] > minimumFitness:
			# 		population.append(individual)
			# 	else:
			# 		print "individual not viable.  Throwing out and creating another random individual."

			# print "initial population fully filled out!"
			# print					
		else:
			print "starting from seed individual."
			# starting from single seed individual:
			G0 = nx.read_gml(seedIndividual[0:-5]+"0.txt")
			G0 = nx.relabel_nodes(G0,dict((n,d['label']) for n,d in G0.nodes(data=True)))#,copy=False)
			G1 = nx.read_gml(seedIndividual[0:-5]+"1.txt")
			G1 = nx.relabel_nodes(G1,dict((n,d['label']) for n,d in G1.nodes(data=True)))#,copy=False)

			for n in range(popSize):
				population.append(copy.deepcopy([G0,G1]))

			# eval one to catch values for initial population evals
			individual = population[0]
			createPhenotype(individual)
			evaluateAll([individual])

		# ------------------------------------------------------------------------------
		# EVALUATE FITNESS OF INITIAL POPULATION

		# print
		# for individual in population:
		# 	createPhenotype(individual)

		# evaluateAll(population)

	# --------------------------------------------------------------------------
	# if continuing from old run
	else:
		champFile = open("bestSoFar/bestOfGen.txt",'a')
		if saveAllIndividualData:
			sub.call("mkdir allIndividualsData",shell=True)
		if saveCPPNs:
			sub.call("mkdir cppn_gml",shell=True)

		totalEvaluations = 0
		alreadyEvaluated = {}

		gen = lastGenChecked
		maxGenerations += lastGenChecked

		population = []
		cppnFileNames = sub.check_output("ls -d cppn_gml/Gen_%04i/*.txt"%lastGenChecked,shell=True).strip().split("\n")
		if len(cppnFileNames)%2!=0:
			print "Error, must be even number of cppn files. exiting"
			exit(0)
		for i in range(0,len(cppnFileNames),2):
			# print cppnFileNames[i]
			# # nx.read_gml(cppnFileNames[i])
			# print cppnFileNames[i+1]
			# # nx.read_gml(cppnFileNames[i+1])
			if int(cppnFileNames[i].split("--")[2].split("_")[1]) > maxID: maxID = int(cppnFileNames[i].split("--")[2].split("_")[1]) + 1
			# exit(0)
			G0 = nx.read_gml(cppnFileNames[i])
			G0 = nx.relabel_nodes(G0,dict((n,d['label']) for n,d in G0.nodes(data=True)))#,copy=False)
			G1 = nx.read_gml(cppnFileNames[i+1])
			G1 = nx.relabel_nodes(G1,dict((n,d['label']) for n,d in G1.nodes(data=True)))#,copy=False)
			# print
			# print "before relabel"
			# # print "\n".join((nx.generate_gml(G0)))
			# print G0.edge

			if cppnFileNames[i].split("--")[2] == cppnFileNames[i+1].split("--")[2] and cppnFileNames[i].split("--")[3] < cppnFileNames[i+1].split("--")[3]:
				population.append([G0,G1])
			else:
				print "ERROR: net0 and net1 did not match up in loading of population cppns"

	# ---------------------------------------------------------------------------------
	# ITERATE THROUGH EVOLUTION
	print "running evolution from gen",gen,"to gen",maxGenerations
	print
	while gen < maxGenerations:
		gen += 1	
		print
		print "### GENERATION",gen,"###"
		print
		if gen%saveVxaEvery == 0 and saveVxaEvery > 0:
			sub.call("mkdir Gen_%04i"%gen,shell=True)
			if saveCPPNs:
				sub.call("mkdir cppn_gml/Gen_%04i"%gen,shell=True)

		# ----------------------------------------------------------------------------
		# Update ages
		for individual in population:
			individual[0].graph["age"]+=1
			individual[0].graph["trueAge"]+=1
			individual[0].graph["variationType"] = "survived"
			individual[0].graph["parentDist1"] = individual[0].graph["distance"]
			individual[0].graph["parentDist2"] = -1
			individual[0].graph["parentFit1"] = individual[0].graph["fitness"]
			individual[0].graph["parentFit2"] = -1
			individual[0].graph["parent1Height"] = individual[0].graph["height"]
			individual[0].graph["parent2Height"] = -1
			individual[0].graph["parentID1"] = individual[0].graph["id"]
			individual[0].graph["parentMd5Shape1"] = individual[0].graph["md5Shape"]
			individual[0].graph["parentMd5Control1"] = individual[0].graph["md5Control"]
			individual[0].graph["parentFrequency"] = individual[0].graph["frequency"]
			individual[0].graph["parentID2"] = -1
			individual[0].graph["networkNum"] = -1
			individual[0].graph["voxelDiff"] = 0
			individual[0].graph["shapeDiff"] = 0
			individual[0].graph["controlDiff"] = 0

		# # ------------------------------------------------------------------------------
		newChildren = []
		spotsToFill = popSize # - len(population)


		# # PERFORM CROSSOVER
		# while len(newChildren)+1 < spotsToFill*(proportionFromCrossover) and len(population) > 1:

		# 	networkNum = 1*(random.random() > mutateBodyProb)

		# 	random.shuffle(population)
		# 	resultsFromCross = crossoverBoth(population[0],population[1],networkNum)

		# 	# for i in range(2):
		# 	for individual in resultsFromCross:
		# 		individual[0].graph["networkNum"] = networkNum
		# 		# if networkNum == protectInnovatonAlong: resultsFromCross[i][0].graph["age"] = 0

		# 	for individual in resultsFromCross:
		# 		pruneNetwork(individual,networkNum)
		# 		createPhenotype(individual)
		# 		newChildren.append(individual)

		# -------------------------------------------------------------------------------
		# PERFORM MUTATION

		indCounter = 0
		while len(newChildren) < spotsToFill*(proportionFromMutation+proportionFromCrossover) and len(population)!=0:
			random.shuffle(population)

			# individual = copy.deepcopy(population[0])

			individual = copy.deepcopy(population[indCounter]) # nac: for debug nested mutation
			indCounter += 1

			networkNum = 1*(random.random() > mutateBodyProb)

			if nestedAgeInterval > 0:
				# if gen%nestedAgeInterval==0:
				if gen%nestedAgeInterval==1:
					networkNum = 0
				else:
					networkNum = 1

			individual[0].graph["networkNum"] = networkNum
			print "networkNum:", networkNum
			# if networkNum == protectInnovatonAlong:	individual[0].graph["age"] = 0
			
			individual[0].graph["parentDist1"] = individual[0].graph["distance"]
			individual[0].graph["parentDist2"] = -1
			individual[0].graph["parentFit1"] = individual[0].graph["fitness"]
			individual[0].graph["parentFit2"] = -1
			individual[0].graph["parent1Height"] = individual[0].graph["height"]
			individual[0].graph["parent2Height"] = -1
			individual[0].graph["parentID1"] = individual[0].graph["id"]
			individual[0].graph["parentMd5Shape1"] = individual[0].graph["md5Shape"]
			individual[0].graph["parentMd5Control1"] = individual[0].graph["md5Control"]
			individual[0].graph["parentFrequency"] = individual[0].graph["frequency"]
			individual[0].graph["parentID2"] = -1
			individual[0].graph["id"] = maxID
			maxID += 1

			for thisNet in range(len(outputNodeNames)):
				for outputNode in outputNodeNames[thisNet]:
					individual[thisNet].node[outputNode]["oldState"] = individual[thisNet].node[outputNode]["state"]
			oldFrequency = individual[0].graph["frequency"]
			

			shapeMatrixOld = np.zeros((origSizeX,origSizeY,origSizeZ))
			fixedMatrix = np.zeros((origSizeX,origSizeY,origSizeZ))
			makeOneShapeOnly(individual)
			for z in range(origSizeZ):
				for y in range(origSizeY):
					for x in range(origSizeX):
						if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
							shapeMatrixOld[x,y,z] = 0
						elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
								shapeMatrixOld[x,y,z] = 2
						elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
							# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
								shapeMatrixOld[x,y,z] = 3
							# else:
							# 	shapeMatrixOld[x,y,z] = 4
						else:
							shapeMatrixOld[x,y,z] = 1

						if swarmClimb and y > int(origSizeY/2):
							fixedMatrix[x,y,z] = 1

			oldIndividual = copy.deepcopy(individual)
			done = False
			mutationCounter = 0
			while not done:
				# print "mutationChances:",mutationCounter
				mutationCounter+=1
				individual = copy.deepcopy(oldIndividual)
				randomNum = random.random()
				randomProbSum = 0

				for i in range(len(mutationChances)):
					randomProbSum += mutationChances[i]
					if randomNum < randomProbSum:
						variationDegree = mutationFunctions[i](individual[networkNum],networkNum)
						# print mutationFunctions[i].__name__,"on network",networkNum,"of id",individual[0].graph["id"]
						individual[0].graph["variationType"] = mutationFunctions[i].__name__ + variationDegree
						break

				pruneNetwork(individual,networkNum)
				createPhenotype(individual)

				shapeMatrixNew = np.zeros((origSizeX,origSizeY,origSizeZ))
				makeOneShapeOnly(individual)
				for z in range(origSizeZ):
					for y in range(origSizeY):
						for x in range(origSizeX):
							if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
								shapeMatrixNew[x,y,z] = 0
							elif swarmClimb and y>=int(origSizeY/2):
								shapeMatrixNew[x,y,z] = 5
							elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
								shapeMatrixNew[x,y,z] = 2
							elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
								# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
								shapeMatrixNew[x,y,z] = 3
								# else:
								# 	shapeMatrixNew[x,y,z] = 4
							else:
								shapeMatrixNew[x,y,z] = 1

				for outputNode in outputNodeNames[networkNum]:
					if outputNode in  ["phaseOffset"]:
						if np.sum(individual[networkNum].node[outputNode]["oldState"][shapeMatrixNew==3] != individual[networkNum].node[outputNode]["state"][shapeMatrixNew==3])>0:
							done = True
					if outputNode in  ["frequency"]:
						if abs(oldFrequency - individual[0].graph["frequency"]) > 0:
							done = True
					else:
						if np.sum(1*((shapeMatrixNew-shapeMatrixOld)!=0)) >= minVoxelPercentMutated*np.sum(shapeMatrixOld>0) and np.sum(shapeMatrixNew>0) >= minPercentFull*origSizeX*origSizeY*origSizeZ and np.sum(shapeMatrixNew==3) >= minPercentMuscle*origSizeX*origSizeY*origSizeZ:
							done = True

				if mutationCounter > 10000:
					print "couldn't find a successful mutation in 10000 tries!  Exiting mutate."
					done = True

				# for outputNode in outputNodeNames[networkNum]:
				# 	if outputNode == "phaseOffset":
				# 		if individual[networkNum].node[outputNode]["oldStatePostHocMutation"][individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"]]>0] != individual[networkNum].node[outputNode]["state"][individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"]]>0]:
				# 			done = True
				# 	else:
				# 		if np.sum(1*((shapeMatrixNew-shapeMatrixOld)!=0)) >= minVoxelPercentMutated*np.sum(oldMaterialDistribution[1:]) and np.sum(materialCounts[1:]) >= minPercentFull*origSizeX*origSizeY*origSizeZ and np.sum(materialCounts[3:]) >= minPercentMuscle*origSizeX*origSizeY*origSizeZ:
				# 			done = True

			individual[0].graph["voxelDiff"] = np.sum(1*((shapeMatrixNew-shapeMatrixOld)!=0))
			individual[0].graph["shapeDiff"] = np.sum(1*((1*(shapeMatrixNew>0)-1*(shapeMatrixOld>0))!=0))	
			individual[0].graph["controlDiff"] = np.sum( np.abs( individual[1].node["phaseOffset"]["state"][shapeMatrixNew>2] - individual[1].node["phaseOffset"]["oldState"][shapeMatrixNew>2] ) ) + (origSizeX*origSizeY*origSizeZ)* abs(oldFrequency - individual[0].graph["frequency"]) 
												 # np.sum( np.abs( individual[1].node["frequency"]["state"][shapeMatrixNew>2] - individual[1].node["frequency"]["oldState"][shapeMatrixNew>2] ) )
			# individual[0].graph["energy"] = float(np.sum(shapeMatrixNew>2))/np.sum(shapeMatrixNew>0)			
			# if individual[0].graph["shapeDiff"] > 0:
			# 	print "shapeMatixOld:"
			# 	print 1*(shapeMatrixOld>0)
			# 	print 
			# 	print "shapeMatixNew:"
			# 	print 1*(shapeMatrixNew>0)

			# individual[0].graph["height"] = np.where(shapeMatrixNew>0)

			# for outputNode in outputNodeNames[networkNum]:
			# 	individual[networkNum].node[outputNode]["oldState"] = None
			for thisNet in range(len(outputNodeNames)):
				for outputNode in outputNodeNames[thisNet]:
					individual[thisNet].node[outputNode]["oldState"] = None
			

			newChildren.append(individual)

		# ------------------------------------------------------------------------------------------
		# FILL IN REST WITH NEW RANDOM INDIVIDUALS

		# while len(newChildren) < popSize:
		# 	individual = [createRandomNetwork(0),createRandomNetwork(1)]
		# 	individual[0].graph["id"] = maxID
		# 	individual[0].graph["parentID1"] = -1
		# 	individual[0].graph["parentMd5Shape1"] = "none"
		# 	individual[0].graph["parentID2"] = -1
		# 	individual[0].graph["parentDist1"] = -1
		# 	individual[0].graph["parentDist2"] = -1
		# 	individual[0].graph["parentFit1"] = -1
		# 	individual[0].graph["parentFit2"] = -1
		# 	individual[0].graph["age"] = 0
		# 	individual[0].graph["trueAge"] = 0	
		# 	individual[0].graph["networkNum"] = -1
		# 	individual[0].graph["md5"] = None
		# 	individual[0].graph["variationType"] = "newlyGenerated"
		# 	maxID += 1
		# 	createPhenotype(individual)
		# 	newChildren.append(individual)

		# ------------------------------------------------------------------------------
		# EVALUATE FITNESS
		print
		numEvaluatedThisGen = 0
		startTime = time.time()

		numEvaluatedThisGen = evaluateAll(newChildren)
		
		endTime = time.time()
		print "All Voxelyze evals finished in",endTime-startTime, "seconds"
		print "numEvaluatedThisGen: "+str(numEvaluatedThisGen)+"/"+str(len(newChildren))
		print "totalEvaluations:",totalEvaluations

		# --------------------------------------------------------------------------
		# combine children and parents for selection
		population += newChildren

		# ---------------------------------------------------------------------------
		# PERFORM SELECTION
		# select surviors for new population

		newPopulation = []
		fitnessList = []

		# sort newest networks to the bottom, so they have an advantage in a tie (i.e. encourage neutral mutations).  
		population.sort(reverse = False, key = lambda individual: individual[0].graph["id"])

		# CALC "DOMINATED BY" FOR PARETO OPTIMIZATION
		for individual in population:
			individual[0].graph["dominatedBy"] = []
		
		for individual in population:
			G = individual[0]
			for otherIndividual in population:
				otherG = otherIndividual[0]

				# if (otherG.graph["distance"] >= G.graph["distance"] and otherG.graph["fitnessEnergy"] <= G.graph["fitnessEnergy"] and otherG.graph["age"] <= G.graph["age"])\
				# if (otherG.graph["distance"] >= G.graph["distance"] and otherG.graph["age"] <= G.graph["age"])\
				# if (otherG.graph["distance"] >= G.graph["distance"] and otherG.graph["age"] <= G.graph["age"] and otherG.graph["height"] <= G.graph["height"])\
				if (otherG.graph["fitness"] >= G.graph["fitness"] and otherG.graph["age"] <= G.graph["age"])\
				and not (G.graph["id"] in otherG.graph["dominatedBy"])\
				and not (otherG.graph["id"] == G.graph["id"]):
					G.graph["dominatedBy"] += [otherG.graph["id"]]

			# EXTRA PENALTY FOR DOING NOTHING OR BEING INVALID
			if G.graph["fitness"] == minimumFitness:# or G.graph["fitnessEnergy"] == 0:
				G.graph["dominatedBy"] += [G.graph["id"] for i in range(popSize*2)]

		population.sort(reverse = True, key = lambda individual: individual[0].graph["id"]) 				# sort age third
		population.sort(reverse = False, key = lambda individual: individual[0].graph["age"]) 				# sort age third
		population.sort(reverse = True,  key = lambda individual: individual[0].graph["height"])
		population.sort(reverse = True,  key = lambda individual: individual[0].graph["distance"]) 		# sort distance second
		population.sort(reverse = True,  key = lambda individual: individual[0].graph["fitness"]) 		
		population.sort(reverse = False, key = lambda individual: len(individual[0].graph["dominatedBy"])) 	# sort dominated by first


		# -------------------------------------------------------------------------
		# PERFORM SELECTION BY PARETO FRONTS
		
		if nestedAgeInterval < 0 or gen%nestedAgeInterval==0:
			done = False
			paretoLevel = 0
			while not done:
				thisLevel = []
				sizeLeft = popSize - len(newPopulation)
				for individual in population:
					if len(individual[0].graph["dominatedBy"]) == paretoLevel:
						thisLevel += [individual]

				# IF WHOLE PARETO LEVEL CAN FIT, ADD IT
				if len(thisLevel) > 0:
					if sizeLeft >= len(thisLevel):
						newPopulation += thisLevel
						
					# OTHERWISE, SELECT BY SORTED RANKING (ABOVE) (TODO: SELECT TO MAXIMIZE DIVERSITY):
					else:
						# TRUNCATE SELECTION BY DISTANCE
						# newPopulation += thisLevel[0:sizeLeft]

						# RANK PROPORTIONAL SELECTION BY DISTANCE
						newPopulation += [thisLevel[0]]
						while len(newPopulation) < popSize:
							randomNum = random.random()
							for i in range(1,len(thisLevel)):
								if randomNum >= math.log(i)/math.log(len(thisLevel)) and randomNum < math.log(i+1)/math.log(len(thisLevel)) and not thisLevel[i] in newPopulation:
									newPopulation += [thisLevel[i]]
									continue

				paretoLevel += 1		
				if len(newPopulation) == popSize:
					done = True

		else:
			print "selecting with morpholgy md5 only!"
			md5ShapeList = []
			for individual in population:
				md5ShapeList.append(individual[0].graph["md5Shape"])
			md5ShapeSet = set(md5ShapeList)
			
			for thisMd5Shape in md5ShapeSet:
				print "md5:",thisMd5Shape,"numPreviously:",int(md5ShapeList.count(thisMd5Shape)),"numNow:",int(md5ShapeList.count(thisMd5Shape)/2)
				tmpPop = []
				for individual in population:
					if individual[0].graph["md5Shape"] == thisMd5Shape:
						tmpPop.append(individual)
				tmpPop.sort(reverse = True, key = lambda individual: individual[0].graph["id"])
				tmpPop.sort(reverse = True,  key = lambda individual: individual[0].graph["fitness"])
				# tmpPop.sort(reverse = False, key = lambda individual: len(individual[0].graph["dominatedBy"]))
				newPopulation += tmpPop[:int(md5ShapeList.count(thisMd5Shape)/2)]
				# counter = 0
				# while counter < int(md5ShapeList.count(thisMd5Shape)/2):

				# 	coutner += 1

				

		# --------------------------------------------------------------------------------
		# PRINT POPULATION TO STDOUT AND SAVE ALL INDIVIDUAL DATA
		print 
		print "Gen "+str(gen)+":"
		# print "dom.\tfitness\t\t\tdistance\t\theight\t\theight^10/1000\tage\ttrue age\tvoxelDiff\tshapeDiff\tcontrolDiff\t\tenergy\t\tmd5Shape"
		print "dom.\tfitness\t\t\tdistance\tage\ttrue age\tvoxelDiff\tshapeDiff\tcontrolDiff\tfrequency\t\tmd5Shape"
		print "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
		for individual in population:
		# for individual in newPopulation:
			# print individual[0].graph["height"]
			# print str(len(individual[0].graph["dominatedBy"]))+"\t%9.5f\t"%(individual[0].graph["fitness"])+"\t%9.05f\t"%(individual[0].graph["distance"])+"\t%.05f\t\t"%(individual[0].graph["height"])+"%9.5f\t"%(individual[0].graph["height"]**10/1000)+str(individual[0].graph["age"])+"\t"+str(individual[0].graph["trueAge"])+"\t\t"+str(individual[0].graph["voxelDiff"])+"\t\t"+str(individual[0].graph["shapeDiff"])+"\t\t%9.5f"%(individual[0].graph["controlDiff"])+"\t\t"+str(individual[0].graph["energy"])+"\t\t"+str(individual[0].graph["md5Shape"])
			print str(len(individual[0].graph["dominatedBy"]))+"\t%9.5f\t"%(individual[0].graph["fitness"])+"\t%9.05f\t"%(individual[0].graph["distance"])+str(individual[0].graph["age"])+"\t"+str(individual[0].graph["trueAge"])+"\t\t"+str(individual[0].graph["voxelDiff"])+"\t\t"+str(individual[0].graph["shapeDiff"])+"\t\t%9.5f"%(individual[0].graph["controlDiff"])+"\t\t%9.5f"%(individual[0].graph["frequency"])+"\t\t"+str(individual[0].graph["md5Shape"])
		
		if saveAllIndividualData:
			recordingFile = open("allIndividualsData/Gen_%04i.txt"%gen,"w")
			recordingFile.write("id\tdom\tfitness\t\tdistance\t\theight\t\tenergy\t\tfrequency\t\tage\ttrueAge\tvoxelDiff\tShapeDiff\tempty\ttissue\tbone\tmuscle1\tmuscle2\tnodes\tedges\tP1_id\tP2_id\tP1_fit\t\tP2_fit\t\tP1_dist\t\tP2_dist\t\tvariationType\tnetNum\tmd5\t\t\tmd5Shape\t\t\tP1_md5Shape\t\t\tmd5Control\t\t\tP1_md5Control\t\t\tP1_freq\t\tselected\n")
			for individual in population:
				G = individual[0]
				recordingFile.write(str(individual[0].graph["id"])+
									"\t"+str(len(individual[0].graph["dominatedBy"]))+
									"\t%.08f"%(individual[0].graph["fitness"])+
									"\t%.08f"%(individual[0].graph["distance"])+
									"\t%.08f"%(individual[0].graph["height"])+
									"\t%.08f"%(individual[0].graph["energy"])+
									"\t%.08f"%(individual[0].graph["frequency"])+
									"\t"+str(individual[0].graph["age"])+
									"\t"+str(individual[0].graph["trueAge"])+
									"\t"+str(individual[0].graph["voxelDiff"])+
									"\t"+str(individual[0].graph["shapeDiff"])+
									"\t%4i"%(individual[0].graph["materialDistribution"][0])+
									"\t%4i"%(individual[0].graph["materialDistribution"][1])+
									"\t%4i"%(individual[0].graph["materialDistribution"][2])+
									"\t%4i"%(individual[0].graph["materialDistribution"][3])+
									"\t%4i"%(individual[0].graph["materialDistribution"][4])+
									"\t"+str(len(individual[0].nodes()))+
									"\t"+str(len(individual[0].edges()))+
									"\t"+str(individual[0].graph["parentID1"])+
									"\t\t"+str(individual[0].graph["parentID2"])+
									"\t\t%.08f"%(individual[0].graph["parentFit1"])+
									"\t%.08f"%(individual[0].graph["parentFit2"])+
									"\t\t%.08f"%(individual[0].graph["parentDist1"])+
									"\t%.08f"%(individual[0].graph["parentDist2"])+
									"\t"+individual[0].graph["variationType"]+
									"\t"+str(individual[0].graph["networkNum"])+
									"\t"+individual[0].graph["md5"]+
									"\t"+individual[0].graph["md5Shape"]+
									"\t"+individual[0].graph["parentMd5Shape1"]+
									"\t"+individual[0].graph["md5Control"]+
									"\t"+individual[0].graph["parentMd5Control1"]+
									"\t"+str(individual[0].graph["parentFrequency"])+
									"\t"+str(1*(individual in newPopulation))+
									"\n")

		champFile = open("bestSoFar/bestOfGen.txt",'a')
		champFile.write(str(gen)+"\t\t%.08f\t\t"%(population[0][0].graph["fitness"])+"\t\t%.08f\t\t"%(population[0][0].graph["distance"])+"\t\t%6f\t\t"%(population[0][0].graph["height"])+str(population[0][0].graph["age"])+"\t"+str(population[0][0].graph["trueAge"])+"\n")
		champFile.close()

		# --------------------------------------------------------------------------
		# SAVE CPPNS AS GMLS
		if gen%saveVxaEvery == 0 and saveVxaEvery > 0 and saveCPPNs:
			for individual in population:
				tmpInd = copy.deepcopy(individual)
				for networkNum in range(2):

					# REMOVE STATE INFORMATOIN TO REDUCE FILE SIZE
					for nodeName in tmpInd[networkNum].nodes():
						tmpInd[networkNum].node[nodeName]["state"] = None
						tmpInd[networkNum].node[nodeName]["evaluated"] = 0
						if "oneShapeOnly" in tmpInd[networkNum].node[nodeName]:
							tmpInd[networkNum].node[nodeName]["oneShapeOnly"] = None
					if "dominatedBy" in tmpInd[networkNum].graph:
						tmpInd[networkNum].graph["dominatedBy"] = None
					if "materialDistribution" in tmpInd[networkNum].graph:
						tmpInd[networkNum].graph["materialDistribution"] = None
					
					nx.write_gml(tmpInd[networkNum],"cppn_gml/Gen_%04i/cppn--fit_%.08f--dist_%.08f--height_%06f--id_%05i--net"%(gen,tmpInd[0].graph["fitness"],tmpInd[0].graph["distance"],tmpInd[0].graph["height"],tmpInd[0].graph["id"])+str(networkNum)+".txt")



		population = newPopulation		

	print 
	print "DONE!"
	print "finished "+str(maxGenerations)+ " generatons in "+str(time.time()-startAll)+" seconds ( = "+str((time.time()-startAll)/60)+" minutes = "+str((time.time()-startAll)/(3600))+" hours)"
	print


def evaluateAll(subPopulation,numEvaluatedThisGen=0):
	global gen
	global totalEvaluations
	global bestDistOnlySoFar
	global bestEnergyOnlySoFar
	global bestHeightOnlySoFar
	global bestFitOnlySoFar
	global bestObj1SoFar
	global bestHeightSoFar

	for individual in subPopulation: 				
		individual[0].graph["distance"] = -999
		individual[0].graph["fitness"] = -999
		individual[0].graph["height"] = -999
		individual[0].graph["energy"] = -999

	numEvaluatedThisGen = 0
	for individual in subPopulation:
		[materialCounts,md5,md5Shape,md5Control] = writeVoxelyzeFileOrig(individual)

		# SET AGE TO ZERO FOR MORPHOLOGY PROTECTION ONLY IF MORPHOLOGY ACTUALLY CHANGED
		if individual[0].graph["networkNum"] == protectInnovatonAlong:
			if protectInnovatonAlong == 0:
				if md5Shape not in alreadyEvaluatedShape: 
					# print "RESETTING AGE OF INDIVIDUAL",individual[0].graph["id"],"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
					if individual[0].graph["voxelDiff"] >= minVoxelDiffForAgeReset*origSizeX*origSizeY*origSizeZ:
						individual[0].graph["age"] = 0
				else:
					if alreadyEvaluatedShape[md5Shape] > individual[0].graph["age"]:
						individual[0].graph["age"] = alreadyEvaluatedShape[md5Shape]
			else:
				individual[0].graph["age"] = 0

		alreadyEvaluatedShape[md5Shape] = individual[0].graph["age"]

		individual[0].graph["md5Control"] = md5Control
		individual[0].graph["md5Shape"] = md5Shape
		individual[0].graph["md5"] = md5
		individual[0].graph["materialDistribution"] = materialCounts

		# DON'T EVALUATE IF HAS NO VOXELS OR NO MUSCLES
		if sum(materialCounts[1:]) < minPercentFull*origSizeX*origSizeY*origSizeZ or sum(materialCounts[3:]) < minPercentMuscle*origSizeX*origSizeY*origSizeZ:
			# print "not enough voxels filled:  assigning fitness of 10^-12 \n"
			individual[0].graph["distance"] = 0.0
			individual[0].graph["fitness"] = minimumFitness
			individual[0].graph["height"] = 0.0
			individual[0].graph["energy"] = 0.0
			# print "NOT ENOUGH VOXELS OR MUSCLES, assigning minimum fitness"
			if gen%saveVxaEvery == 0 and saveVxaEvery > 0:
				sub.call("mv voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" Gen_%04i/"%(gen)+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["id"]),shell=True)
			else:
				sub.call("rm voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"],shell=True)
		
		# DON'T EVALUATE IF YOU'VE ALREADY EVALUATED THE SAME PHENOTYPE BEFORE
		elif md5 in alreadyEvaluated:
			individual[0].graph["distance"] = alreadyEvaluated[md5][0]
			individual[0].graph["height"] = alreadyEvaluated[md5][1]
			individual[0].graph["fitness"] = alreadyEvaluated[md5][2]
			individual[0].graph["energy"] = alreadyEvaluated[md5][3]
			print "individual already evaluated:  cached fitness is",individual[0].graph["fitness"]," (",individual[0].graph["distance"],",",individual[0].graph["height"],")"
			if gen%saveVxaEvery == 0 and saveVxaEvery > 0:
				sub.call("mv voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" Gen_%04i/"%(gen)+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["id"]),shell=True)
			else:
				sub.call("rm voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"],shell=True)

		# EVALUATE WITH VOXELYZE
		else:
			# heightSum = 0
			# voxelCounter = 0
			# createPhenotype(individual)
			# makeOneShapeOnly(individual)
			# for z in range(origSizeZ):
			# 	for y in range(origSizeY):
			# 		for x in range(origSizeX):
			# 			if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] > 0:
			# 				heightSum += z+0.5
			# 				voxelCounter += 1
			# 			# else:
			# 				# print "absent voxel at "

			# individual[0].graph["height"] = float(heightSum)/voxelCounter

			shapeMatrixNew = np.zeros((origSizeX,origSizeY,origSizeZ))
			makeOneShapeOnly(individual)
			for z in range(origSizeZ):
				for y in range(origSizeY):
					for x in range(origSizeX):
						if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
							shapeMatrixNew[x,y,z] = 0
						elif swarmClimb and y>=int(origSizeY/2):
							shapeMatrixNew[x,y,z] = 5
						elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
							shapeMatrixNew[x,y,z] = 2
						elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
							# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
								shapeMatrixNew[x,y,z] = 3
							# else:
							# 	shapeMatrixNew[x,y,z] = 4
						else:
							shapeMatrixNew[x,y,z] = 1

			# individual[0].graph["energy"] = float(np.sum(shapeMatrixNew>2))/(origSizeX*origSizeY*origSizeZ)#/np.sum(shapeMatrixNew>0)
			individual[0].graph["energy"] = float(np.sum(shapeMatrixNew>0))/(origSizeX*origSizeY*origSizeZ)#/np.sum(shapeMatrixNew>0)

			numEvaluatedThisGen += 1
			totalEvaluations += 1
			# print "Staring Voxelyze..."
			sub.Popen("./voxelyze -f voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"],shell=True)

	print "launched",numEvaluatedThisGen,"voxelyze calls, out of",len(subPopulation),"individuals"

			
	numEvalsFinished = 0
	allDone = False
	while not allDone:
		# CHECK TO SEE IF ALL ARE FINISHED:
		allDone = True
		for individual in subPopulation: 				
			if individual[0].graph["distance"] == -999:
				allDone=False

		# CHECK FOR ANY FITNESS FILES THAT ARE PRESENT
		lsCheck = sub.check_output(["ls","fitnessFiles/"])
		if lsCheck:
			lsCheck = lsCheck.split()[0]
			if "softbotsOutput--id_" in lsCheck:
				thisId = int(lsCheck.split("_")[1].split(".")[0])
				(thisFitness,thisDistance,thisHeight) = readSoftbotFitnessFile("fitnessFiles/"+lsCheck)
				# thisFitness = thisDistance/height
				sub.call("rm fitnessFiles/"+lsCheck,shell=True)
				numEvalsFinished += 1
				print lsCheck,"fit = %0.5f"%(thisDistance),"  (",numEvalsFinished,"/",numEvaluatedThisGen,")"

				# ASSIGN THE VALUE IN THEM TO THE CORRESPONDING INDIVIDUAL
				for individual in subPopulation:
					if individual[0].graph["id"] == thisId:
						individual[0].graph["distance"] = thisDistance
						# individual[0].graph["fitness"] = thisFitness
						individual[0].graph["height"] = thisHeight	
						individual[0].graph["fitness"] = thisDistance #* thisHeight**10 / 1000 * (1/individual[0].graph["energy"])
						alreadyEvaluated[md5] = (individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["fitness"],individual[0].graph["energy"])

						# UPDATE THE RUN STATISTICS AND FILE MANAGEMENT
						if individual[0].graph["distance"] > bestDistOnlySoFar:
							bestDistOnlySoFar = individual[0].graph["distance"]
							sub.call("cp voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" bestSoFar/distOnly/"+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--freq_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["frequency"],individual[0].graph["id"]),shell=True)
						if individual[0].graph["energy"] > bestEnergyOnlySoFar:
							bestEnergyOnlySoFar = individual[0].graph["energy"]
							sub.call("cp voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" bestSoFar/energyOnly/"+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--freq_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["frequency"],individual[0].graph["id"]),shell=True)
						if individual[0].graph["height"] > bestHeightOnlySoFar:
							bestHeightOnlySoFar = individual[0].graph["height"]
							sub.call("cp voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" bestSoFar/heightOnly/"+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--freq_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["frequency"],individual[0].graph["id"]),shell=True)
						if individual[0].graph["fitness"] > bestFitOnlySoFar:
							bestFitOnlySoFar = individual[0].graph["fitness"]
							sub.call("cp voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" bestSoFar/fitOnly/"+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--freq_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["frequency"],individual[0].graph["id"]),shell=True)
						if individual[0].graph["distance"] > bestObj1SoFar and individual[0].graph["height"] < bestHeightSoFar:
							sub.call("cp voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" bestSoFar/"+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--freq_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["frequency"],individual[0].graph["id"]),shell=True)
							bestObj1SoFar = individual[0].graph["distance"]
							bestHeightSoFar = individual[0].graph["height"]
						if gen%saveVxaEvery == 0 and saveVxaEvery > 0:
							sub.call("mv voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"]+" Gen_%04i/"%(gen)+runName+"--Gen_%04i--fit_%.08f--dist_%.08f--height_%06f--freq_%06f--id_%05i.vxa"%(gen,individual[0].graph["fitness"],individual[0].graph["distance"],individual[0].graph["height"],individual[0].graph["frequency"],individual[0].graph["id"]),shell=True)
						else:
							sub.call("rm voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"],shell=True)
						break

			# WAIT A SEC AND TRY AGAIN		
			else:
				time.sleep(1)
		else:
			time.sleep(1)
				
	return numEvaluatedThisGen

def createRandomNetwork(networkNum):

	G = createMinimalNetwork(networkNum)

	for i in range(numRandomNodes):
		addNode(G,networkNum)

	for i in range(numRandomLinkAdds):
		addLink(G,networkNum)

	for i in range(numRandomWeightChanges):
		mutateWeight(G,networkNum)

	for i in range(numRandomLinkRemovals):
		removeLink(G,networkNum)

	for i in range(numRandomActivationFunction):
		mutFunct(G,networkNum)

	# pruneNetwork(G,networkNum)

	# createPhenotype(G)
	# print "BIAS EDGES:"
	# for thisEdge in G.out_edges(nbunch=["b"]):
	# 	# print thisEdge,G.edge[thisEdge[0]][thisEdge[1]]["weight"]
	# 	G.edge[thisEdge[0]][thisEdge[1]]["weight"] = 0.0

	return G

def createPhenotype(individual,networkNum = -1,sizeX=origSizeX,sizeY=origSizeY,sizeZ=origSizeZ):
	clearNodeStates(individual,networkNum)
	setInputNodeState(individual)
	for networkNum in range(len(outputNodeNames)):
		for outputNode in outputNodeNames[networkNum]:
			individual[networkNum].node[outputNode]["state"] = np.zeros((sizeX,sizeY,sizeZ))
			individual[networkNum].node[outputNode]["state"] = calcNodeState(individual[networkNum],outputNode,sizeX,sizeY,sizeZ)
	individual[0].graph["frequency"] = 7.5+5.0*max(-0.5,min(0.5,np.mean(individual[1].node["frequency"]["state"])))
	# print "frequency output mean:",np.mean(individual[1].node["frequency"]["state"])

def clearNodeStates(individual,networkNum = -1):
	if networkNum >= 0:
		for thisNode in individual[networkNum].nodes():
			individual[networkNum].node[thisNode]["evaluated"]=False
	else:
		for networkNum in range(len(outputNodeNames)):
			for thisNode in individual[networkNum].nodes():
				individual[networkNum].node[thisNode]["evaluated"]=False


def setInputNodeState(individual,sizeX=origSizeX,sizeY=origSizeY,sizeZ=origSizeZ):
	for G in individual:
		inputX = np.zeros((sizeX,sizeY,sizeZ))
		inputY = np.zeros((sizeX,sizeY,sizeZ))
		inputZ = np.zeros((sizeX,sizeY,sizeZ))
		inputD = np.zeros((sizeX,sizeY,sizeZ))
		inputB = np.ones((sizeX,sizeY,sizeZ))

		for x in range(sizeX):
			for y in range(sizeY):
				for z in range(sizeZ):
					# inputX[x,y,z] = ( 2*float( x - sizeX )/sizeX + 1 + 1.0/sizeX) * scalingFactor * (1+1.0/sizeX)
					# inputY[x,y,z] = ( 2*float( y - sizeY )/sizeY + 1 + 1.0/sizeY) * scalingFactor * (1+1.0/sizeY)
					# inputZ[x,y,z] = ( 2*float( z - sizeZ )/sizeZ + 1 + 1.0/sizeZ) * scalingFactor * (1+1.0/sizeZ)
					inputX[x,y,z] = x
					inputY[x,y,z] = y
					inputZ[x,y,z] = z
					
		inputX = normalizeMatrix(inputX)
		inputY = normalizeMatrix(inputY)
		inputZ = normalizeMatrix(inputZ)

		inputD = normalizeMatrix( pow( pow(inputX, 2) + pow(inputY, 2) + pow(inputZ, 2), 0.5 ) ) 

		for thisNode in G.nodes():
			if thisNode == "x":
				G.node[thisNode]["state"]=inputX
				G.node[thisNode]["evaluated"]=True
			if thisNode == "y":
				G.node[thisNode]["state"]=inputY
				G.node[thisNode]["evaluated"]=True
			if thisNode == "z":
				G.node[thisNode]["state"]=inputZ
				G.node[thisNode]["evaluated"]=True
			if thisNode == "d":
				G.node[thisNode]["state"]=inputD
				G.node[thisNode]["evaluated"]=True
			if thisNode == "b":
				G.node[thisNode]["state"]=inputB
				G.node[thisNode]["evaluated"]=True

def calcNodeState(G,thisNode,sizeX=origSizeX,sizeY=origSizeY,sizeZ=origSizeZ):

	if G.node[thisNode]["evaluated"]:
		return G.node[thisNode]["state"]

	G.node[thisNode]["evaluated"]=True
	
	inputEdges = G.in_edges(nbunch=[thisNode])
	newState = np.zeros((sizeX,sizeY,sizeZ))

	# if "id" in G.graph:
	for inputEdge in inputEdges:
		# oldState = np.max(np.abs(newState))
		newState += calcNodeState(G,inputEdge[0],sizeX=sizeX,sizeY=sizeY,sizeZ=sizeZ)*G.edge[inputEdge[0]][inputEdge[1]]["weight"]
		# if np.max(np.abs(newState))==0:	
			# print "input to",thisNode,"from",inputEdge[0],"(",np.mean(G.node[inputEdge[0]]["state"]),")","with weight",G.edge[inputEdge[0]][inputEdge[1]]["weight"],"increases state from",oldState,"to",np.max(np.abs(newState))
		
	G.node[thisNode]["state"] = newState

	if G.node[thisNode]["function"]=="abs":
		newState = np.abs(newState)
	if G.node[thisNode]["function"]=="nAbs":
		newState = -np.abs(newState)
	if G.node[thisNode]["function"]=="sin":
		newState = np.sin(newState)
	if G.node[thisNode]["function"]=="square":
		newState = np.power(newState,2)
	if G.node[thisNode]["function"]=="nSquare":
		newState = -np.power(newState,2)
	if G.node[thisNode]["function"]=="sqrt":
		newState = np.power(np.abs(newState),0.5)
	if G.node[thisNode]["function"]=="nSqrt":
		newState = -np.power(np.abs(newState),0.5)
	# if G.node[thisNode]["function"]=="blur":
	# 	newState = filters.gaussian_filter(newState, sigma=30)
	if G.node[thisNode]["function"]=="edge":
		newState = filters.convolve(newState,edgeMatrix)
	if G.node[thisNode]["function"]=="gradient":
		newState = morphology.morphological_gradient(newState,size=filterSize)
	if G.node[thisNode]["function"]=="erosion":
		newState = morphology.grey_erosion(newState,size=filterSize)
	if G.node[thisNode]["function"]=="dilation":
		newState = morphology.grey_dilation(newState,size=filterSize)
	if G.node[thisNode]["function"]=="opening":
		newState = morphology.grey_opening(newState,size=filterSize)
	if G.node[thisNode]["function"]=="closing":
		newState = morphology.grey_closing(newState,size=filterSize)
	# if G.node[thisNode]["function"]=="skeleton":
	# 	newState = skeletonize(newState[:,:,0])
	# if G.node[thisNode]["function"]=="coral-RD":
	# 	newState = grayScottReactionDiffusion(newState,0.16, 0.08, 0.060, 0.062, 3000)
	# if G.node[thisNode]["function"]=="fingerprint-RD":
	# 	newState = grayScottReactionDiffusion(newState,0.19, 0.05, 0.060, 0.062, 1000)
	# if G.node[thisNode]["function"]=="spiral-RD":
	# 	newState = grayScottReactionDiffusion(newState,0.12, 0.08, 0.020, 0.050, 1200)
	# if G.node[thisNode]["function"]=="unstable-RD":
	# 	newState = grayScottReactionDiffusion(newState,0.16, 0.08, 0.020, 0.055, 1200)
	# if G.node[thisNode]["function"]=="worm-RD":
	# 	newState = grayScottReactionDiffusion(newState,0.16, 0.08, 0.054, 0.063, 3000)
	# if G.node[thisNode]["function"]=="zebrafish-RD":
	# 	newState = grayScottReactionDiffusion(newState,0.16, 0.08, 0.035, 0.060, 1200)


	# print G.node[thisNode]["function"],"[",np.min(newState),np.max(newState),"]"
	
	newState = 2.0/(1.0+np.exp(-newState))-1.0
	# print G.node[thisNode]["function"],"[",np.min(newState),np.max(newState),"]"
	
	# newState = normalizeMatrix(newState)
	# print G.node[thisNode]["function"],"[",np.min(newState),np.max(newState),"]"

	# if not thisNode in outputNodeNames:
	# 	newState *= scalingFactor
	# print G.node[thisNode]["label"],"[",np.min(newState),np.max(newState),"]"

	return newState

def crossoverBoth(individual1,individual2,networkNum):
	global maxID

	G1 = individual1[networkNum]
	G2 = individual2[networkNum]

	hiddenNodes1 = list(set(G1.nodes()) - set(inputNodeNames[networkNum]) - set(outputNodeNames[networkNum]))
	if len(hiddenNodes1) == 0:
		# print "no hidden nodes in G1, returning G2 without crossover"
		return [individual1,individual2]
	crossoverRoot1 = random.choice(hiddenNodes1)

	hiddenNodes2 = list(set(G2.nodes()) - set(inputNodeNames[networkNum]) - set(outputNodeNames[networkNum]))
	if len(hiddenNodes2) == 0:
		# print "no hidden nodes in G1, returning G2 without crossover"
		return [individual1,individual2]
	crossoverRoot2 = random.choice(hiddenNodes2)

	G1new, numNodes1 = crossover(copy.deepcopy(G1),copy.deepcopy(G2),crossoverRoot1,crossoverRoot2,networkNum)
	G2new, numNodes2 = crossover(copy.deepcopy(G2),copy.deepcopy(G1),crossoverRoot2,crossoverRoot1,networkNum)

	newIndividual1 = copy.deepcopy(individual1)
	newIndividual2 = copy.deepcopy(individual2)

	newIndividual1[networkNum] = G1new
	newIndividual2[networkNum] = G2new

	newIndividual1[0].graph["parentID1"]=individual1[0].graph["id"]
	newIndividual1[0].graph["parentID2"]=individual2[0].graph["id"]
	newIndividual2[0].graph["parentID1"]=individual1[0].graph["id"]
	newIndividual2[0].graph["parentID2"]=individual2[0].graph["id"]

	newIndividual1[0].graph["parentMd5Shape1"] = individual1[0].graph["md5Shape"]
	newIndividual2[0].graph["parentMd5Shape1"] = individual1[0].graph["md5Shape"]
	newIndividual1[0].graph["parentMd5Shape2"] = individual2[0].graph["md5Shape"]
	newIndividual2[0].graph["parentMd5Shape2"] = individual2[0].graph["md5Shape"]

	newIndividual1[0].graph["parentDist1"]=individual1[0].graph["distance"]
	newIndividual1[0].graph["parentDist2"]=individual2[0].graph["distance"]
	newIndividual2[0].graph["parentDist1"]=individual1[0].graph["distance"]
	newIndividual2[0].graph["parentDist2"]=individual2[0].graph["distance"]

	newIndividual1[0].graph["parentFit1"]=individual1[0].graph["fitness"]
	newIndividual1[0].graph["parentFit2"]=individual2[0].graph["fitness"]
	newIndividual2[0].graph["parentFit1"]=individual1[0].graph["fitness"]
	newIndividual2[0].graph["parentFit2"]=individual2[0].graph["fitness"]

	newIndividual1[0].graph["parent1Height"]=individual1[0].graph["height"]
	newIndividual1[0].graph["parent2Height"]=individual2[0].graph["height"]
	newIndividual2[0].graph["parent1Height"]=individual1[0].graph["height"]
	newIndividual2[0].graph["parent2Height"]=individual2[0].graph["height"]

	newIndividual1[0].graph["id"] = maxID
	maxID += 1
	newIndividual2[0].graph["id"] = maxID
	maxID += 1
	newIndividual1[0].graph["age"] = max(individual1[0].graph["age"],individual2[0].graph["age"])
	newIndividual2[0].graph["age"] = max(individual1[0].graph["age"],individual2[0].graph["age"])

	newIndividual1[0].graph["trueAge"] = max(individual1[0].graph["trueAge"],individual2[0].graph["trueAge"])
	newIndividual2[0].graph["trueAge"] = max(individual1[0].graph["trueAge"],individual2[0].graph["trueAge"])

	newIndividual1[0].graph["variationType"] = "crossoverSize"+str(numNodes2+numNodes1)
	newIndividual2[0].graph["variationType"] = "crossoverSize"+str(numNodes2+numNodes1)
	

	# print "crossed",individual1[0].graph["id"],"and",individual2[0].graph["id"],"to make",
	# return [G1new,G2new]
	return [newIndividual1,newIndividual2]

def crossover(G1,G2,crossoverRoot1,crossoverRoot2,networkNum):
	
	for edge in G1.out_edges(nbunch=[crossoverRoot1]):
		G1.remove_edge(edge[0],edge[1])

	subtreeNodes = dfs_tree(G1.reverse(),crossoverRoot1).nodes()

	for thisNode in G1.nodes():
		if not thisNode in subtreeNodes:
			G1.remove_node(thisNode)

	tmpEdges = {}
	for thisNode in inputNodeNames[networkNum]:
		if thisNode in subtreeNodes:
			for thisEdge in G1.out_edges(nbunch=[thisNode]):
				tmpEdges[(thisEdge[0],thisEdge[1])] = G1.edge[thisEdge[0]][thisEdge[1]]["weight"]

	for thisNode in G1.nodes():
		if thisNode in inputNodeNames[networkNum]:
			G1.remove_node(thisNode)

	G2maxHiddenNum = getMaxHiddenNodeIndex(G2)
	newNodeNameDict = {}
	for thisNode in G1.nodes():
		if type(thisNode) is int:
		# if not G1.node[thisNode]["id"] in inputNodeNames[networkNum] and not G1.node[thisNode]["id"] in outputNodeNames[networkNum]:
			newNodeNameDict[thisNode]=thisNode+G2maxHiddenNum
			# print "adding node",thisNode,"as",thisNode+G2maxHiddenNum
		else:
			newNodeNameDict[thisNode]=thisNode
			
	G1 = nx.relabel_nodes(G1,newNodeNameDict)
	
	newG2 = nx.union(G1,G2)
	# for thisNode in inputNodeNames:
	# 	if thisNode in G1.nodes():
	# 		for thisEdge in G1.out_edges(nbunch=[thisNode]):
	# 			newG2.add_edge(thisEdge[0],thisEdge[1],weight=G1.edge[thisEdge[0]][thisEdge[1]]["weight"])
	# print "tmpEdges:",tmpEdges

	for thisEdge in tmpEdges:
		if thisEdge[1] in outputNodeNames:
			newG2.add_edge(thisEdge[0],thisEdge[1],weight=tmpEdges[(thisEdge[0],thisEdge[1])])
		else:
			newG2.add_edge(thisEdge[0],thisEdge[1]+G2maxHiddenNum,weight=tmpEdges[(thisEdge[0],thisEdge[1])])

	for thisEdge in G2.out_edges(nbunch=[crossoverRoot2]):
		newG2.add_edge(crossoverRoot1+G2maxHiddenNum,thisEdge[1],weight=G2.edge[thisEdge[0]][thisEdge[1]]["weight"])
		
	return newG2, len(G1.nodes()) 

def createMinimalNetwork(networkNum):
	G = nx.DiGraph()

	for thisInputNodeName in inputNodeNames[networkNum]:
		G.add_node(thisInputNodeName,nodeType="inputNode",function="none")
	for thisOutputNodeName in outputNodeNames[networkNum]:
		G.add_node(thisOutputNodeName,nodeType="outputNode",function="sigmoid")

	for inputNode in nx.nodes(G):
		if G.node[inputNode]["nodeType"]=="inputNode":
			for outputNode in nx.nodes(G):
				if G.node[outputNode]["nodeType"]=="outputNode":
					G.add_edge(inputNode,outputNode,weight=0.0)
	return G


def addNode(G,networkNum):
	#-----------------------------------------------------------------------------------
	# CHOOSE TWO RANDOM NODES (between which a link could exist)
	if len(G.edges())==0:
		print "no edges in graph!"
		return "NoEdges"
	thisEdge = random.choice(G.edges())
	node1 = thisEdge[0]
	node2 = thisEdge[1]

	#-----------------------------------------------------------------------------------
	# CREATE A NEW NODE HANGING FROM THE PREVIOUS OUTPUT NODE
	newNodeIndex = getMaxHiddenNodeIndex(G)
	G.add_node(newNodeIndex,nodeType="hiddenNode",function="sigmoid")	
	G.add_edge(newNodeIndex,node2,weight=1.0)
	#-----------------------------------------------------------------------------------
	# IF THIS EDGE ALREADY EXISTED HERE, REMOVE IT
	# BUT USE IT'S WEIGHT TO MINIMIZE DISRUPTION WHEN CONNECTING TO PREVIOUS INPUT NODE
	if (node1,node2) in nx.edges(G):
		# print "node from",node1,"to",node2,"exists"
		weight = G.edge[node1][node2]["weight"]
		G.remove_edge(node1,node2)
		G.add_edge(node1,newNodeIndex,weight=weight)
	#-----------------------------------------------------------------------------------
	# IF NOT, USE A WEIGHT OF ZERO (to minimize disruption of new edge)izeX = 100

	else:
		G.add_edge(node1,newNodeIndex,weight=0.0)
	#-----------------------------------------------------------------------------------
	return ""

def addLink(G,networkNum):

	done = False
	attempt = 0
	while not done:
		done = True
		#-----------------------------------------------------------------------------------
		# CHOOSE TWO RANDOM NODES (between which a link could exist, but doesn't)
		node1 = random.choice(G.nodes())
		node2 = random.choice(G.nodes())
		while (not newEdgeIsValid(G,node1,node2)) and attempt<999:
			node1 = random.choice(G.nodes())
			node2 = random.choice(G.nodes())
			attempt += 1
		if attempt > 999:
			print "no valid edges to add found in 1000 attempts."
			done = True
			# return False
		#-----------------------------------------------------------------------------------
		# CREATE A LINK BETWEEN THEM
		if random.random() > 0.5:
			G.add_edge(node1,node2,weight=0.1)
		else:
			G.add_edge(node1,node2,weight=-0.1)
	
		#-----------------------------------------------------------------------------------
		# IF THE LINK CREATES A CYCLIC GRAPH, ERASE IT AND TRY AGAIN
		if hasCycles(G):
			G.remove_edge(node1,node2)
			done = False
			attempt += 1
		if attempt > 999:
			print "no valid edges to add found in 1000 attempts."
			done = True
		#-----------------------------------------------------------------------------------
	return ""

def removeLink(G,networkNum):
	if len(G.edges())==0:
		return "NoEdges"
	thisLink = random.choice(G.edges())
	G.remove_edge(thisLink[0],thisLink[1])
	return ""

def resetNetwork(G,networkNum):
	G = createRandomNetwork(networkNum)
	return ""

def removeNode(G,networkNum):

	hiddenNodes = list(set(G.nodes()) - set(inputNodeNames[networkNum]) - set(outputNodeNames[networkNum]))
	if len(hiddenNodes)==0:
		return "NoHidden"
	thisNode = random.choice(hiddenNodes)

	# if there are edge paths going through this node, keep them connected to minimize disruption
	incomingEdges = G.in_edges(nbunch=[thisNode])
	outgoingEdges = G.out_edges(nbunch=[thisNode])

	for incomingEdge in incomingEdges:
		for outgoingEdge in outgoingEdges:
			G.add_edge(incomingEdge[0],outgoingEdge[1],weight=G.edge[incomingEdge[0]][thisNode]["weight"]*G.edge[thisNode][outgoingEdge[1]]["weight"])

	G.remove_node(thisNode)
	return ""

def pruneNetwork(individual,networkNum):

	done = False
	while not done:
		done = True
		for thisNode in individual[networkNum].nodes():
			if len(individual[networkNum].in_edges(nbunch=[thisNode]))==0 and not thisNode in inputNodeNames[networkNum] and not thisNode in outputNodeNames[networkNum]:
				# print "pruning node",thisNode
				individual[networkNum].remove_node(thisNode)
				done = False
		for thisNode in individual[networkNum].nodes():
			if len(individual[networkNum].out_edges(nbunch=[thisNode]))==0 and not thisNode in inputNodeNames[networkNum] and not thisNode in outputNodeNames[networkNum]:
				# print "pruning node",thisNode
				individual[networkNum].remove_node(thisNode)
				done = False


def mutateWeight(G,networkNum):
	if len(G.edges())==0:
		print "Graph has no edges to mutate!"
		return "NoEdges"
	thisEdge = random.choice(G.edges())
	node1 = thisEdge[0]
	node2 = thisEdge[1]
	oldWeight = G[node1][node2]["weight"]
	newWeight = oldWeight
	while oldWeight == newWeight:
		newWeight = random.gauss(oldWeight,mutationStd)
		newWeight = max(-1.0,min(newWeight,1.0))
	G[node1][node2]["weight"]=newWeight
	return "%.04f"%(newWeight-oldWeight)

def mutFunct(G,networkNum):
	thisNode = random.choice(G.nodes())
	while thisNode in inputNodeNames[networkNum]:
		thisNode = random.choice(G.nodes())
	oldFunction = G.node[thisNode]["function"]
	while G.node[thisNode]["function"] == oldFunction:
		G.node[thisNode]["function"] = random.choice(activationFunctionNames)
	return "_"+oldFunction+"-to-"+G.node[thisNode]["function"]

def hasCycles(G):
	return sum(1 for e in nx.simple_cycles(G)) != 0

def getMaxHiddenNodeIndex(G):
	maxIndex = 0
	for inputNode in nx.nodes(G):
		if G.node[inputNode]["nodeType"]=="hiddenNode" and int(inputNode)>=maxIndex:
			maxIndex = inputNode + 1
	return maxIndex

def newEdgeIsValid(G,node1,node2):
	if node1==node2:
		return False
	if G.node[node1]['nodeType'] == "outputNode":
		return False
	if G.node[node2]['nodeType'] == "inputNode":
		return False
	if (node2,node1) in nx.edges(G):
		return False
	if (node1,node2) in nx.edges(G):
		return False
	return True

def normalizeMatrix(matrix):
	# print "max:",np.max(matrix),"min:",np.min(matrix)
	matrix -= np.min(matrix)
	# print "max:",np.max(matrix),"min:",np.min(matrix)
	matrix /= np.max(matrix)
	# print "max:",np.max(matrix),"min:",np.min(matrix)
	matrix = np.nan_to_num(matrix)
	# print "max:",np.max(matrix),"min:",np.min(matrix)
	matrix *= 2
	# print "max:",np.max(matrix),"min:",np.min(matrix)
	matrix -= 1
	# print "max:",np.max(matrix),"min:",np.min(matrix)
	# print matrix
	# exit(0)
	return matrix

# def grayScottReactionDiffusion(matrix,Du, Dv, F, k, t):
# 	# From: Nicolas P. Rougier (http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html)
# 	# Parameters from http://www.aliensaint.com/uo/java/rd/
# 	# -----------------------------------------------------
# 	# Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065 # Bacteria 1
# 	# Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065 # Bacteria 2
# 	# Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062 # Coral
# 	# Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062 # Fingerprint
# 	# Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050 # Spirals
# 	# Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050 # Spirals Dense
# 	# Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050 # Spirals Fast
# 	# Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055 # Unstable
# 	# Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065 # Worms 1
# 	# Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063 # Worms 2
# 	# Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060 # Zebrafish

# 	sizeX=matrix.shape[0]
# 	sizeY=matrix.shape[1]
# 	sizeZ=matrix.shape[2]

# 	Z = np.zeros((sizeX+2,sizeY+2), [('U', np.double), ('V', np.double)])
# 	U,V = Z['U'], Z['V']
# 	u,v = U[1:-1,1:-1], V[1:-1,1:-1]

# 	r = 20
# 	u[...] = 1.0
# 	# U[n/2-r:n/2+r,n/2-r:n/2+r] = 0.50
# 	# V[n/2-r:n/2+r,n/2-r:n/2+r] = 0.25
# 	# get shape of mateix
# 	# U[1:-1,1:-1] = (matrix[:,:,0]>0)*0.5
# 	# V[1:-1,1:-1] = (matrix[:,:,0]>0)*0.25
# 	U[1:-1,1:-1] = (matrix[:,:,0]/scalingFactor/2.0+0.5)*0.5*1.5
# 	V[1:-1,1:-1] = (matrix[:,:,0]/scalingFactor/2.0+0.5)*0.25*1.5
# 	# calculate edges
# 	U[0,:]=U[1,:]
# 	U[-1,:]=U[-2,:]
# 	V[0,:]=V[1,:]
# 	V[-1,:]=V[-2,:]
# 	U[:,0]=U[:,1]
# 	U[:,-1]=U[:,-2]
# 	V[:,0]=V[:,1]
# 	V[:,-1]=V[:,-2]
# 	u += 0.05*np.random.random((sizeX,sizeY))
# 	v += 0.05*np.random.random((sizeX,sizeY))

# 	for i in xrange(t):
# 		Lu = (                 U[0:-2,1:-1] +
# 			  U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
# 							   U[2:  ,1:-1] )
# 		Lv = (                 V[0:-2,1:-1] +
# 			  V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
# 							   V[2:  ,1:-1] )

# 		uvv = u*v*v
# 		u += (Du*Lu - uvv +  F   *(1-u))
# 		v += (Dv*Lv + uvv - (F+k)*v    )

# 	return np.array(V[1:-1,1:-1]).reshape((sizeX,sizeY,sizeZ))

def shiftFold(matrix,c):
	# todo: implement for fractals
	0


def writeVoxelyzeFileOrig(individual):
	voxelyzeFile = open("voxelyzeFiles/voxelyzeFile--id_%05i.vxa"%individual[0].graph["id"],"w")
	voxelyzeFile.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n\
<VXA Version=\"1.0\">\n\
<Simulator>\n\
<Integration>\n\
<Integrator>0</Integrator>\n\
<DtFrac>0.9</DtFrac>\n\
</Integration>\n\
<Damping>\n\
<BondDampingZ>1</BondDampingZ>\n\
<ColDampingZ>0.8</ColDampingZ>\n\
<SlowDampingZ>0.01</SlowDampingZ>\n\
</Damping>\n\
<Collisions>\n\
<SelfColEnabled>"+str(int(SelfCollisionsEnabled))+"</SelfColEnabled>\n\
<ColSystem>3</ColSystem>\n\
<CollisionHorizon>2</CollisionHorizon>\n\
</Collisions>\n\
<Features>\n\
<FluidDampEnabled>0</FluidDampEnabled>\n\
<PoissonKickBackEnabled>0</PoissonKickBackEnabled>\n\
<EnforceLatticeEnabled>0</EnforceLatticeEnabled>\n\
</Features>\n\
<SurfMesh>\n\
<CMesh>\n\
<DrawSmooth>1</DrawSmooth>\n\
<Vertices/>\n\
<Facets/>\n\
<Lines/>\n\
</CMesh>\n\
</SurfMesh>\n\
<StopCondition>\n\
<StopConditionType>2</StopConditionType>\n\
<StopConditionValue>"+str(fitnessEvaluationCycles/individual[0].graph["frequency"]+fitnessEvalInitTime)+"</StopConditionValue>\n\
<InitCmTime>"+str(fitnessEvalInitTime)+"</InitCmTime>\n\
</StopCondition>\n\
<GA>\n\
<WriteFitnessFile>1</WriteFitnessFile>\n\
<FitnessFileName>"+"fitnessFiles/softbotsOutput--id_%05i.xml"%individual[0].graph["id"]+"</FitnessFileName>\n\
</GA>\n\
</Simulator>\n\
<Environment>\n")
	
	if inCage:
		voxelyzeFile.write("<Boundary_Conditions>\n\
<NumBCs>4</NumBCs>\n\
<FRegion>\n\
<PrimType>0</PrimType>\n\
<X>0</X>\n\
<Y>0</Y>\n\
<Z>0</Z>\n\
<dX>0.01</dX>\n\
<dY>1</dY>\n\
<dZ>1</dZ>\n\
<Radius>0</Radius>\n\
<R>0.4</R>\n\
<G>0.6</G>\n\
<B>0.4</B>\n\
<alpha>1</alpha>\n\
<DofFixed>63</DofFixed>\n\
<ForceX>0</ForceX>\n\
<ForceY>0</ForceY>\n\
<ForceZ>0</ForceZ>\n\
<TorqueX>0</TorqueX>\n\
<TorqueY>0</TorqueY>\n\
<TorqueZ>0</TorqueZ>\n\
<DisplaceX>0</DisplaceX>\n\
<DisplaceY>0</DisplaceY>\n\
<DisplaceZ>0</DisplaceZ>\n\
<AngDisplaceX>0</AngDisplaceX>\n\
<AngDisplaceY>0</AngDisplaceY>\n\
<AngDisplaceZ>0</AngDisplaceZ>\n\
</FRegion>\n\
<FRegion>\n\
<PrimType>0</PrimType>\n\
<X>0</X>\n\
<Y>0</Y>\n\
<Z>0</Z>\n\
<dX>1</dX>\n\
<dY>0.01</dY>\n\
<dZ>1</dZ>\n\
<Radius>0</Radius>\n\
<R>0.4</R>\n\
<G>0.6</G>\n\
<B>0.4</B>\n\
<alpha>1</alpha>\n\
<DofFixed>63</DofFixed>\n\
<ForceX>0</ForceX>\n\
<ForceY>0</ForceY>\n\
<ForceZ>0</ForceZ>\n\
<TorqueX>0</TorqueX>\n\
<TorqueY>0</TorqueY>\n\
<TorqueZ>0</TorqueZ>\n\
<DisplaceX>0</DisplaceX>\n\
<DisplaceY>0</DisplaceY>\n\
<DisplaceZ>0</DisplaceZ>\n\
<AngDisplaceX>0</AngDisplaceX>\n\
<AngDisplaceY>0</AngDisplaceY>\n\
<AngDisplaceZ>0</AngDisplaceZ>\n\
</FRegion>\n\
<FRegion>\n\
<PrimType>0</PrimType>\n\
<X>0.99</X>\n\
<Y>0</Y>\n\
<Z>0</Z>\n\
<dX>0.01</dX>\n\
<dY>1</dY>\n\
<dZ>1</dZ>\n\
<Radius>0</Radius>\n\
<R>0.4</R>\n\
<G>0.6</G>\n\
<B>0.4</B>\n\
<alpha>1</alpha>\n\
<DofFixed>63</DofFixed>\n\
<ForceX>0</ForceX>\n\
<ForceY>0</ForceY>\n\
<ForceZ>0</ForceZ>\n\
<TorqueX>0</TorqueX>\n\
<TorqueY>0</TorqueY>\n\
<TorqueZ>0</TorqueZ>\n\
<DisplaceX>0</DisplaceX>\n\
<DisplaceY>0</DisplaceY>\n\
<DisplaceZ>0</DisplaceZ>\n\
<AngDisplaceX>0</AngDisplaceX>\n\
<AngDisplaceY>0</AngDisplaceY>\n\
<AngDisplaceZ>0</AngDisplaceZ>\n\
</FRegion>\n\
<FRegion>\n\
<PrimType>0</PrimType>\n\
<X>0</X>\n\
<Y>0.99</Y>\n\
<Z>0</Z>\n\
<dX>1</dX>\n\
<dY>0.01</dY>\n\
<dZ>1</dZ>\n\
<Radius>0</Radius>\n\
<R>0.4</R>\n\
<G>0.6</G>\n\
<B>0.4</B>\n\
<alpha>1</alpha>\n\
<DofFixed>63</DofFixed>\n\
<ForceX>0</ForceX>\n\
<ForceY>0</ForceY>\n\
<ForceZ>0</ForceZ>\n\
<TorqueX>0</TorqueX>\n\
<TorqueY>0</TorqueY>\n\
<TorqueZ>0</TorqueZ>\n\
<DisplaceX>0</DisplaceX>\n\
<DisplaceY>0</DisplaceY>\n\
<DisplaceZ>0</DisplaceZ>\n\
<AngDisplaceX>0</AngDisplaceX>\n\
<AngDisplaceY>0</AngDisplaceY>\n\
<AngDisplaceZ>0</AngDisplaceZ>\n\
</FRegion>\n\
</Boundary_Conditions>\n")
	
	elif swarmClimb:
		voxelyzeFile.write("<Boundary_Conditions>\n\
<NumBCs>1</NumBCs>\n\
<FRegion>\n\
<PrimType>0</PrimType>\n\
<X>0</X>\n\
<Y>0.5</Y>\n\
<Z>0</Z>\n\
<dX>1</dX>\n\
<dY>0.5</dY>\n\
<dZ>1</dZ>\n\
<Radius>0</Radius>\n\
<R>0.4</R>\n\
<G>0.6</G>\n\
<B>0.4</B>\n\
<alpha>1</alpha>\n\
<DofFixed>63</DofFixed>\n\
<ForceX>0</ForceX>\n\
<ForceY>0</ForceY>\n\
<ForceZ>0</ForceZ>\n\
<TorqueX>0</TorqueX>\n\
<TorqueY>0</TorqueY>\n\
<TorqueZ>0</TorqueZ>\n\
<DisplaceX>0</DisplaceX>\n\
<DisplaceY>0</DisplaceY>\n\
<DisplaceZ>0</DisplaceZ>\n\
<AngDisplaceX>0</AngDisplaceX>\n\
<AngDisplaceY>0</AngDisplaceY>\n\
<AngDisplaceZ>0</AngDisplaceZ>\n\
</FRegion>\n\
</Boundary_Conditions>\n")

	else:
		voxelyzeFile.write("<Fixed_Regions>\n\
<NumFixed>0</NumFixed>\n\
</Fixed_Regions>\n\
<Forced_Regions>\n\
<NumForced>0</NumForced>\n\
</Forced_Regions>\n")


	voxelyzeFile.write("<Gravity>\n\
<GravEnabled>1</GravEnabled>\n\
<GravAcc>-9.81</GravAcc>\n\
<FloorEnabled>1</FloorEnabled>\n\
<FloorSlope>"+str(floorSlope)+"</FloorSlope>\n\
</Gravity>\n\
<Thermal>\n\
<TempEnabled>1</TempEnabled>\n\
<TempAmp>39</TempAmp>\n\
<TempBase>25</TempBase>\n\
<VaryTempEnabled>1</VaryTempEnabled>\n\
<TempPeriod>"+str(1.0/(individual[0].graph["frequency"]))+"</TempPeriod>\n\
</Thermal>\n\
</Environment>\n\
<VXC Version=\"0.93\">\n\
<Lattice>\n\
<Lattice_Dim>0.01</Lattice_Dim>\n\
<X_Dim_Adj>1</X_Dim_Adj>\n\
<Y_Dim_Adj>1</Y_Dim_Adj>\n\
<Z_Dim_Adj>1</Z_Dim_Adj>\n\
<X_Line_Offset>0</X_Line_Offset>\n\
<Y_Line_Offset>0</Y_Line_Offset>\n\
<X_Layer_Offset>0</X_Layer_Offset>\n\
<Y_Layer_Offset>0</Y_Layer_Offset>\n\
</Lattice>\n\
<Voxel>\n\
<Vox_Name>BOX</Vox_Name>\n\
<X_Squeeze>1</X_Squeeze>\n\
<Y_Squeeze>1</Y_Squeeze>\n\
<Z_Squeeze>1</Z_Squeeze>\n\
</Voxel>\n\
<Palette>\n\
<Material ID=\"1\">\n\
<MatType>0</MatType>\n\
<Name>Passive_Soft</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>1</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>"+str(softestMaterial)+"e+006</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"2\">\n\
<MatType>0</MatType>\n\
<Name>Passive_Hard</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>0</Green>\n\
<Blue>1</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>"+str(softestMaterial)+"e+008</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"3\">\n\
<MatType>0</MatType>\n\
<Name>Active_+</Name>\n\
<Display>\n\
<Red>1</Red>\n\
<Green>0</Green>\n\
<Blue>0</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>"+str(softestMaterial)+"e+006</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"4\">\n\
<MatType>0</MatType>\n\
<Name>Active_-</Name>\n\
<Display>\n\
<Red>0</Red>\n\
<Green>1</Green>\n\
<Blue>0</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>"+str(softestMaterial)+"e+006</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>-0.01</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
<Material ID=\"5\">\n\
<MatType>0</MatType>\n\
<Name>Aperture</Name>\n\
<Display>\n\
<Red>1</Red>\n\
<Green>0.784</Green>\n\
<Blue>0</Blue>\n\
<Alpha>1</Alpha>\n\
</Display>\n\
<Mechanical>\n\
<MatModel>0</MatModel>\n\
<Elastic_Mod>5e+007</Elastic_Mod>\n\
<Plastic_Mod>0</Plastic_Mod>\n\
<Yield_Stress>0</Yield_Stress>\n\
<FailModel>0</FailModel>\n\
<Fail_Stress>0</Fail_Stress>\n\
<Fail_Strain>0</Fail_Strain>\n\
<Density>1e+006</Density>\n\
<Poissons_Ratio>0.35</Poissons_Ratio>\n\
<CTE>0</CTE>\n\
<uStatic>1</uStatic>\n\
<uDynamic>0.5</uDynamic>\n\
</Mechanical>\n\
</Material>\n\
</Palette>\n\
<Structure Compression=\"ASCII_READABLE\">\n\
<X_Voxels>"+str(origSizeX+4*inCage)+"</X_Voxels>\n\
<Y_Voxels>"+str(origSizeY+4*inCage+1*swarmClimb)+"</Y_Voxels>\n\
<Z_Voxels>"+str(origSizeZ+1*inCage)+"</Z_Voxels>\n\
<Data>\n")

	# print "muscleType:",np.min(individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"]),np.max(individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"])

	stringForMd5 = ""
	materialCounts = np.zeros((6))
	makeOneShapeOnly(individual)
	for z in range(origSizeZ+1*inCage):
		voxelyzeFile.write("<Layer><![CDATA[")
		# for y in range(origSizeY):
		# 	for x in range(origSizeX):
		for y in range(-2*inCage,origSizeY+2*inCage):
			
			if swarmClimb and y == int(origSizeY/2):
				for x in range(-2*inCage,origSizeX+2*inCage):
					voxelyzeFile.write("0")

			for x in range(-2*inCage,origSizeX+2*inCage):

				if y < 0 or y >= origSizeY or x < 0 or x >= origSizeX or z >= origSizeZ:
					# if y < -1 or y >= origSizeY+1 or x < -1 or x >= origSizeX+1:
					# 	if ((x-x/2.0)**2+(z-z/2.0)**2)**0.5 < apertureProportion*origSizeZ:
					if ((y < -1 or y >= origSizeY+1) and ((x-origSizeX/2.0)**2+(z-origSizeZ/2.0)**2)**0.5 > apertureProportion*origSizeZ/2.0) or ((x < -1 or x >= origSizeY+1) and ((y-origSizeY/2.0)**2+(z-origSizeZ/2.0)**2)**0.5 > apertureProportion*origSizeZ/2.0) or (z >= origSizeZ and (x==-2 or x==origSizeX+2 or y == -2 or y==origSizeY+2)):
					# if ((y < -1 or y >= origSizeY+1) and ((x-origSizeX/2.0)**2+(z-origSizeZ/2.0)**2)**0.5 > apertureProportion*origSizeZ/2.0) or ((x < -1 or x >= origSizeY+1) and ((y-origSizeY/2.0)**2+(z-origSizeZ/2.0)**2)**0.5 > apertureProportion*origSizeZ/2.0):
						voxelyzeFile.write("5")
						# else:
						# 	voxelyzeFile.write("0")	
					else:
						voxelyzeFile.write("0")
				else:
					# print x,y,z
					# print "matPresent:    ",individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z]
					# print "muscleOrTissue:",individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z]
					# print "muscleType:    ",individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z]
					# print "hardOrSoft:    ",individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z]
					if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][x,y,z] <= 0:
						voxelyzeFile.write("0")
						stringForMd5 += "0"
						materialCounts[0] += 1
						# print "TYPE EMPTY"
					elif swarmClimb and y>=int(origSizeY/2):
						voxelyzeFile.write("5")
						stringForMd5 += "5"
						materialCounts[5] += 1

					elif individual[1*("materialHardOrSoft" in outputNodeNames[1])].node["materialHardOrSoft"]["state"][x,y,z] > 0:
							voxelyzeFile.write("2")
							stringForMd5 += "2"
							materialCounts[2] += 1
							# print "TYPE HARD"
					elif individual[1*("materialMuscleOrTissue" in outputNodeNames[1])].node["materialMuscleOrTissue"]["state"][x,y,z] > 0:
						# if individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"][x,y,z] <= 0:
						# 	voxelyzeFile.write("3")
						# 	stringForMd5 += "3"
						# 	materialCounts[3] += 1
						# 	# print "TYPE MUSCLE 1"
						# else:
						# 	voxelyzeFile.write("4")
						# 	stringForMd5 += "4"
						# 	materialCounts[4] += 1
						# 	# print "TYPE MUSCLE 2"
						voxelyzeFile.write("3")
						stringForMd5 += "3"
						materialCounts[3] += 1
					else:			
						voxelyzeFile.write("1")
						stringForMd5 += "1"
						materialCounts[1] += 1
						# print "TYPE SOFT"

		voxelyzeFile.write("]]></Layer>\n")
	voxelyzeFile.write("</Data>\n")
# 	if 
# 		<NumNeurons>"+str(numTotalNeurons)+"</NumNeurons>\n<Weights>\n")
#
# 			for i in range(numTotalNeurons):
# 				voxelyzeFile.write("<Layer><![CDATA[")
# 				for j in range(numTotalNeurons):
# 					voxelyzeFile.write(str(getSynapseValueAt(G,i,j))+", ")	
# 				voxelyzeFile.write("]]></Layer>\n")
#
# 			voxelyzeFile.write("</Weights>\n")

	stringForMd5Shape = copy.deepcopy(stringForMd5)
	stringForMd5Control = ""

	voxelyzeFile.write("<PhaseOffset>\n")
	for z in range(origSizeZ+1*inCage):
		voxelyzeFile.write("<Layer><![CDATA[")
		# for y in range(origSizeY):
		# 	for x in range(origSizeX):
		for y in range(-2*inCage,origSizeY+2*inCage):

			if swarmClimb and y == int(origSizeY/2):
				for x in range(-2*inCage,origSizeX+2*inCage):
					voxelyzeFile.write("0, ")

			for x in range(-2*inCage,origSizeX+2*inCage):
				if y < 0 or y >= origSizeY or x < 0 or x >= origSizeX or z >= origSizeZ:
					voxelyzeFile.write("0, ")
				
				if swarmClimb and y > int(origSizeY/2):
					voxelyzeFile.write("0, ")

				else:
					voxelyzeFile.write(str(individual[1*("phaseOffset" in outputNodeNames[1])].node["phaseOffset"]["state"][x,y,z])+", ")
					stringForMd5 += str(individual[1*("phaseOffset" in outputNodeNames[1])].node["phaseOffset"]["state"][x,y,z])
					stringForMd5Control += str(individual[1*("phaseOffset" in outputNodeNames[1])].node["phaseOffset"]["state"][x,y,z])
		voxelyzeFile.write("]]></Layer>\n")

	voxelyzeFile.write("</PhaseOffset>\n")

	voxelyzeFile.write("</Structure>\n\
</VXC>\n\
</VXA>")
	voxelyzeFile.close()

	m = hashlib.md5()
	m.update(stringForMd5)

	mShape = hashlib.md5()
	mShape.update(stringForMd5Shape)

	mControl = hashlib.md5()
	mControl.update(stringForMd5Control)
	

	# if np.min(individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"])==0 and 0==np.max(individual[1*("materialMuscleType" in outputNodeNames[1])].node["materialMuscleType"]["state"]):
	# 	for edge in individual[1].in_edges(nbunch=["materialMuscleType"]):
	# 		print edge,individual[1].edge[edge[0]][edge[1]]["weight"]
	print materialCounts, "\t", individual[0].graph["frequency"]
	return [materialCounts,m.hexdigest(),mShape.hexdigest(),mControl.hexdigest()]


def makeOneShapeOnly(individual):

	oldSize = np.sum(individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"]>0)

	if np.sum(individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"]>0) < 2:
		individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"] = individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"]

	else:
		notYetChecked = []
		for z in range(origSizeZ):
			for y in range(origSizeY):
				for x in range(origSizeX):
					notYetChecked.append((x,y,z))

		individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"] = np.zeros((origSizeX,origSizeY,origSizeZ))
		done = False
		largestShape = []
		queueToCheck = []
		while len(notYetChecked) > len(largestShape):
			queueToCheck.append(notYetChecked.pop(0))
			# queueToCheck.append(notYetChecked[0])
			thisShape = []
			if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"][queueToCheck[0]] > 0:
				thisShape.append(queueToCheck[0])
			
			while len(queueToCheck) > 0:
				thisVoxel = queueToCheck.pop(0)
				x = thisVoxel[0]
				y = thisVoxel[1]
				z = thisVoxel[2]

				for neighborVoxel in [(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)]:
					if neighborVoxel in notYetChecked:
						notYetChecked.remove(neighborVoxel)
						if individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["state"][neighborVoxel] > 0:
							queueToCheck.append(neighborVoxel)
							thisShape.append(neighborVoxel)

			if len(thisShape) > len(largestShape):
				largestShape = thisShape

		for loc in thisShape:
			individual[1*("materialPresent" in outputNodeNames[1])].node["materialPresent"]["oneShapeOnly"][loc] = 1


def readSoftbotFitnessFile(filename="softbotsOutput.xml"):
	fitness = 0
	height = 0
	dist = 0
	NumOutOfCageX = 0
	fitnessClimb = 0
	thisFile = open(filename)
	fitnessTagComposite = "<CompositeFitness>"
	fitnessTagHeight = "<Height>"
	# fitnessTagDist = "<Distance>"
	fitnessTagDist = "<DistX>"
	fitnessTagCage = "<NumOutOfCageX>"
	fitnessTagClimb = "<HighestVoxel>"
	# fitnessTag = "<FinalCOM_DistX>"
	for line in thisFile:
		if fitnessTagComposite in line:
			fitness = abs(float(line[line.find(fitnessTagComposite)+len(fitnessTagComposite):line.find("</"+fitnessTagComposite[1:])]))
		if fitnessTagHeight in line:
			height = abs(float(line[line.find(fitnessTagHeight)+len(fitnessTagHeight):line.find("</"+fitnessTagHeight[1:])]))
		if fitnessTagDist in line:
			dist = float(line[line.find(fitnessTagDist)+len(fitnessTagDist):line.find("</"+fitnessTagDist[1:])])
		if fitnessTagCage in line:
			NumOutOfCageX = float(line[line.find(fitnessTagCage)+len(fitnessTagCage):line.find("</"+fitnessTagCage[1:])])
		if fitnessTagClimb in line:
			fitnessClimb = float(line[line.find(fitnessTagClimb)+len(fitnessTagClimb):line.find("</"+fitnessTagClimb[1:])])

	if swarmClimb:
		return (fitness, fitnessClimb, height)

	if inCage:
		return (fitness, NumOutOfCageX, height)

	return (fitness, dist, height)

if __name__ == "__main__":
	mainNeuralNet()
