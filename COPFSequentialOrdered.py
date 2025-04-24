"""

This file was originally an .ipynb, converted to .py. The sequential and ordered version of this problem has vast different individual representation, which means
different crossover, mutation operators, etc, as well... So it could not be incorporated to the general library seamlessly. Plotting is also not available.

"""

#!/usr/bin/env python
# coding: utf-8

# # "Function" Cluster OP - Ordered
# 
# ### Given a set of clusters associated with a score containing points, find the path that respects the time constraint $T_{max}$, given that the cluster score is collected according to it's specific function f(n), n being the number of nodes visited that belong to the cluster.
# 
# ##### The first and last cluster is the depot, which contains a single point and no score. The vehicle must visit all pretended nodes in each cluster before progressing to the next one.

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from operator import itemgetter
import time


# Define generic algorithm specific parameters and starting the DEAP toolbox.

# In[2]:

toolbox = base.Toolbox()


# Defining a Cluster class which will store and return information relative to the clusters of our future instance.

# In[3]:


class Cluster:
    def __init__(self, totalScore, pointList, function, *argv):
        self._pointlist = pointList
        self.totalScore = totalScore
        self.function = function(totalScore, len(pointList), *argv)

    def getLength(self):
        return len(self._pointlist)
    def getIndex(self, index):
        return self._pointlist[index]
    def getScore(self, n):
        return self.function.eval(n)


# Building a function to read a .cop CEIL_2D file, storing node coordinates, depot coordinates, TMAX and cluster information.

# In[4]:


from COPFfunctions import linear, exponential, exponential_with_initial, quadratic, logarithmic

funcMap = {'linear' : linear, 'exponential' : exponential, 'exponentiali' : exponential_with_initial, 'logarithmic' : logarithmic, 'quadratic' : quadratic}

def readCopFile(filepath, defaultFunc = 'linear'):
    file = open(filepath)
    #Read top of file
    arr = [line.rstrip() for line in file]
    numPoints = int(arr[3].split(' ')[-1])
    TMAX = int(arr[4].split(' ')[-1])
    numClusters = int(arr[5].split(' ')[-1])
    pointPool = []
    
    #Reading points
    aux = arr[8].split(' ')
    DEPOT = complex(float(aux[-2]), float(aux[-1]))

    for i in range(8 + 1, numPoints + 8):
        j = arr[i].split(' ')
        pointPool.append(complex(float(j[-2]), float(j[-1])))

    #Build clusters
    clusterPool = []
    
    #Check if file has functions attributed, otherwise stick to linear
    hasFunctions = True
    if arr[numPoints+9].split(' ')[1].isnumeric():
        print(f"Warning! No functions attributed to instance. Defaulting to {defaultFunc} function for every cluster.")
        hasFunctions = False

    for i in range(numPoints + 9, len(arr)):
        j = arr[i].split(' ')
    
        if hasFunctions:
            funString, *otherargs = j[1].split('_')
            attributedFunction = funcMap[funString]
            otherargs = [float(x) for x in otherargs] #Parsing other arguments
        else:
            attributedFunction = funcMap[defaultFunc]
            otherargs = []

        attributedPoints = []
        for k in range(3 if hasFunctions else 2,len(j)):
            attributedPoints.append(pointPool[int(j[k])-2])
        clusterPool.append(Cluster(int(j[2 if hasFunctions else 1]), attributedPoints, attributedFunction, *otherargs))
    return TMAX, DEPOT, clusterPool

def readSopFile(filepath, defaultFunc = 'linear'):
    file = open(filepath)
    #Read top of file
    arr = [line.rstrip() for line in file]
    numPoints = int(arr[3].split(' ')[-1])
    TMAX = int(arr[4].split(' ')[-1])
    numClusters = int(arr[7].split(' ')[-1])
    pointPool = []
    
    #Reading points
    aux = [x for x in arr[10].split(' ') if x != ''] #TEMP FIX -> Prevents weird files
    DEPOT = complex(float(aux[-2]), float(aux[-1]))

    for i in range(10 + 1, numPoints + 10):
        j = arr[i].split(' ')
        j = [x for x in j if x != ''] #TEMP FIX -> prevents weird files 
        pointPool.append(complex(float(j[-2]), float(j[-1])))

    #Build clusters
    clusterPool = []
    
    #Check if file has functions attributed, otherwise stick to linear
    hasFunctions = True
    if arr[numPoints+12].split(' ')[1].isnumeric():
        #print(f"Warning! No functions attributed to instance. Defaulting to {defaultFunc} function for every cluster.")
        hasFunctions = False

    for i in range(numPoints + 12, len(arr)):
        j = arr[i].split(' ')
    
        if hasFunctions:
            funString, *otherargs = j[1].split('_')
            attributedFunction = funcMap[funString]
            otherargs = [float(x) for x in otherargs] #Parsing other arguments
        else:
            attributedFunction = funcMap[defaultFunc]
            otherargs = []

        attributedPoints = []
        for k in range(3 if hasFunctions else 2,len(j)):
            attributedPoints.append(pointPool[int(j[k])-2])
        clusterPool.append(Cluster(int(j[2 if hasFunctions else 1]), attributedPoints, attributedFunction, *otherargs))
    return TMAX, DEPOT, clusterPool

def countPositive(arr):
    return len(list(filter(lambda x: (x > 0), arr)))


# Defining a ClusterInstance class, which represents a certain cluster and it's node visit order. An individual is represented by a list of cluster instances. The node order is represented as a list of numbers in the range [-1,1] (excluding zero). Using this list, each index carries 3 pieces of information: order (or "priority") of visit, represented by the magnitude of the value stored; if it's visited or not, represented by the number being positive (if it is visited) or negative (if it is not visited), while still storing it's magnitude; original point index in the cluster reference (our original Cluster class), represented by its index. 
# 

# In[5]:


class ClusterInstance:
    def __init__(self, clusterPool, index):
        self.nodeList = [np.random.uniform(low = 0.01) for i in range(clusterPool[index].getLength())]
        self.order = np.random.uniform(low = 0.01)
        self.index = index

def generateRandomIndividual(clusterPool, omissionProbability):
    ind = []
    for i in range(len(clusterPool)):
        ind.append(ClusterInstance(clusterPool, i))
    for cluster in ind:
        for idx in range(len(cluster.nodeList)):
            if random.random() >= omissionProbability:
                cluster.nodeList[idx] *= -1
    return ind


# Building helper functions to translate/decode our representation to an actual path and making functions to calculate the distance of the path and validate an individual according to the instance's $T_{max}$

# In[6]:


#Function to decode an individual due to the [0,1] representation. A decoded path is a list which elements are also lists, which elements are the points of each cluster, ordered.
def decodeIndividual(ind, clusterPool):
    #Sort each cluster instance, if it has at least a single node visited
    activeClusterInstances = sorted([cluster for cluster in ind if any([x for x in cluster.nodeList if x > 0])], key = lambda x : x.order)

    decodedPath = []
    for cluster in activeClusterInstances:
        #Enumerate to preserve original indexes
        enumerated = [(x,y) for x,y in enumerate(cluster.nodeList)]
        #Filter only visited nodes
        visitedNodes = list(filter(lambda x : x[1] >= 0, enumerated))
        #Sort by the second value of the tuple, the order
        visitedNodes.sort(key = lambda x : x[1])
        decodedPath.append([(clusterPool[cluster.index].getIndex(i[0]), i[0]) for i in visitedNodes])
    return decodedPath, activeClusterInstances

def calculateDistance (depot, decodedPath):
    if(len(decodedPath) == 0): return 0

    distance = 0
    rawPath = [j[0] for i in decodedPath for j in i]

    #Distance from depot to first cluster
    distance += abs(depot - rawPath[0])
    #calculate between clusters
    for i in range(0, len(rawPath) - 1):
        distance += abs(rawPath[i+1] - rawPath[i])
    #Distance from last cluster to depot
    distance += abs(depot - rawPath[-1])

    return distance

def makeIndividualValid(depot, ind, TMAX, clusterPool):
    decodedPath, activeClusters = decodeIndividual(ind, clusterPool)
    while(calculateDistance(depot, decodedPath) > TMAX):
        #TODO Make random removal better
        #Choose a random cluster, then a random node from the cluster, "toggle" its visit with its original index and remove from decodedpath
        clusterIdx = random.randint(0,len(activeClusters) - 1)
        nodeIdx = random.randint(0, len(decodedPath[clusterIdx]) - 1)

        #Toggle Visit
        activeClusters[clusterIdx].nodeList[decodedPath[clusterIdx][nodeIdx][1]] *= -1
        del decodedPath[clusterIdx][nodeIdx]
        #Check if cluster is not active anymore
        if len(decodedPath[clusterIdx]) == 0: 
            del decodedPath[clusterIdx]
            del activeClusters[clusterIdx]

    return ind


# To build a decent starting population and running makeIndividualValid too many times in the start of the algorithm, we will build an omission probability function. It will return a good starting probability of disabling a node visit when building an initial population.

# In[7]:


def nodeOmissionProbability(clusterPool, depot, TMAX, loopMax = 10000):
    count, loop = 0,0
    totalNodes = sum([cluster.getLength() for cluster in clusterPool])
    for i in range(loopMax):
        #Build an instance
        instance = [ClusterInstance(clusterPool,idx) for idx in range(len(clusterPool))]
        numNodes = random.randint(0,totalNodes)

        #Generate a particular list, which items are tuples (x,y) - x is the idx of the cluster of the node and y is the idx of the point of the node
        nodeReference = [(x,y) for x in range(len(clusterPool)) for y in range(clusterPool[x].getLength())]
        #Take a random sample of the instance and toggle visit on all of them
        sample = random.sample(nodeReference, numNodes)
        for clusterIdx, nodeIdx in sample: instance[clusterIdx].nodeList[nodeIdx] *= -1 

        #Now check if individual is valid and add statistics
        decodedPath, x = decodeIndividual(instance, clusterPool)
        if(calculateDistance(depot, decodedPath) <= TMAX): count += 1
    return 1 - (count/loopMax)

def totalPathSum(depot,ind,clusterPool):
    all = []
    for cluster in ind:
        enumerated = [(x,y) for x,y in enumerate(cluster.nodeList)]
        enumerated.sort(key = lambda x : abs(x[1]))
        all.append([(clusterPool[cluster.index].getIndex(i[0]), i[0]) for i in enumerated])

    return calculateDistance(depot,all)


# Now building generic genetic algorithm functions.

# In[8]:


def evaluate(ind, depot, TMAX, clusterPool):
    #First we validate our individual
    ind = makeIndividualValid(depot,ind,TMAX, clusterPool)
    
    totalScore = sum([clusterPool[x].getScore(countPositive(ind[x].nodeList)) for x in range(len(ind))])
    
    return totalScore,

#Now we apply point-level crossover (crossing over two points), which will be a two point crossover
def pointCrossover(ind1,ind2):    
    for i in range(len(ind1)):
        if len(ind1[i].nodeList) == 1 : continue
        ind1[i].nodeList, ind2[i].nodeList = tools.cxTwoPoint(ind1[i].nodeList, ind2[i].nodeList)
    return ind1, ind2

#Defining our general crossover function
def crossover(ind1,ind2):
    #Cluster level crossover
    ind1, ind2 = tools.cxTwoPoint(ind1,ind2)
    #Point level crossover
    ind1, ind2 = pointCrossover(ind1,ind2)
    return ind1,ind2

def swapCluster(ind):
    if(len(ind) <= 1): return ind
    #Swaps even if isn't visited
    first = random.randrange(len(ind))
    second = random.randrange(len(ind))
    while(second == first): second = random.randrange(len(ind))
    aux = ind[first].order
    ind[first].order = ind[second].order
    ind[second].order = aux
    return ind

#Node level mutating -> swap points, flip point
def swapPoints(indexpointlist):
    if(len(indexpointlist) <= 1): return indexpointlist
    first = random.randrange(len(indexpointlist))
    second = random.randrange(len(indexpointlist))
    while(second == first): second = random.randrange(len(indexpointlist))
    aux = indexpointlist[first]
    indexpointlist[first] = indexpointlist[second]
    indexpointlist[second] = aux
    return indexpointlist

def flipPoint(clusterInstance):
    idx = random.randint(0, len(clusterInstance.nodeList) - 1)
    clusterInstance.nodeList[idx] *= -1
def mutate(ind, moveRate, pointMutRate):
    #Apply cluster level mutation
    if(np.random.uniform() <= moveRate):
        ind = swapCluster(ind)

    #Start point level
    for i in ind:
        if(random.random() <= pointMutRate):
            flipPoint(i)

    return (ind,)

def eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
    halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Elitism
        k = int(len(population)*.10)
        elite = toolbox.clone(tools.selBest(population + offspring, k))

        # Replace the current population by the offspring
        population[:] = offspring
        
        # Select the next generation population from parents and offspring
        # population[:] = toolbox.select(population + offspring, len(population))              
        
        population[0:k] = elite

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# Defining an instance and starting the algorithm.

# In[9]:
def getPointsVisitedOrder(ind):
    #Sort each cluster instance, if it has at least a single node visited
    activeClusterInstances = sorted([cluster for cluster in ind if any([x for x in cluster.nodeList if x > 0])], key = lambda x : x.order)

    decodedPath = []
    for cluster in activeClusterInstances:
        #Enumerate to preserve original indexes
        enumerated = [(x,y) for x,y in enumerate(cluster.nodeList)]
        #Filter only visited nodes
        visitedNodes = list(filter(lambda x : x[1] >= 0, enumerated))
        #Sort by the second value of the tuple, the order
        visitedNodes.sort(key = lambda x : x[1])
        #Append tuple with (cluster index, [cluster node visit order])
        decodedPath.append((cluster.index, [i[0] for i in visitedNodes]))
    return decodedPath

def mainFunctionOPClusterOPOrdered(instancePath, params, defaultFunc = 'linear'):
    #Process params
    POPSIZE, NGEN, CXPB, MUTPB, MUTMOVERATE, POINTMUTRATE = itemgetter('popsize', 'ngen', 'cxpb', 'mutpb', 'mutMoveRate', 'pointMutRate')(params)

    if instancePath.endswith('sop'):
        TMAX, depot, clusterPool = readSopFile(instancePath, defaultFunc)
    else:
        TMAX, depot, clusterPool = readCopFile(instancePath, defaultFunc)

    omissionProbability = nodeOmissionProbability(clusterPool, depot, TMAX)

    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate, moveRate = MUTMOVERATE, pointMutRate = POINTMUTRATE)
    toolbox.register("select", tools.selTournament, tournsize = 2)
    toolbox.register("evaluate", evaluate, depot = depot, TMAX = TMAX, clusterPool = clusterPool)

    try:
        del creator.FitnessMax #Running this so deap stops warning for free
        del creator.Individual
    except:
        pass

    #Working on class representations
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)

    #Making indices of the individual, individual and the population
    toolbox.register("indices", generateRandomIndividual, clusterPool, omissionProbability = omissionProbability)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    start = time.time()
    population = toolbox.population(n = POPSIZE)

    hof = tools.HallOfFame(1)

    result, log = eaSimpleElitism(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=False, stats=None, halloffame=hof)
    end = time.time()

    #Gathering relevant information for testing
    decodedPathBest, decodedClustersBest = decodeIndividual(hof[0],clusterPool)
    instanceName = instancePath.split('/')[-1]
    score = toolbox.evaluate(hof[0])[0]
    distance = calculateDistance(depot, decodedPathBest)
    numCluster = len(clusterPool)
    pointsVisitedOrder = getPointsVisitedOrder(hof[0])

    #Gathering cluster functions
    funcs = []
    for cluster in clusterPool:
        funcs.append(cluster.function.str)

    return {'instance': instanceName, 'score': score, 'numCluster': numCluster, 'distance': distance, 
            'TMAX': TMAX, 'runtime': end - start, 'pointsVisitedOrder': pointsVisitedOrder, 'funcs' : funcs, 'gen': NGEN}

