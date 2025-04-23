import random
import numpy as np
from scipy.stats import truncnorm
from deap import algorithms, tools

# General calculations --------

def decodeIndividual(ind, pointPool):
    enumerated = enumerate(ind)
    enumerated = list(filter(lambda x : x[1] >= 0, enumerated))
    #Decode only visited
    enumerated.sort(key = lambda x : x[1])

    return [(pointPool[x[0]], x[0]) for x in enumerated]

def calculateDistance(depot, decodedPath):
    if(len(decodedPath) == 0): return 0

    distance = 0
    rawPath = [i[0] for i in decodedPath]

    #Distance from depot to first cluster
    distance += abs(depot - rawPath[0])

    #calculate between clusters
    for i in range(0, len(rawPath) - 1):
        distance += abs(rawPath[i+1] - rawPath[i])
        
    #Distance from last cluster to depot
    distance += abs(depot - rawPath[-1])

    return distance

def makeIndividualValid(ind, depot, pointPool, TMAX):
    decodedPath = decodeIndividual(ind, pointPool)
    while(calculateDistance(depot, decodedPath) > TMAX):
        #Choose and remove a random node
        randIdx = random.randrange(len(decodedPath))
        #Toggle on ind and remove on decoded
        ind[decodedPath[randIdx][1]] *= -1
        del decodedPath[randIdx]

    return ind 

# Initialization --------

def nodeOmissionProbability(depot, pointPool, TMAX, loopMax = 10000):
    count, loop = 0,0
    totalNodes = len(pointPool)
    for i in range(loopMax):
        #Build an instance
        instance = generateRandomIndividual(pointPool)
        numNodes = random.randint(0,totalNodes)

        #Take a random sample of indexes
        sample = random.sample(range(len(instance)), numNodes)
        for idx in sample: instance[idx] *= -1

        #Now check if individual is valid and add statistics
        decodedPath = decodeIndividual(instance, pointPool)
        if(calculateDistance(depot, decodedPath) <= TMAX): count += 1
    return 1 - (count/loopMax)

def generateRandomIndividual(pointPool, omissionPb = None):
    ind = [np.random.uniform(low = 0.01) for i in pointPool]

    if omissionPb != None:
        for idx in range(len(ind)):
            #Omit
            if random.random() <= omissionPb: ind[idx] *= -1
            
    return ind

def generateRandomIndividualClustered(pointPool, clusterPool, omissionPb = None):
    ind = list(range(len(pointPool)))
    for cluster in clusterPool:
        mean = np.random.uniform(low = 0.01)
        sigma = 0.05
        for point in cluster.pointList:
            ind[point] = float(truncnorm.rvs((0.0-mean)/sigma,(1-mean)/sigma,loc=mean,scale=sigma))

    if omissionPb != None:
        for idx in range(len(ind)):
            #Omit
            if random.random() <= omissionPb: ind[idx] *= -1
    
    return ind

# Genetic Operators --------

def evaluateMulti(ind, depot, pointPool, clusterPool, TMAX):
    #Make ind valid
    ind = makeIndividualValid(ind, depot, pointPool, TMAX)
    decoded = decodeIndividual(ind, pointPool)
    distance = calculateDistance(depot, decoded)
    score = sum([cluster.getScore(ind) for cluster in clusterPool])

    if score == 0 : return -np.inf, -np.inf
    return score, score / distance

def evaluateSingle(ind, depot, pointPool, clusterPool, TMAX):
    #Make ind valid
    ind = makeIndividualValid(ind, depot, pointPool, TMAX)
    decoded = decodeIndividual(ind, pointPool)

    return sum([cluster.getScore(ind) for cluster in clusterPool]),

def mutate(ind, clusterPool):
    cluster = np.random.choice(clusterPool)
    for point in cluster.pointList:
        if random.random() <= 0.15:
            ind[point] *= -1
        elif random.random() <= 0.15:
             #Choose random and swap
            idx = random.choice(cluster.pointList)
            ind[point], ind[idx] = ind[idx], ind[point]
                     
    return ind,

# Genetic Algorithm --------

def mainEvolution(population, toolbox, cxpb, mutpb, ngen, stallCheck=None, stats=None, halloffame = None, verbose = __debug__, seed=None):
    random.seed(seed)
    POPSIZE = len(population)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    if halloffame is not None:
            halloffame.update(population)

    # Compile statistics about the population
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    if stallCheck is not None:
        best = toolbox.evaluate(tools.selBest(population, 1)[0])[0]

    # Begin the generational process
    for gen in range(1, ngen):
        if stallCheck is not None and gen % 100 == 0:
            newBest = toolbox.evaluate(tools.selBest(population, 1)[0])[0]
            if best == newBest: break #Algorithm has converged
            else: best = newBest

        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        if halloffame is not None:
                halloffame.update(offspring)

        # Select the next generation population from parents and offspring
        population[:] = toolbox.select(population + offspring, POPSIZE)

        # Compile statistics about the new population
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    
    if halloffame is not None:
            halloffame.update(population)

    return population, logbook
