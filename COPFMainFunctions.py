from COPFGenetic import *
from COPFUtils import *
from operator import itemgetter
from deap import base, creator
import time

toolbox = base.Toolbox()

def mainNonSequentialMulti(instancePath, params, defaultFunc='linear'):
    #Process params
    POPSIZE, NGEN, CXPB, MUTPB, stallCheck = itemgetter('popsize', 'ngen', 'cxpb', 'mutpb', 'stallCheck')(params)

    if instancePath.endswith('sop'):
        TMAX, depot, mapToOriginalPoints, pointPool, clusterPool = readSopFile(instancePath, defaultFunc)
    else:
        TMAX, depot, mapToOriginalPoints, clusterPool = readCopFile(instancePath, defaultFunc)

    refPoints = tools.uniform_reference_points(2, 16)
    omissionProbability = nodeOmissionProbability(depot, pointPool, TMAX)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, clusterPool = clusterPool)
    toolbox.register("select", tools.selNSGA3WithMemory(refPoints))
    toolbox.register("evaluate", evaluateMulti, pointPool = pointPool, depot = depot, clusterPool = clusterPool, TMAX = TMAX)

    #Working on class representations
    try:
        del creator.FitnessMax #Running this so deap stops warning for free
        del creator.Individual
    except:
        pass
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness = creator.FitnessMax)

    #Making indices of the individual, individual and the population
    toolbox.register("indices", generateRandomIndividualClustered, pointPool, clusterPool, omissionPb = omissionProbability)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    start = time.time()
    population = toolbox.population(n = POPSIZE)
    hof = tools.HallOfFame(1)

    result , log = mainEvolution(population, toolbox, cxpb = CXPB, mutpb = MUTPB, ngen = NGEN, stallCheck = stallCheck, verbose=False, stats=None, halloffame=hof)
    end = time.time()

    #Gathering relevant information for testing
    decodedPathBest = decodeIndividual(hof[0], pointPool)
    instanceName = instancePath.split('/')[-1]
    score = toolbox.evaluate(hof[0])[0]
    distance = calculateDistance(depot, decodedPathBest)
    numCluster = len(clusterPool)
    gen = log[-1]['gen'] + 1 #convert 0-start to 1-start

    #Get point order
    pointsVisitedOrder = convertToOriginalPoints(mapToOriginalPoints, hof[0])
    
    funcs = []
    for cluster in clusterPool:
        funcs.append(cluster.function.str)

    return {'instance': instanceName, 'score': score, 'numCluster': numCluster, 'distance': distance, 'TMAX': TMAX, 'runtime': end - start, 'pointsVisitedOrder': pointsVisitedOrder, 'funcs' : funcs, 'gen' : gen}

def mainNonSequentialSingle(instancePath, params, defaultFunc='linear'):
    #Process params
    POPSIZE, NGEN, CXPB, MUTPB, stallCheck = itemgetter('popsize', 'ngen', 'cxpb', 'mutpb', 'stallCheck')(params)

    if instancePath.endswith('sop'):
        TMAX, depot, mapToOriginalPoints, pointPool, clusterPool = readSopFile(instancePath, defaultFunc)
    else:
        TMAX, depot, pointPool, clusterPool = readCopFile(instancePath, defaultFunc)

    omissionProbability = nodeOmissionProbability(depot, pointPool, TMAX)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, clusterPool = clusterPool)
    toolbox.register("select", tools.selTournament, tournsize = 2)
    toolbox.register("evaluate", evaluateSingle, pointPool = pointPool, depot = depot, clusterPool = clusterPool, TMAX = TMAX)

    #Working on class representations
    try:
        del creator.FitnessMax #Running this so deap stops warning for free
        del creator.Individual
    except:
        pass
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)

    #Making indices of the individual, individual and the population
    toolbox.register("indices", generateRandomIndividualClustered, pointPool, clusterPool, omissionPb = omissionProbability)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    start = time.time()
    population = toolbox.population(n = POPSIZE)

    hof = tools.HallOfFame(1)
    result , log = mainEvolution(population, toolbox, cxpb = CXPB, mutpb = MUTPB, ngen = NGEN, stallCheck = stallCheck, verbose=False, stats=None, halloffame=hof)
    end = time.time()

    #Gathering relevant information for testing
    decodedPathBest = decodeIndividual(hof[0], pointPool)
    instanceName = instancePath.split('/')[-1]
    score = toolbox.evaluate(hof[0])[0]
    distance = calculateDistance(depot, decodedPathBest)
    numCluster = len(clusterPool)
    gen = log[-1]['gen'] + 1 #convert 0-start to 1-start

    #Get point order
    pointsVisitedOrder = convertToOriginalPoints(mapToOriginalPoints, hof[0])
    
    funcs = []
    for cluster in clusterPool:
        funcs.append(cluster.function.str)

    return {'instance': instanceName, 'score': score, 'numCluster': numCluster, 'distance': distance, 'TMAX': TMAX, 'runtime': end - start, 'pointsVisitedOrder': pointsVisitedOrder, 'funcs' : funcs, 'gen' : gen}
