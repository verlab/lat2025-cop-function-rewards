from COPFGenetic import *
from COPFUtils import *
from operator import itemgetter
from deap import base, creator
import time

toolbox = base.Toolbox()

def mainNonSequentialMulti(instancePath: str, params: dict, defaultFunc: str='linear') -> dict:
    '''Full algorithm for non-sequential multi-objective approach of the
    Clustered Orienteering Problem with Function-based Rewards.
    '''

    TMAX, depot, mapToOriginalPoints, pointPool, clusterPool = readSopFile(instancePath, defaultFunc)
    #Pack things into dict
    inst = {'TMAX': TMAX, 'depot': depot, 'map': mapToOriginalPoints, 
            'pointPool': pointPool, 'clusterPool': clusterPool, 'path': instancePath}

    refPoints = tools.uniform_reference_points(2, 16)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, clusterPool = clusterPool, 
                     pointMutRate = params['pointMutRate'], mutMoveRate = params['mutMoveRate'])
    toolbox.register("select", tools.selNSGA3WithMemory(refPoints))
    toolbox.register("evaluate", evaluateMulti, pointPool = pointPool, 
                     depot = depot, clusterPool = clusterPool, TMAX = TMAX)

    #Working on class representations
    try:
        del creator.FitnessMax #Running this so deap stops warning for free
        del creator.Individual
    except:
        pass
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness = creator.FitnessMax)

    return mainGeneral(inst, params, toolbox)

def mainNonSequentialSingle(instancePath: str, params: dict, defaultFunc: str='linear') -> dict:
    '''Full algorithm for non-sequential single-objective approach of the 
    Clustered Orienteering Problem with Function-based Rewards.
    '''

    TMAX, depot, mapToOriginalPoints, pointPool, clusterPool = readSopFile(instancePath, defaultFunc)
    #Pack things into dict
    inst = {'TMAX': TMAX, 'depot': depot, 'map': mapToOriginalPoints, 
            'pointPool': pointPool, 'clusterPool': clusterPool, 'path': instancePath}

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, clusterPool = clusterPool, 
                     pointMutRate = params['pointMutRate'], mutMoveRate = params['mutMoveRate'])
    toolbox.register("select", tools.selTournament, tournsize = 2)
    toolbox.register("evaluate", evaluateSingle, pointPool = pointPool, 
                     depot = depot, clusterPool = clusterPool, TMAX = TMAX)

    #Working on class representations
    try:
        del creator.FitnessMax #Running this so deap stops warning for free
        del creator.Individual
    except:
        pass
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMax)

    return mainGeneral(inst, params, toolbox)

def mainGeneral(inst:dict, params: dict, toolbox) -> dict:
    '''General main function, shared by both main functions. Should be run after preparing toolboxes and general methods.'''

    #Process params
    POPSIZE, NGEN, CXPB, MUTPB, stallCheck = itemgetter('popsize', 'ngen', 
                                                        'cxpb', 'mutpb', 'stallCheck')(params)
    
    omissionPb = nodeOmissionProbability(inst['depot'], inst['pointPool'], inst['TMAX'])

    #Making indices of the individual, individual and the population
    toolbox.register("indices", generateRandomIndividualClustered, inst['pointPool'], inst['clusterPool'], omissionPb = omissionPb)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    start = time.time()
    population = toolbox.population(n = POPSIZE)
    hof = tools.HallOfFame(1)

    _ , log = mainEvolution(population, toolbox, cxpb = CXPB, mutpb = MUTPB, ngen = NGEN, 
                                 stallCheck = stallCheck, verbose=False, stats=None, halloffame=hof)
    end = time.time()

    #Gathering relevant information for testing
    decodedPathBest = decodeIndividual(hof[0], inst['pointPool'])
    instanceName = inst['path'].split('/')[-1]
    score = toolbox.evaluate(hof[0])[0]
    distance = calculateDistance(inst['depot'], decodedPathBest)
    numCluster = len(inst['clusterPool'])
    gen = log[-1]['gen'] + 1 #convert 0-start to 1-start

    #Get point order
    pointsVisitedOrder = convertToOriginalPoints(inst['map'], hof[0])
    
    funcs = []
    for cluster in inst['clusterPool']:
        funcs.append(cluster.function.str)

    return {'instance': instanceName, 'score': score, 'numCluster': numCluster, 'distance': distance, 
            'TMAX': inst['TMAX'], 'runtime': end - start, 'pointsVisitedOrder': pointsVisitedOrder, 
            'funcs' : funcs, 'gen' : gen}