import multiprocessing as mp
from COPFMainFunctions import mainNonSequentialMulti, mainNonSequentialSingle
from COPFSequentialOrdered import mainFunctionOPClusterOPOrdered
from COPFUtils import funcMap
import argparse
import os.path

parser = argparse.ArgumentParser(prog='Benchmarker.py')
parser.add_argument('instancesPath', type=str, help='path to list of instances')
parser.add_argument('execs', type=int, help='number of times to execute each instance, method and function combination')

parser.add_argument('--multi', action='store_true', help='Benchmark non-sequential multi objective implementation')
parser.add_argument('--single', action='store_true', help='Benchmark non-sequential single objective implementation')
parser.add_argument('--singleSeq', action='store_true', help='Benchmark sequential single objective implementation')


args = parser.parse_args()

#Define parameters for testing
params = {'popsize': 400, 'ngen': 1, 'cxpb': 0.7, 'mutpb': 0.4, 'mutMoveRate': 0.75, 'pointMutRate': 0.75, 'stallCheck': 100}

order = {'instance': 'Instance Name', 'score': 'Score', 'numCluster': 'Cluster Num',
         'distance': 'Total Distance', 'TMAX': 'TMAX', 'runtime': 'Runtime',
         'pointsVisitedOrder' : 'Path', 'funcs': 'Cluster Function type', 'gen': 'Generations'}

def buildInstanceList(inputFilePath):
    file = open(inputFilePath, 'r')
    instances = file.read().splitlines()
    file.close()
    return instances

def writeHeader(params, filePath, benchMarkName):
    file = open(filePath, 'a')
    file.write(f"# {benchMarkName} benchmark with parameters :")
    for key, value in params.items():
        file.write(f" -{key} = {value}")
    file.write(f"\n{';'.join(order.values())}\n")
    file.close()

def writeToOutput(outputFilePath, results):
    #Write and close
    file = open(outputFilePath, 'a')
    file.write(f"{';'.join([str(results[i]) for i in order.keys()])}\n")
    file.close()

def benchmarkSingle(benchFunc, params, inputFilePath, outputFilePath, loopRange, defaultFunc='linear'):
    'Benchmarks a SINGLE instance loopRange times.'
    for _ in range(loopRange):
        results = benchFunc(f"{inputFilePath}", params, defaultFunc)
        writeToOutput(outputFilePath, results)

SOPINSTANCEPATH = './sop-instances'
SOPINSTANCERANDOMPATH = './sop-instances-random'
OUTPUTPATH = './results'

pool = mp.Pool(processes=None)
instances = buildInstanceList(args.instancesPath)

#Exponential with initial is not tested...
del funcMap['exponentiali']

toTest = []

if not (args.multi or args.single or args.singleSeq): #Test everything
    toTest = [(mainNonSequentialMulti, 'non-sequential-multi'), (mainNonSequentialSingle, 'non-sequential-single'),
             (mainFunctionOPClusterOPOrdered, 'sequential-single')]
else: #Test selected
    if args.multi: toTest.append((mainNonSequentialMulti, 'non-sequential-multi'))
    if args.single: toTest.append((mainNonSequentialSingle, 'non-sequential-single'))
    if args.singleSeq: toTest.append((mainFunctionOPClusterOPOrdered, 'sequential-single'))

#Run things
for method,name in toTest:
    #Try all with a single function
    for func in funcMap.keys():
        outputFile = f'{OUTPUTPATH}/{name}/All{func}.csv'
        if not os.path.isfile(outputFile):
            writeHeader(params, outputFile, name)
        toRun = [(method, params, f'{SOPINSTANCEPATH}/{inst}', outputFile, args.execs, func) for inst in instances]
        pool.starmap(benchmarkSingle, toRun)

    #Try random functions
    outputFile = f'{OUTPUTPATH}/{name}/AllRandom.csv'
    if not os.path.isfile(outputFile):
            writeHeader(params, outputFile, name)
    toRun = [(method, params, f'{SOPINSTANCERANDOMPATH}/{inst}', outputFile, args.execs, func) for inst in instances]
    pool.starmap(benchmarkSingle, toRun)