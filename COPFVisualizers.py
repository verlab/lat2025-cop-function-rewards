from COPFUtils import *
from COPFGenetic import decodeIndividual
import matplotlib.pyplot as plt

def buildActualPathFromIndividual(decodedind, DEPOT):
    result = []
    result.append(DEPOT)
    result += [x[0] for x in decodedind]
    result.append(DEPOT)
    return result

def buildActualPathFromOrder(order, pointPool, depot):
    return [depot] + [pointPool[x] for x in order] + [depot]

def plotPathFromIndividual(ind, pointPool, depot, alpha=1, color=None, title=""):
    decodedPathBest = decodeIndividual(ind, pointPool)
    path = buildActualPathFromIndividual(decodedPathBest, depot)
    plotPath(path, alpha, color, title)
    
def plotPathFromOrder(order, pointPool, depot, alpha=1, color=None, title=""):
    path = buildActualPathFromOrder(order, pointPool, depot)
    plotPath(path, alpha, color, title)

def plotPath(path, alpha, color=None, title=''):
    plotLine(list(path), alpha=alpha, color=color)
    plotLine([path[0]], style='D', alpha=alpha, size=10, color="green")
    plotLine([path[-1]], style='D', alpha=alpha, size=10, color="orange")
    if title:
        plt.title(title)
    plt.savefig('out.pdf')

def plotLine(points, style='bo-', alpha=1, size=7, color=None):
    X, Y = XY(points)
    
    if color:
        plt.plot(X, Y, style, alpha=alpha, color=color, markersize=size)
    else:
        plt.plot(X, Y, style, alpha=alpha, markersize=size)
    
def XY(points):
    return [p.real for p in points], [p.imag for p in points]

def printClusterSummary(ind, clusterPool):
    print("Cluster Summary:")
    for idx, cluster in enumerate(clusterPool):
        n = cluster.getNumberNodesVisited(ind)
        if n > 0: print(f"Cluster {idx} ({cluster.function.str}): {n} nodes visited, {cluster.getScore(ind):.2f} score collected")