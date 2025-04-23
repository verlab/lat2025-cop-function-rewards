from COPFfunctions import linear, exponential, exponential_with_initial, quadratic, logarithmic
funcMap = {'linear' : linear, 'exponential' : exponential, 'exponentiali' : exponential_with_initial, 
           'logarithmic' : logarithmic, 'quadratic' : quadratic}

class Cluster:
    def __init__(self, totalScore, pointList, function, *argv):
        self.pointList = pointList
        self.totalScore = totalScore
        self.function = function(totalScore, len(pointList), *argv)
    def getLength(self):
        return len(self.pointList)
    def getIndex(self, index):
        return self.pointList[index]
    def getNumberNodesVisited(self,ind):
        return len([ind[i] for i in self.pointList if ind[i] >= 0])
    
    def getScore(self, ind):
        n = self.getNumberNodesVisited(ind)
        return self.function.eval(n)

def readCopFile(filepath, defaultFunc='linear', verbose=False):
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
        if verbose: print("Warning! No functions attributed to instance. Defaulting to linear function for every cluster.")
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
            attributedPoints.append(int(j[k])-2)
        clusterPool.append(Cluster(int(j[2 if hasFunctions else 1]), attributedPoints, attributedFunction, *otherargs))
    return TMAX, DEPOT, pointPool, clusterPool

def readSopFile(filepath, defaultFunc='linear', verbose=False):
    file = open(filepath)
    #Read top of file
    arr = [line.rstrip() for line in file]
    numPoints = int(arr[3].split(' ')[-1])
    TMAX = int(arr[4].split(' ')[-1])
    numClusters = int(arr[7].split(' ')[-1])
    #The original point pool as described by the instance
    pointPool = []
    
    #Reading points
    aux = arr[10].split(' ')
    DEPOT = complex(float(aux[-2]), float(aux[-1]))

    for i in range(10 + 1, numPoints + 10):
        j = arr[i].split(' ')
        pointPool.append(complex(float(j[-2]), float(j[-1])))

    #Build clusters
    clusterPool = []
    finalPointPool = [] #"Clusterized" point pool
    originalConvertedPointsMap = {} # Map clusterized to original points
    
    #Check if file has functions attributed, otherwise stick to linear
    hasFunctions = True
    if arr[numPoints+12].split(' ')[1].isnumeric():
        if verbose: print("Warning! No functions attributed to instance. Defaulting to linear function for every cluster.")
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
            #Convert to "clustered" representation and keep track of mappings/already added
            convertedIdx = len(finalPointPool)
            originalIdx = int(j[k]) - 2

            #ONLY add to pointPool if not yet converted
            if originalIdx not in originalConvertedPointsMap:
                attributedPoints.append(convertedIdx) #Add both to cluster points and finalpointpool
                finalPointPool.append(pointPool[originalIdx])
                originalConvertedPointsMap[originalIdx] = convertedIdx # Keep track of conversion

            else:
                attributedPoints.append(originalConvertedPointsMap[originalIdx])
                
        clusterPool.append(Cluster(int(j[2 if hasFunctions else 1]), attributedPoints, attributedFunction, *otherargs))
    
    #Finally, reverse our dictionary since we want to map convertedIdx -> originalIdx
    originalConvertedPointsMap = {v: k for k, v in originalConvertedPointsMap.items()}
    return TMAX, DEPOT, originalConvertedPointsMap, finalPointPool, clusterPool

def convertToOriginalPoints(map, ind):
    enumerated = enumerate(ind)
    enumerated = list(filter(lambda x : x[1] >= 0, enumerated))
    enumerated.sort(key = lambda x : x[1])
    return [map[x[0]] + 2 for x in enumerated] #ADD TO COVER FOR DEPOT BEING POINT 1 + MOVING FROM 0-START TO 1-START

def revertToRepresentationPoints(map,ind):
    reverse = {v: k for k, v in map.items()}
    return [reverse[x-2] for x in ind]