from COPFfunctions import linear, exponential, exponential_with_initial, quadratic, logarithmic
funcMap = {'linear' : linear, 'exponential' : exponential, 'exponentiali' : exponential_with_initial, 
           'logarithmic' : logarithmic, 'quadratic' : quadratic}

class Cluster:
    '''Class representing a cluster. Encapsulates its point list and function, for score calculation of each individual.
    
    `function` should be a class that returns an object with `eval` and `name` fields. See ` COPFfunctions.py` for more information.
    '''

    def __init__(self, totalScore: float, pointList: list[int], function, *argv):
        self.pointList = pointList
        self.totalScore = totalScore
        self.function = function(totalScore, len(pointList), *argv)

    def getLength(self) -> int:
        '''Gets the amount of points belonging to this cluster.'''

        return len(self.pointList)
    
    def getIndex(self, index: int) -> int:
        '''Gets point `index` of this cluster.'''

        return self.pointList[index]
    
    def getNumberNodesVisited(self,ind: list[float]) -> int:
        '''Given the individual `ind`, returns the amount of individuals visited by `ind` that belong to this cluster.'''

        return len([ind[i] for i in self.pointList if ind[i] >= 0])
    
    def getScore(self, ind: list[float]) -> float:
        '''Returns the score collected from this cluster given the individual `ind`.'''

        n = self.getNumberNodesVisited(ind)
        return self.function.eval(n)

def readSopFile(filepath: str, defaultFunc='linear', 
                verbose=False) -> tuple[int,complex,dict[int,int],list[complex],list[Cluster]]:
    '''Reads a `.sop` file and returns relevant information. 
    
    The pointPool representation is "clustered", and a point `i` of the resulting pointPool is not equal to a point `i` of the
    original instance. To return a path to the instance's representation, use the returned dict with `convertToOriginalPoints`
    '''
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
    
    #Check if file has functions attributed, otherwise stick to defaultFunc
    hasFunctions = True
    if arr[numPoints+12].split(' ')[1].isnumeric():
        if verbose: print(f"Warning! No functions attributed to instance. Defaulting to {defaultFunc} function for every cluster.")
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

def convertToOriginalPoints(map: dict[int,int], ind: list[float]):
    '''Converts an individual in the clusterized pointPool representation into the original instance's'''

    enumerated = enumerate(ind)
    enumerated = list(filter(lambda x : x[1] >= 0, enumerated))
    enumerated.sort(key = lambda x : x[1])
    return [map[x[0]] + 2 for x in enumerated] #ADD TO COVER FOR DEPOT BEING POINT 1 + MOVING FROM 0-START TO 1-START

def revertToRepresentationPoints(map: dict[int,int], ind: list[float]):
    '''Convert an individual in the original instance pointPool to the clusterized representation.
    Inverse of `convertToOriginalpoints`.
    '''

    reverse = {v: k for k, v in map.items()}
    return [reverse[x-2] for x in ind]