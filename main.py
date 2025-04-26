from COPFMainFunctions import mainNonSequentialMulti, mainNonSequentialSingle
from COPFSequentialOrdered import mainFunctionOPClusterOPOrdered
from COPFUtils import funcMap, readSopFile, revertToRepresentationPoints
from COPFVisualizers import plotPathFancy

import argparse
import sys
import pprint

def parseArgs():
    parser = argparse.ArgumentParser(prog='main.py')

    #GA params
    parser.add_argument('instancePath', type=str, help='path to instance file')
    parser.add_argument('popsize', type=int, help='population size')
    parser.add_argument('ngen', type=int, help='number of generations')
    parser.add_argument('cxpb', type=float, help='probability for crossover')
    parser.add_argument('mutpb', type=float, help='probability for mutation')
    parser.add_argument('stallCheck', type=float, help='maximum number of stall generations')

    #Optional GA params
    parser.add_argument('--mutMove', type=float, help='Rate of which points of the same cluster are swapped in mutation', default=.15, metavar='rate')
    parser.add_argument('--mutPoint', type=float, help='Rate of which points of the same cluster are toggled in mutation', default=.15, metavar='rate')

    #Which main to run
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--single', action='store_true', help='Execute non-sequential single objective implementation')
    group.add_argument('--singleSeq', action='store_true', help='Execute sequential single objective implementation')

    #Program params
    parser.add_argument('-d', type=str, help='default function assumed of the cluster, in case no function specified', default='linear', metavar='default-func')

    return parser.parse_args()


if __name__ == '__main__':
    args = parseArgs()

    if not (0 <= args.cxpb <= 1):
        sys.exit('eror: crossover probability should be in the range [0,1]')
    if not (0 <= args.mutpb <= 1):
        sys.exit('eror: crossover probability should be in the range [0,1]')
    if args.stallCheck <= 0:
        sys.exit('error: stallCheck should be > 0')
    if args.d not in funcMap:
        sys.exit(f'error: default function {args.d} should be one of: {[k for k in funcMap.keys()]}')
    
    params = {'instancePath': args.instancePath, 'popsize': args.popsize,
              'ngen': args.ngen, 'cxpb': args.cxpb, 'mutpb': args.mutpb,
              'stallCheck': args.stallCheck, 'mutMoveRate': args.mutMove,
              'pointMutRate': args.mutPoint}
    
    method = mainNonSequentialMulti
    if args.single: method = mainNonSequentialSingle
    if args.singleSeq: method = mainFunctionOPClusterOPOrdered

    ans = method(args.instancePath, params, args.d)

    #Delete long prints
    del ans['funcs']
    order = ans['pointsVisitedOrder']
    del ans['pointsVisitedOrder']

    #Print summary
    pprint.pprint(ans)

    #We can also plot! (if not ordered)
    if not args.singleSeq:
        plotPathFancy(args.instancePath, args.d, order)
    else:
        print("Sequential single-objective implementation does not support visualization")