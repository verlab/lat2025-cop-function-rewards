from COPFMainFunctions import mainNonSequentialMulti, mainNonSequentialSingle
import argparse
import sys
import pprint

def parseArgs():
    parser = argparse.ArgumentParser(prog='main.py')

    #'popsize', 'ngen', 'cxpb', 'mutpb', 'stallCheck'
    parser.add_argument('instancePath', type=str, help='path to instance file')
    parser.add_argument('popsize', type=int, help='population size')
    parser.add_argument('ngen', type=int, help='number of generations')
    parser.add_argument('cxpb', type=float, help='probability for crossover')
    parser.add_argument('mutpb', type=float, help='probability for mutation')
    parser.add_argument('stallCheck', type=float, help='maximum number of stall generations')

    #parser.add_argument('-d', help='run in debug mode', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parseArgs()
    if not (0 <= args.cxpb <= 1):
        sys.exit('eror: crossover probability should be in the range [0,1]')
    if not (0 <= args.mutpb <= 1):
        sys.exit('eror: crossover probability should be in the range [0,1]')
    if args.stallCheck <= 0:
        sys.exit('error: stallCheck should be > 0')
    
    params = {'instancePath': args.instancePath, 'popsize': args.popsize,
              'ngen': args.ngen, 'cxpb': args.cxpb, 'mutpb': args.mutpb,
              'stallCheck': args.stallCheck}

    ans = mainNonSequentialMulti(args.instancePath, params)
    del ans['funcs']
    pprint.pprint(ans)
    