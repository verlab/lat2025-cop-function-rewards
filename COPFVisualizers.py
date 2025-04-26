from COPFUtils import *
from COPFGenetic import decodeIndividual
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import shapely.geometry as geometry
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay

colorMap = {'linear' : 'orange', 'quadratic' : 'r', 'exponential' : 'm', 'logarithmic' : 'y', 'exponential_with_initial' : 'c'}

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

def plotCircles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection

def alpha_shape(points, alpha):
   
    if len(points) < 4:
        return geometry.MultiPoint(list(points)).convex_hull
    
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[ [i, j] ])
        
    coords = np.array([point for point in points])
    #print("coords", coords)
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
  
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        
        s = (a + b + c) / 2.0
        #print("s", s)
    
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        #print("area", area)
        circum_r = a * b * c / (4.0 * area)
        #print("circum_r", circum_r)
        
        #print("alpha", alpha)
        
        if alpha == 0 or circum_r < 1.0 / alpha:
            #print("add edge")
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
   
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles)

def plot_polygon(polygon, fc='#999999', ec='#000000'):
    plt.fill(*polygon.exterior.xy, c=fc)

def plotManualLegend(funcList, colorMap, fontsize=12):
    uniqueFunctions = set(funcList)
    handles = []
    for func in uniqueFunctions:
        handles.append(mpatches.Patch(color=colorMap[func], label=func))
    plt.legend(handles=handles, loc='upper left', fontsize=fontsize)

def pointListToPoint(pointList, indexes):
    return [[pointList[index][0], pointList[index][1]] for index in indexes]

def drawPolys(points, clusters, funcList, bufferRatio=.01, fontsize=12, legendFontsize=12):
    allX = [p[0] for p in points]
    allY = [p[1] for p in points]
    xRange = max(allX) - min(allX)
    yRange = max(allY) - min(allY)
    
    # Use the average range as the basis for buffer size
    avg_range = (xRange + yRange) / 2
    dynamicBuffer = avg_range * bufferRatio

    for cluster in clusters:
        concaveHull = alpha_shape(pointListToPoint(points, cluster.pointList),0)
        plot_polygon(concaveHull.buffer(dynamicBuffer), colorMap[cluster.function.str])
        #new
        cpoints = [points[x] for x in cluster.pointList] #Get points
        x = [x[0] for x in cpoints]
        y = [x[1] for x in cpoints]
        plt.text(np.mean(x), np.mean(y), f"{cluster.totalScore}", fontsize=fontsize)
        
    plotManualLegend(funcList, colorMap, legendFontsize)

def plotPathFancy(instancePath, defaultFunc, order):
    _, depot, pointMap, pointPool, clusterPool = readSopFile(instancePath, defaultFunc)
    path = revertToRepresentationPoints(pointMap, order)
    tuplePointPool = [(x.real,x.imag) for x in pointPool]
    tupleDepot = (depot.real,depot.imag)

    maxx, minx = max([x[0] for x in tuplePointPool]), min([x[0] for x in tuplePointPool])
    maxy, miny = max([x[1] for x in tuplePointPool]), min([x[1] for x in tuplePointPool])
    
    #Adjust figure
    FIG_HEIGHT = 12
    fig_width = FIG_HEIGHT*(maxx-minx)/(maxy-miny)
    figsize = (fig_width*0.9,FIG_HEIGHT)
    plt.figure(num=None, figsize=figsize, dpi=100, facecolor='w', edgecolor='k')

    #Draw polygons
    drawPolys(tuplePointPool, clusterPool, [c.function.str for c in clusterPool], bufferRatio=.01, fontsize=20, legendFontsize=20.5)

    #Draw path
    actualPath = buildActualPathFromOrder(path, tuplePointPool, tupleDepot)
    plt.plot([x[0] for x in actualPath], [x[1] for x in actualPath], '-g', lw=3) 

    #Draw circles and "small circles around"
    plotCircles([x[0] for x in tuplePointPool], [x[1] for x in tuplePointPool], 1, alpha=0.1, edgecolor='black', linewidth=0.9, linestyle=':')
    plt.plot([x[0] for x in tuplePointPool], [x[1] for x in tuplePointPool], 'ok', ms=4.75)

    #Draw depot
    plotCircles([tupleDepot[0]], [tupleDepot[1]], 1.25, c='b', alpha=.6, edgecolor='black', linewidth=0.9, linestyle=':')
    plt.plot([tupleDepot[0]], [tupleDepot[1]], 'ok', ms=5)

    plt.savefig(f"output.pdf", format="pdf", bbox_inches="tight")