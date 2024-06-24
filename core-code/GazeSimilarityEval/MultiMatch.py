#!/usr/bin/env python

# import cupy as np
from asyncio.base_tasks import _task_print_stack
import math
import numpy as np
import sys
import logging
import scipy.sparse as sp
import glob
import json
from tqdm import tqdm


def cart2pol(x, y):
    """Transform cartesian into polar coordinates.

    :param x: float
    :param y : float

    :return: rho: float, length from (0,0)
    :return: theta: float, angle in radians
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return rho, theta

def checkRange(val,range):
    dowlim=range[0]
    uplim=range[1]
    if(val>uplim):
        return uplim
    if(val<dowlim):
        return dowlim
    return  val
def calcangle(x1, x2):
    """Calculate angle between to vectors (saccades).

    :param: x1, x2: list of float

    :return: angle: float, angle in degrees
    """
    np.seterr(divide='ignore', invalid='ignore')
    val=np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    val=checkRange(val,[-1,1])
    angle = math.degrees(
        math.acos(val)
    )
    return angle


def remodnav_reader(data, screensize, pursuits=False):
    """
    Helper function to read and preprocess REMoDNaV data for use in
    interactive python sessions.

    :param data: path to a REMoDNaV output
    :param screensize: list, screendimensions in x and y direction
    :param pursuits: if True, pursuits will be relabeled to fixations
    """
    from multimatch_gaze.tests import utils as ut

    data = ut.read_remodnav(data)
    # this function can be called without any previous check that
    # screensize are two values, so I'm putting an additional check
    # here
    try:
        assert len(screensize) == 2
    except:
        raise ValueError(
            "Screensize should be the dimensions of the"
            "screen in x and y direction, such as "
            "[1000, 800]. I received {}.".format(screensize)
        )
    if pursuits:
        data = ut.pursuits_to_fixations(data)
    data = ut.preprocess_remodnav(data, screensize)
    return data


def gen_scanpath_structure(data):
    """Transform a fixation vector into a vector based scanpath representation.

    Takes an nx3 fixation vector (start_x, start_y, duration) in the form of
    of a record array and transforms it into a vector-based scanpath
    representation in the form of a nested dictionary. Saccade starting and
    end points, as well as length in x & y direction, and vector length (theta)
    and direction (rho) are calculated from fixation coordinates as a vector
    representation in 2D space.
    Structure:
    fix --> fixations --> (start_x, start_y, duration)
    sac --> saccades --> (start_x, start_y, lenx, leny, rho, theta)

    :param: data: record array

    :return: eyedata: dict, vector-based scanpath representation
    """

    # everything into a dict
    # keep coordinates and durations of fixations
    # fixations = dict(x=data["start_x"], y=data["start_y"], dur=data["duration"],)
    fixations = dict(x=data["x"], y=data["y"], dur=data["duration"], )
    # calculate saccade length and angle from vector lengths between fixations
    lenx = np.diff(data["x"])
    leny = np.diff(data["y"])
    # lenx = np.diff(data["start_x"])
    # leny = np.diff(data["start_y"])
    rho, theta = cart2pol(lenx, leny)

    saccades = dict(
        # fixations are the start coordinates for saccades
        # x=data[:-1]["start_x"],
        # y=data[:-1]["start_y"],
        x=data["x"][:-1],
        y=data["y"][:-1],
        lenx=lenx,
        leny=leny,
        theta=theta,
        rho=rho,
    )
    return dict(fix=fixations, sac=saccades)


def keepsaccade(i, j, sim, data):
    """
    Helper function for scanpath simplification. If no simplification can be
    performed on a particular saccade, this functions stores the original data.
    :param i: current index
    :param j: current index
    :param sim: dict with current similarities
    :param data: original dict with vector based scanpath representation
    """
    for t, k in (
        ("sac", "lenx"),
        ("sac", "leny"),
        ("sac", "x"),
        ("sac", "y"),
        ("sac", "theta"),
        ("sac", "rho"),
        ("fix", "dur"),
    ):
        sim[t][k].insert(j, data[t][k][i])

    return i + 1, j + 1


def _get_empty_path():
    return dict(
        fix=dict(dur=[],),
        sac=dict(
            x=[],
            y=[],
            lenx=[],
            leny=[],
            theta=[],
            # why 'len' here and 'rho' in input data?
            # MIH -> always rho
            # len=[],
            rho=[],
        ),
    )


def simlen(path, TAmp, TDur):
    """Simplify scanpaths based on saccadic length.

    Simplify consecutive saccades if their length is smaller than the
    threshold TAmp and the duration of the closest fixations is lower
    than threshold TDur.

    :param: path: dict, output of gen_scanpath_structure
    :param: TAmp: float, length in px
    :param: TDur: float, time in seconds

    :return: eyedata: dict; one iteration of length based simplification
    """
    # shortcuts
    saccades = path["sac"]
    fixations = path["fix"]

    if len(saccades["x"]) <= 1:
        return path

    # the scanpath is long enough
    i = 0
    j = 0
    sim = _get_empty_path()
    # while we don't run into index errors
    while i <= len(saccades["x"]) - 1:
        # if saccade is the last one
        if i == len(saccades["x"]) - 1:
            # and if saccade has a length shorter than the threshold:
            if saccades["rho"][i] < TAmp:
                # and if the fixation duration is short:
                if (fixations["dur"][-1] < TDur) or (fixations["dur"][-2] < TDur):
                    # calculate sum of local vectors for simplification
                    v_x = saccades["lenx"][-2] + saccades["lenx"][-1]
                    v_y = saccades["leny"][-2] + saccades["leny"][-1]
                    rho, theta = cart2pol(v_x, v_y)
                    # save them in the new vectors
                    sim["sac"]["lenx"][j - 1] = v_x
                    sim["sac"]["leny"][j - 1] = v_y
                    sim["sac"]["theta"][j - 1] = theta
                    sim["sac"]["rho"][j - 1] = rho
                    sim["fix"]["dur"].insert(j, fixations["dur"][i - 1])
                    j -= 1
                    i += 1
                # if fixation duration is longer than the threshold:
                else:
                    # insert original event data in new list -- no
                    # simplification
                    i, j = keepsaccade(i, j, sim, path)
            # if saccade does NOT have a length shorter than the threshold:
            else:
                # insert original path in new list -- no simplification
                i, j = keepsaccade(i, j, sim, path)
        # if saccade is not the last one
        else:
            # and if saccade has a length shorter than the threshold
            if (saccades["rho"][i] < TAmp) and (i < len(saccades["x"]) - 1):
                # and if fixation durations are short
                if (fixations["dur"][i + 1] < TDur) or (fixations["dur"][i] < TDur):
                    # calculate sum of local vectors in x and y length for
                    # simplification
                    v_x = saccades["lenx"][i] + saccades["lenx"][i + 1]
                    v_y = saccades["leny"][i] + saccades["leny"][i + 1]
                    rho, theta = cart2pol(v_x, v_y)
                    # save them in the new vectors
                    sim["sac"]["lenx"].insert(j, v_x)
                    sim["sac"]["leny"].insert(j, v_y)
                    sim["sac"]["x"].insert(j, saccades["x"][i])
                    sim["sac"]["y"].insert(j, saccades["y"][i])
                    sim["sac"]["theta"].insert(j, theta)
                    sim["sac"]["rho"].insert(j, rho)
                    # add the old fixation duration
                    sim["fix"]["dur"].insert(j, fixations["dur"][i])
                    i += 2
                    j += 1
                # if fixation durations longer than the threshold
                else:
                    # insert original path in new lists -- no simplification
                    i, j = keepsaccade(i, j, sim, path)
            # if saccade does NOT have a length shorter than the threshold:
            else:
                # insert original path in new list -- no simplification
                i, j = keepsaccade(i, j, sim, path)
    # append the last fixation duration
    sim["fix"]["dur"].append(fixations["dur"][-1])

    return sim


def simdir(path, TDir, TDur):
    """Simplify scanpaths based on angular relations between saccades (direction).

    Simplify consecutive saccades if the angle between them is smaller than the
    threshold TDir and the duration of the intermediate fixations is lower
    than threshold TDur.

    :param: path: dict, output of gen_scanpath_structure
    :param: TDir: float, angle in degrees
    :param: TDur: float, time in seconds

    :return: eyedata: dict, one iteration of direction based simplification
    """
    # shortcuts
    saccades = path["sac"]
    fixations = path["fix"]

    if len(saccades["x"]) < 1:
        return path
    # the scanpath is long enough
    i = 0
    j = 0
    sim = _get_empty_path()
    # while we don't run into index errors
    while i <= len(saccades["x"]) - 1:
        if i < len(saccades["x"]) - 1:
            # lets check angles
            v1 = [saccades["lenx"][i], saccades["leny"][i]]
            v2 = [saccades["lenx"][i + 1], saccades["leny"][i + 1]]
            angle = calcangle(np.array(v1), np.array(v2))
        else:
            # an angle of infinite size won't go into any further loop
            angle = float("inf")
        # if the angle is smaller than the threshold and its not the last saccade
        if (angle < TDir) & (i < len(saccades["x"]) - 1):
            # if the fixation duration is short:
            if fixations["dur"][i + 1] < TDur:
                # calculate the sum of local vectors
                v_x = saccades["lenx"][i] + saccades["lenx"][i + 1]
                v_y = saccades["leny"][i] + saccades["leny"][i + 1]
                rho, theta = cart2pol(v_x, v_y)
                # save them in the new vectors
                sim["sac"]["lenx"].insert(j, v_x)
                sim["sac"]["leny"].insert(j, v_y)
                sim["sac"]["x"].insert(j, saccades["x"][i])
                sim["sac"]["y"].insert(j, saccades["y"][i])
                sim["sac"]["theta"].insert(j, theta)
                sim["sac"]["rho"].insert(j, rho)
                # add the fixation duration
                sim["fix"]["dur"].insert(j, fixations["dur"][i])
                i += 2
                j += 1
            else:
                # insert original data in new list -- no simplification
                i, j = keepsaccade(i, j, sim, path)
        else:
            # insert original path in new list -- no simplification
            i, j = keepsaccade(i, j, sim, path)
    # now append the last fixation duration
    sim["fix"]["dur"].append(fixations["dur"][-1])

    return sim


def simplify_scanpath(path, TAmp, TDir, TDur):
    """Simplify scanpaths until no further simplification is possible.

    Loops over simplification functions simdir and simlen until no
    further simplification of the scanpath is possible.

    :param: path: dict, vector based scanpath representation,
                  output of gen_scanpath_structure
    :param: TAmp: float, length in px
    :param: TDir: float, angle in degrees
    :param: TDur: float, duration in seconds

    :return: eyedata: dict, simplified vector-based scanpath representation
    """
    prev_length = len(path["fix"]["dur"])
    while True:
        path = simdir(path, TDir, TDur)
        path = simlen(path, TAmp, TDur)
        length = len(path["fix"]["dur"])
        if length == prev_length:
            return path
        else:
            prev_length = length


def cal_vectordifferences(path1, path2):
    """Create matrix of vector-length differences of all vector pairs

    Create M, a Matrix with all possible saccade-length differences between
    saccade pairs.

    :param: path1, path2: dicts, vector-based scanpath representations

    :return: M: array-like
        Matrix of vector length differences

    """
    # take length in x and y direction of both scanpaths
    x1 = np.asarray(path1["sac"]["lenx"])
    x2 = np.asarray(path2["sac"]["lenx"])
    y1 = np.asarray(path1["sac"]["leny"])
    y2 = np.asarray(path2["sac"]["leny"])
    # initialize empty list for rows, will become matrix to store sacc-length
    # pairings
    rows = []
    # calculate saccade length differences, vectorized
    for i in range(0, len(x1)):
        x_diff = abs(x1[i] * np.ones(len(x2)) - x2)
        y_diff = abs(y1[i] * np.ones(len(y2)) - y2)
        # calc final length from x and y lengths, append, stack into matrix M
        rows.append(np.asarray(np.sqrt(x_diff ** 2 + y_diff ** 2)))
    M = np.vstack(rows)
    return M


def createdirectedgraph(scanpath_dim, M, M_assignment):
    """Create a directed graph:
    The data structure of the result is a nested dictionary such as
    weightedGraph = {0 : {1:259.55, 15:48.19, 16:351.95},
    1 : {2:249.354, 16:351.951, 17:108.97},
    2 : {3:553.30, 17:108.97, 18:341.78}, ...}

    It defines the possible nodes to reach from a particular node, and the weight that
    is associated with the path to each of the possible nodes.

    :param: scanpath_dim: list, shape of matrix M
    :param: M: array-like, matrix of vector length differences
    :param: M_assignment: array-like, Matrix, arranged with values from 0 to number of entries in M

    :return: weighted graph: dict, Dictionary within a dictionary pairing weights (distances) with
            node-pairings

    """
    rows=[]
    cols = []
    weight = []
    # rows = np.zeros([1,1])
    # cols = np.zeros([1,1])
    # weight = np.zeros([1,1])

    # loop through every node rowwise
    for i in range(0, scanpath_dim[0]):
        # loop through every node columnwise
        for j in range(0, scanpath_dim[1]):
            currentNode = i * scanpath_dim[1] + j
            # if in the last (bottom) row, only go right
            if (i == scanpath_dim[0] - 1) & (j < scanpath_dim[1] - 1):
                rows.append(currentNode)
                cols.append(currentNode + 1)
                weight.append(M[i, j + 1])

            # if in the last (rightmost) column, only go down
            elif (i < scanpath_dim[0] - 1) & (j == scanpath_dim[1] - 1):
                rows.append(currentNode)
                cols.append(currentNode + scanpath_dim[1])
                weight.append(M[i + 1, j])

            # if in the last (bottom-right) vertex, do not move any further
            elif (i == scanpath_dim[0] - 1) & (j == scanpath_dim[1] - 1):
                rows.append(currentNode)
                cols.append(currentNode)
                weight.append(0)

            # anywhere else, move right, down and down-right.
            else:
                # np.concatenate()
                rows.append(currentNode)
                rows.append(currentNode)
                rows.append(currentNode)
                cols.append(currentNode + 1)
                cols.append(currentNode + scanpath_dim[1])
                cols.append(currentNode + scanpath_dim[1] + 1)
                weight.append(M[i, j + 1])
                weight.append(M[i + 1, j])
                weight.append(M[i + 1, j + 1])

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    weight = np.asarray(weight)
    numVert = scanpath_dim[0] * scanpath_dim[1]
    return numVert, rows, cols, weight


def dijkstra(numVert, rows, cols, data, start, end):
    """
    Dijkstra algorithm:
    Use dijkstra's algorithm from the scipy module to find the shortest path through a directed
    graph (weightedGraph) from start to end.

    :param: weightedGraph: dict, dictionary within a dictionary pairing weights (distances) with
            node-pairings
    :param: start: int, starting point of path, should be 0
    :param: end: int, end point of path, should be (n, m) of Matrix M

    :return: path: array, indices of the shortest path, i.e. best-fitting saccade pairs
    :return: dist: float, sum of weights
    """
    # Create a scipy csr matrix from the rows,cols and append. This saves on memory.
    arrayWeightedGraph = (
        sp.coo_matrix((data, (rows, cols)), shape=(numVert, numVert))
    ).tocsr()

    # Run scipy's dijkstra and get the distance matrix and predecessors
    dist_matrix, predecessors = sp.csgraph.dijkstra(
        csgraph=arrayWeightedGraph, directed=True, indices=0, return_predecessors=True
    )

    # Backtrack thru the predecessors to get the reverse path
    path = [end]
    dist = float(dist_matrix[end])
    # If the predecessor is -9999, that means the index has no parent and thus we have reached the start node
    while end != -9999:
        path.append(predecessors[end])
        end = predecessors[end]

    # Return the path in ascending order and return the distance
    return path[-2::-1], dist


def cal_angulardifference(data1, data2, path, M_assignment):
    """Calculate angular similarity of two scanpaths:

    :param: data1: dict; contains vector-based scanpath representation of the
        first scanpath
    :param: data2: dict, contains vector-based scanpath representation of the
        second scanpath
    :param: path: array,
        indices for the best-fitting saccade pairings between scanpaths
    :param: M_assignment: array-like, Matrix arranged with values from 0 to number of entries in
        M, the matrix of vector length similarities

    :return: anglediff: array of floats, angular differences between pairs of saccades
        of two scanpaths

    """
    # get the angle between saccades from the scanpaths
    theta1 = data1["sac"]["theta"]
    theta2 = data2["sac"]["theta"]
    # initialize list to hold individual angle differences
    anglediff = []
    # calculate angular differences between the saccades along specified path
    for p in path:
        # which saccade indices correspond to path?
        i, j = np.where(M_assignment == p)
        # extract the angle
        spT = [theta1[i.item()], theta2[j.item()]]
        for t in range(0, len(spT)):
            # get results in range -pi, pi
            if spT[t] < 0:
                spT[t] = math.pi + (math.pi + spT[t])
        spT = abs(spT[0] - spT[1])
        if spT > math.pi:
            spT = 2 * math.pi - spT
        anglediff.append(spT)
    return anglediff


def cal_durationdifference(data1, data2, path, M_assignment):
    """Calculate similarity of two scanpaths fixation durations.

    :param: data1: array-like
        dict, contains vector-based scanpath representation of the
        first scanpath
    :param: data2: array-like
        dict, contains vector-based scanpath representation of the
        second scanpath
    :param: path: array
        indices for the best-fitting saccade pairings between scanpaths
    :param: M_assignment: array-like
         Matrix, arranged with values from 0 to number of entries in M, the
         matrix of vector length similarities

    :return: durdiff: array of floats,
        array of fixation duration differences between pairs of saccades from
        two scanpaths

    """
    # get the duration of fixations in the scanpath
    dur1 = data1["fix"]["dur"]
    dur2 = data2["fix"]["dur"]
    # initialize list to hold individual duration differences
    durdiff = []
    # calculation fixation duration differences between saccades along path
    for p in path:
        # which saccade indices correspond to path?
        i, j = np.where(M_assignment == p)
        maxlist = [dur1[i.item()], dur2[j.item()]]
        # compute abs. duration diff, normalize by largest duration in pair
        if(max(maxlist)==0):
            durdiff.append(0)
        else:
            durdiff.append(abs(dur1[i.item()] - dur2[j.item()]) / abs(max(maxlist)))
    return durdiff


def cal_lengthdifference(data1, data2, path, M_assignment):
    """Calculate length similarity of two scanpaths.

    :param: data1: array-like
        dict, contains vector-based scanpath representation of the
        first scanpath
    :param: data2: array-like
        dict, contains vector-based scanpath representation of the
        second scanpath
    :param: path: array
        indices for the best-fitting saccade pairings between scanpaths
    :param: M_assignment: array-like
         Matrix, arranged with values from 0 to number of entries in M, the
         matrix of vector length similarities

    :return: lendiff: array of floats
        array of length difference between pairs of saccades of two scanpaths

    """
    # get the saccade lengths rho
    len1 = np.asarray(data1["sac"]["rho"])
    len2 = np.asarray(data2["sac"]["rho"])
    # initialize list to hold individual length differences
    lendiff = []
    # calculate length differences between saccades along path
    for p in path:
        i, j = np.where(M_assignment == p)
        lendiff.append(abs(len1[i] - len2[j]))
    return lendiff


def cal_positiondifference(data1, data2, path, M_assignment):
    """Calculate position similarity of two scanpaths.

    :param: data1: array-like
        dict, contains vector-based scanpath representation of the
        first scanpath
    :param: data2: array-like
        dict, contains vector-based scanpath representation of the
        second scanpath
    :param: path: array
        indices for the best-fitting saccade pairings between scanpaths
    :param: M_assignment: array-like
         Matrix, arranged with values from 0 to number of entries in M, the
         matrix of vector length similarities

    :return: posdiff: array of floats
        array of position differences between pairs of saccades
        of two scanpaths

    """
    # get the x and y coordinates of points between saccades
    x1 = np.asarray(data1["sac"]["x"])
    x2 = np.asarray(data2["sac"]["x"])
    y1 = np.asarray(data1["sac"]["y"])
    y2 = np.asarray(data2["sac"]["y"])
    # initialize list to hold individual position differences
    posdiff = []
    # calculate position differences along path
    for p in path:
        i, j = np.where(M_assignment == p)
        posdiff.append(
            math.sqrt(
                (x1[i.item()] - x2[j.item()]) ** 2 + (y1[i.item()] - y2[j.item()]) ** 2
            )
        )
    return posdiff


def cal_vectordifferencealongpath(data1, data2, path, M_assignment):
    """Calculate vector similarity of two scanpaths.

    :param: data1: array-like
        dict, contains vector-based scanpath representation of the
        first scanpath
    :param: data2: array-like
        dict, contains vector-based scanpath representation of the
        second scanpath
    :param: path: array-like
        array of indices for the best-fitting saccade pairings between scan-
        paths
    :param: M_assignment: array-like
         Matrix, arranged with values from 0 to number of entries in M, the
         matrix of vector length similarities

    :return: vectordiff: array of floats
            array of vector differences between pairs of saccades of two scanpaths

    """
    # get the saccade lengths in x and y direction of both scanpaths
    x1 = np.asarray(data1["sac"]["lenx"])
    x2 = np.asarray(data2["sac"]["lenx"])
    y1 = np.asarray(data1["sac"]["leny"])
    y2 = np.asarray(data2["sac"]["leny"])
    # initialize list to hold individual vector differences
    vectordiff = []
    # calculate vector differences along path
    # TODO look at this again, should be possible simpler
    for p in path:
        i, j = np.where(M_assignment == p)
        vectordiff.append(
            np.sqrt(
                (x1[i.item()] - x2[j.item()]) ** 2 + (y1[i.item()] - y2[j.item()]) ** 2
            )
        )
    return vectordiff


def getunnormalised(data1, data2, path, M_assignment):
    """Calculate unnormalised similarity measures.

    Calls the five functions to create unnormalised similarity measures for
    each of the five similarity dimensions. Takes the median of the resulting
    similarity values per array.

    :param: data1: array-like
        dict, contains vector-based scanpath representation of the
        first scanpath
    :param: data2: array-like
        dict, contains vector-based scanpath representation of the
        second scanpath
    :param: path: array
        indices for the best-fitting saccade pairings between scanpaths
    :param: M_assignment: array-like
         Matrix, arranged with values from 0 to number of entries in M, the
         matrix of vector length similarities

    :return: unnormalised: array
        array of unnormalised similarity measures on five dimensions

    >>> unorm_res = getunnormalised(scanpath_rep1, scanpath_rep2, path, M_assignment)
    """
    return [
        np.median(fx(data1, data2, path, M_assignment))
        for fx in (
            cal_vectordifferencealongpath,
            cal_angulardifference,
            cal_lengthdifference,
            cal_positiondifference,
            cal_durationdifference,
        )
    ]


def normaliseresults(unnormalised, screensize):
    """Normalize similarity measures.

    Vector similarity is normalised against two times screen diagonal,
    the maximum theoretical distance.
    Direction similarity is normalised against pi.
    Length Similarity is normalised against screen diagonal.
    Position Similarity and Duration Similarity are already normalised.

    :param: unnormalised: array
        array of unnormalised similarity measures,
        output of getunnormalised()

    :return: normalresults: array
        array of normalised similarity measures

    >>> normal_res = normaliseresults(unnormalised, screensize)
    """
    # normalize vector similarity against two times screen diagonal, the maximum
    # theoretical distance
    VectorSimilarity = 1 - unnormalised[0] / (
        2 * math.sqrt(screensize[0] ** 2 + screensize[1] ** 2)
    )
    # normalize against pi
    DirectionSimilarity = 1 - unnormalised[1] / math.pi
    # normalize against screen diagonal
    LengthSimilarity = 1 - unnormalised[2] / math.sqrt(
        screensize[0] ** 2 + screensize[1] ** 2
    )
    PositionSimilarity = 1 - unnormalised[3] / math.sqrt(
        screensize[0] ** 2 + screensize[1] ** 2
    )
    # no normalisazion necessary, already done
    DurationSimilarity = 1 - unnormalised[4]
    normalresults = [
        VectorSimilarity,
        DirectionSimilarity,
        LengthSimilarity,
        PositionSimilarity,
        DurationSimilarity,
    ]
    return normalresults


def docomparison(
    fixation_vectors1,
    fixation_vectors2,
    screensize,
    grouping=False,
    TDir=0.0,
    TDur=0.0,
    TAmp=0.0,
):
    """Compare two scanpaths on five similarity dimensions.


    :param: fixation_vectors1: array-like n x 3 fixation vector of one scanpath
    :param: fixation_vectors2: array-like n x 3 fixation vector of one scanpath
    :param: screensize: list, screen dimensions in px.
    :param: grouping: boolean, if True, simplification is performed based on thresholds TAmp,
        TDir, and TDur. Default: False
    :param: TDir: float, Direction threshold, angle in degrees. Default: 0.0
    :param: TDur: float,  Duration threshold, duration in seconds. Default: 0.0
    :param: TAmp: float, Amplitude threshold, length in px. Default: 0.0

    :return: scanpathcomparisons: array
        array of 5 scanpath similarity measures. Vector (Shape), Direction
        (Angle), Length, Position, and Duration. 1 means absolute similarity, 0 means
        lowest similarity possible.

    # >>> results = docomparison(fix_1, fix_2, screensize = [1280, 720], grouping = True, TDir = 45.0, TDur = 0.05, TAmp = 150)
    # >>> print(results)
    # >>> [[0.95075847681364678, 0.95637548674423822, 0.94082367355291008, 0.94491164030498609, 0.78260869565217384]]
    """
    # check if fixation vectors/scanpaths are long enough
    # if (len(fixation_vectors1) >= 3) & (len(fixation_vectors2) >= 3):
    if (len(fixation_vectors1['duration']) >= 3) & (len(fixation_vectors2['duration']) >= 3):
        # get the data into a geometric representation
        path1 = gen_scanpath_structure(fixation_vectors1)
        path2 = gen_scanpath_structure(fixation_vectors2)
        if grouping:
            # simplify the data
            path1 = simplify_scanpath(path1, TAmp, TDir, TDur)
            path2 = simplify_scanpath(path2, TAmp, TDir, TDur)
        # create M, a matrix of all vector pairings length differences (weights)
        if( len(path2['sac']['lenx'])==0 or len(path1['sac']['lenx'])==0 ):
            return np.repeat(0,5)
        if (len(path2['fix']['dur']) == 0 or len(path1['fix']['dur']) == 0):
            return np.repeat(0,5)

        M = cal_vectordifferences(path1, path2)
        # initialize a matrix of size M for a matrix of nodes
        scanpath_dim = np.shape(M)
        M_assignment = np.arange(scanpath_dim[0] * scanpath_dim[1]).reshape(
            scanpath_dim[0], scanpath_dim[1]
        )
        # create a weighted graph of all possible connections per Node, and their weight
        numVert, rows, cols, weight = createdirectedgraph(scanpath_dim, M, M_assignment)
        # find the shortest path (= lowest sum of weights) through the graph using scipy dijkstra
        path, dist = dijkstra(
            numVert, rows, cols, weight, 0, scanpath_dim[0] * scanpath_dim[1] - 1
        )

        # compute similarities on aligned scanpaths and normalize them
        unnormalised = getunnormalised(path1, path2, path, M_assignment)
        normal = normaliseresults(unnormalised, screensize)
        return normal
    # return nan as result if at least one scanpath it too short
    else:
        return np.repeat(0, 5)

def gazeSim(gazeInfo1,gazeInfo2):
    # similarity hold the list=[shape,length,direction,position,duration]
    res=docomparison(gazeInfo1, gazeInfo2, screensize=[1920, 1080], grouping=True, TDir=45.0,
                   TDur=100, TAmp=160)
    score={'shape':res[0],"len":res[1],"direct":res[2],"pos":res[3],"duration":res[4]}
    return score

def gazeFilter(gazeInfo,ratio=0.05):
    thresh =ratio*800
    g = []
    x_list=[]
    y_list=[]
    if(len(gazeInfo['gaze'])<2):
        gazeInfo['gaze']=[]
        gazeInfo['duration'] = []
        gazeInfo['x'] =[]
        gazeInfo['y'] = []
        return gazeInfo
    x_prev, y_prev = gazeInfo['gaze'][0]
    t = 1
    duration=[]
    n_iter=1
    period_time=0
    gazeInfo['duration']=np.ones(len(gazeInfo['gaze']))
    for (x, y) in gazeInfo['gaze'][1:]:
        timepoint=gazeInfo['duration'][n_iter]
        period_time += timepoint
        if abs(x - x_prev) < thresh and abs(y - y_prev) < thresh:
            t += 1
        else:
            g.append((x_prev, y_prev, 1))
            x_list.append(x_prev)
            y_list.append(y_prev)
            duration.append(period_time)
            x_prev, y_prev = x, y
            period_time=0
            t = 1
        n_iter+=1
    gazeInfo['seq']=g
    gazeInfo['duration']=duration
    gazeInfo['x']=x_list
    gazeInfo['y']=y_list
    return gazeInfo



if __name__ == "__main__":
    gazeDataRoot='../data/INBreast/gaze/'
    savePath='../data/INBreast/gaze_multimatch/'
    namelist=glob.glob('../data/INBreast/gaze/*.json')
    namelist=namelist['train']
    trainSize=len(namelist)


    affinity=np.zeros([trainSize,trainSize,5],dtype=np.float64)

    for idx in tqdm(range(trainSize)):
        gazeInfo1=json.load(open(namelist[idx]))
        gazeInfo1=gazeFilter(gazeInfo1.copy())
        for jdx in tqdm(range(idx,trainSize)):
            gazeInfo2=json.load(open(namelist[jdx]))
            gazeInfo2=gazeFilter(gazeInfo2.copy())
            score=docomparison(gazeInfo1, gazeInfo2, screensize=[1920, 1080], grouping=True, TDir=45.0,
                   TDur=100, TAmp=160)
            affinity[idx][jdx]=score
            affinity[jdx][idx]=score

    np.save('./relation_match', affinity)

  
