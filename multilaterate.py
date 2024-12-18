''' Draw loci corresponding to radio transmission multilateration.

This program plots loci of possible transmitter locations for the scenario 
where there are radio towers
at known locations and a transmitter at an unknown location. The radio
towers accurately timestamp when they receive the transmission, allowing
time difference of arrival (TDOA) to be determined. This forms a
multilateration problem, producing n-1 loci where n is the number
of towers.
Only the 2-dimensional case is considered. It is assumed that the effect
on TDOA fron the vertical component of the transmission path is negligible.
For example, a path that is 5km horizontally and 500m vertically is
in total 5.025km ((5**2 + 0.5**2)**0.5). Depending on clock noise this could
be considered negligible.

'''

import numpy as np
import math
from scipy.optimize import least_squares


def get_locus(tower_1, tower_2, time_1, time_2, v, delta_d, max_d):
    ''' Return a locus in x, y given two towers and their recieve times.

    Given two towers at locations tower_1 and tower_2, a message transmitted
    at some arbitrary time at location (x_t, y_t), and the times at which
    the towers received the transmission, the set of possible
    locations of the transmission is defined by the locus of the intersection
    of two circles with one circle around each tower and the difference in
    radius of the circles defined by the difference in receive tiemes
    of the transmission and the propogation speed of the transmission.

    Args:
        tower_1 (tuple): (x, y) of one tower.
        tower_2 (tuple): (x, y) of other tower.
        time_1 (float): Transmission recieve time at tower_1.
        time_2 (float): Transmission recieve time at tower_2.
        v (int): Speed of transmission propogation.
        delta_d (int): Metre increments to radii of circles when generating
            locus of circle intersection.
        max_d (int): Max distance a transmission will be from the tower that
            first received the transmission. This puts an upper bound on the
            radii of the circle, thus limiting the size of the locus to be
            near the towers.

    Returns
        list of form [x, y], with:
                x: list of x values of locus.
                y: list of y values of locus.
    '''
    # two lines, x0/y0 and x1/y1 corresponding to the two intersections of the
    # circles. These will be concateneated at the end to form a single line.
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    # The radii have constant difference of t_delta_d. "time delta difference"
    t_delta_d = abs(time_1 - time_2) * v

    # Determine which tower received the transmission first.
    if(time_1 < time_2):
        circle1 = (tower_1[0], tower_1[1], 0)
        circle2 = (tower_2[0], tower_2[1], t_delta_d)
    else:
        circle1 = (tower_2[0], tower_2[1], 0)
        circle2 = (tower_1[0], tower_1[1], t_delta_d)

    # Iterate over all potential radii.
    for _ in range(int(max_d/delta_d)):
        intersect = circle_intersection(circle1, circle2)
        if(intersect is not None):
            x0.append(intersect[0][0])
            x1.append(intersect[1][0])
            y0.append(intersect[0][1])
            y1.append(intersect[1][1])

        circle1 = (circle1[0], circle1[1], circle1[2]+delta_d)
        circle2 = (circle2[0], circle2[1], circle2[2]+delta_d)

    # Reverse so the concatenated locus is continous. Could reverse only
    # x1/y1 instead if you wanted.
    x0 = list(reversed(x0))
    y0 = list(reversed(y0))

    # Concatenate
    x = x0 + x1
    y = y0 + y1

    return [x, y]


def get_loci(rec_times, towers, v, delta_d, max_d):
    ''' Return a set of loci on which a transmission may have occurred.

    Args:
        rec_times (np.array 1D): The times at which the towers recieved
            the transmission, in seconds. Element i corresponds to tower i.
        towers (np.array 2D): Locations of towers. Tower i is located at
            (x, y) = (towers[i][0], towers[i][1])
        v (int): Speed of transmission propogation.
        delta_d (int): Metre increments to radii of circles when generating
            locus of circle intersection.
        max_d (int): Max distance a transmission will be from the tower that
            first received the transmission. This puts an upper bound on the
            radii of the circle, thus limiting the size of the locus to be
            near the towers.

    Returns
        list of tuples, where each tuple contains a list of x and a list of
            y elements.
        '''

    if(rec_times.shape[0] == 0):
        return [] # return no loci
    
    loci = []

    # Tower that receives the transmission first.
    order = np.argsort(rec_times)
    first_tower = order[0]
    #print("helll", args,  temp, rec_times, first_tower, second_tower)
    # Iterate over all other towers.

    for k in range(len(order)-1):
        for j in [x for x in range(towers.shape[0]) if x not in order[:k]]:
            # print('tower', str(first_tower), 'to', str(j))
            locus = get_locus(tower_1=(towers[order[k]][0],
                                       towers[order[k]][1]),
                              tower_2=(towers[j][0], towers[j][1]),
                              time_1=rec_times[order[k]],
                              time_2=rec_times[j],
                              v=v, delta_d=delta_d, max_d=max_d)
            # Sometimes empty locus is produced depending on geometry of the
            # situation. Discard these.
            if(len(locus[0]) > 0):
                loci.append(locus)
    return loci


def circle_intersection(circle1, circle2):
    ''' Calculate intersection points of two circles.
    from https://gist.github.com/xaedes/974535e71009fa8f090e

    Args:
        circle1: tuple(x,y,radius)
        circle2: tuple(x,y,radius)

    Returns
        tuple of intersection points (which are (x,y) tuple)

    >>> circle_intersection((-0.5, 0, 1), (0.5, 0, 1))
    ((0.0, -0.8660254037844386), (0.0, 0.8660254037844386))
    >>> circle_intersection((-1, 0, 1), (1, 0, 1))
    ((0.0, 0.0), (0.0, 0.0))

    '''
    x1,y1,r1 = circle1
    x2,y2,r2 = circle2
    # http://stackoverflow.com/a/3349134/798588
    # d is euclidean distance between circle centres
    dx,dy = x2-x1,y2-y1
    d = math.sqrt(dx*dx+dy*dy)
    if d > r1+r2:
        # print('No solutions, the circles are separate.')
        return None # No solutions, the circles are separate.
    elif d < abs(r1-r2):
        # No solutions because one circle is contained within the other
        # print('No solutions because one circle is contained within the other')
        return None
    elif d == 0 and r1 == r2:
        # Circles are coincident - infinite number of solutions.
        # print('Circles are coincident - infinite number of solutions.')
        return None

    a = (r1*r1-r2*r2+d*d)/(2*d)
    h = math.sqrt(r1*r1-a*a)
    xm = x1 + a*dx/d
    ym = y1 + a*dy/d
    xs1 = xm + h*dy/d
    xs2 = xm - h*dy/d
    ys1 = ym - h*dx/d
    ys2 = ym + h*dx/d

    return ((xs1,ys1),(xs2,ys2))

def tdoaEstimation(rec_times, towers, x_init):
    v = 299792458 * 1 # meters/s
    c = np.argmin(rec_times)
    p_c = np.expand_dims(towers[c], axis=0)
    t_c = rec_times[c]
    # Remove the c tower to allow for vectorization.
    all_p_i = np.delete(towers, c, axis=0)
    all_t_i = np.delete(rec_times, c, axis=0)

    next_c = np.argmin(rec_times)
    next_p_c = np.expand_dims(towers[c], axis=0)
    next_t_c = rec_times[c]
    next_all_p_i = np.delete(towers, c, axis=0)
    next_all_t_i = np.delete(rec_times, c, axis=0)


#    print('hel', all_t_i, all_p_i)

    def eval_solution(x):
        """ x is 2 element array of x, y of the transmitter"""
        a = (
              np.linalg.norm(x - p_c, axis=1)
            - np.linalg.norm(x - all_p_i, axis=1) 
            + v*(all_t_i - t_c) 
        )
        #print(a)

        return a

    def eval_solution2(x):
        """ x is 2 element array of x, y of the transmitter"""
        firstSet = (np.linalg.norm(x - p_c, axis=1) - np.linalg.norm(x - all_p_i, axis=1) + v*(all_t_i - t_c) )
        secondSet = (np.linalg.norm(x - next_p_c, axis=1) - np.linalg.norm(x - next_all_p_i, axis=1) + v*((next_all_t_i - next_t_c)) )
        return np.concatenate((firstSet, secondSet))


    # Initial guess.

    # Find a value of x such that eval_solution is minimized.
    # Remember the receive times have error added to them: rec_time_noise_stdd.
    bounds = ([ 0, 0], [ 10, 2.5])
    alpha = 1
    try:
        res = least_squares(eval_solution2, x_init, loss="linear", bounds=bounds)
    except:
        res = None # should cause error in the plot but whatever
    finally:
        res.x = alpha * res.x + (1-alpha) * x_init
    return res

def plotLoci(ax, rec_times, towers, v, delta_d, max_d):
    loci = get_loci(rec_times, towers, v, delta_d, max_d)

    for i in range(towers.shape[0]):
        x = towers[i][0]
        y = towers[i][1]
        ax.scatter(x, y)
        ax.annotate('Anchor '+str(i), (x, y))

    for locus in loci:
        ax.plot(locus[0], locus[1], linewidth=1.0)


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2),
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

