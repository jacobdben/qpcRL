import sys
import numpy as np
import copy

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import KDTree
import kwant

from poisson.discrete import repeated_values
from poisson.continuous import shapes

from matplotlib import pyplot as plt

# code for generating sites in the exterior of a kwant region
# supposing the underlying lattice in known
# uses repeated_values !!!

def mimic_HK_call(hoppingkind, builder):
    '''function mimicing the call of hoppingking without 
        the limition to stay inside the builder'''
    delta = hoppingkind.delta
    family_a = hoppingkind.family_a
    family_b = hoppingkind.family_b
    H = builder.H

    for a in H:
        if a.family != family_a:
            continue
        b = kwant.builder.Site(family_b, a.tag - delta, True)
        yield a, b

def fill_border(lat, builder):
    '''get the coordinates of a (hyper)line surrounding builder sites matching a lattice'''
    i = 1
    nbs = lat.neighbors(i)
    while nbs[0].family_a != nbs[0].family_b and i < 10:
        i += 1
        nbs = lat.neighbors(i)
    
    nbsm = [kwant.HoppingKind(-nb.delta, nb.family_a, nb.family_b) for nb in nbs]
    
    extended_sites = []
    for nb in nbs:
        extended_sites += [p.pos for pair in mimic_HK_call(nb, builder) for p in pair]
    for nb in nbsm:
        extended_sites += [p.pos for pair in mimic_HK_call(nb, builder) for p in pair]
        
    return repeated_values.remove_from_grid(np.array(extended_sites))[0]

def make_surround(datas, distance):
    '''makes function(x) that is true if
        min(distance between x and datas) < distance
        (using KDTree query)'''
    if not isinstance(datas, KDTree):
        try:
            tree = KDTree(datas)
        except:
            print('Input type non recognized') 
            raise
    else:
        tree = datas
    nsites = tree.data.shape[0]
    def func(x):
        x = np.asarray(x)
        dists, _ = tree.query(x, nsites)
        mindist = np.min(dists, axis=-1)
        msk = mindist < distance
        return msk
    return func

def fill_with_lattice(lat, start, shape, distance=0):
    '''Generate list of points matching a lattice inside a shape
    
    Parameters:
    -----------
    lattice: kwant.lattice
        callable kwant lattice
    start: tuple, array, list
        coordinates of the center for lattice generation
    shape: kwant.Builder or numpy.ndarray or callable
        the shape to fill with lattice
        if Builder, the shape will be generated from the
            sites in the Builder
        if numpy.ndarray, the shape will be generated from
            the element of the array
    distance: number (opt, defaults to 0)
        (only if shape is not callable)
        distance at which ti surround the shape
        if shape is a Builder and distance is 0:
            fill_border is called: add the minimal number of sites
            surrounding a shape to get regular voronoi cells for all sites
            
    Returns:
    --------
    sites: np.ndarray
        the coordinates of the new sites
    '''
    syst = kwant.Builder()
    if isinstance(shape, kwant.Builder):
        if distance == 0:
            return fill_border(lat, shape)
        else:
            shapesites = np.array([s.pos for s in shape.sites()])
            shape = make_surround(shapesites, distance)
    elif isinstance(shape, np.ndarray):
        shape = make_surround(shapesites, distance)
    elif callable(shape):
        pass
    else:
        raise ValueError
    syst[lat.shape(shape, start)] = 1
    sites = np.array([s.pos for s in syst.sites()])
    return sites

###########################################################################"

def tests():
    
    # tests
    def make_cross():
        W = 3
        L = 20

        # Define the scattering region
        eps=0.1
        rect = [(-L/2-eps, -W/2-eps), (L/2+eps, -W/2-eps), (L/2+eps, W/2+eps), (-L/2-eps, W/2+eps)]
        rect1 = shapes.Delaunay(rect)
        rect2 = copy.deepcopy(rect1)
        rect3 = copy.deepcopy(rect1)
        rect2.rotate(np.pi/3)
        rect3.rotate(7 * np.pi/4)

        cross = rect1 + rect2 + rect3
        return cross

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    def make_square(shape):
        lat = kwant.lattice.square(1)
        syst = kwant.Builder()
        syst[lat.shape(shape, (0, 0))] = 1
        return syst

    def make_honeycomb(shape):
        graphene = kwant.lattice.honeycomb()
        syst = kwant.Builder()
        syst[graphene.shape(shape, (0, 0))] = 1
        return syst

    cross = make_cross()
    
    print('Visually check that all points of the sytem (X) are inside a'
          +' Voronoi cell of the right shape')
    
    ##### Surround
    # square
    sq_syst = make_square(cross)
    sq_sites = np.array([s.pos for s in sq_syst.sites()])
    sq_extended_sites = fill_with_lattice(kwant.lattice.square(1), (0,0), sq_syst)

    vor = Voronoi(sq_extended_sites)

    kwant.plotter.plot(sq_syst, show=False)
    plt.show()

    plt.scatter(sq_extended_sites[:, 0], sq_extended_sites[:, 1], s=100, marker='o', c='r')
    plt.scatter(sq_sites[:, 0], sq_sites[:, 1], s=100, marker='x')
    plt.show()

    ax = voronoi_plot_2d(vor)
    plt.scatter(sq_sites[:, 0], sq_sites[:, 1], s=100, marker='x', c='r')
    plt.show()

    #honeycomb
    syst_honey = make_honeycomb(cross)
    hc_sites = np.array([s.pos for s in syst_honey.sites()])
    hc_extended_sites = fill_with_lattice(kwant.lattice.honeycomb(1), (0,0), syst_honey)
    vor = Voronoi(hc_extended_sites)

    a, b = kwant.lattice.honeycomb().sublattices
    def family_colors(site):
        return 0 if site.family == a else 1

    # Plot the closed system without leads.
    kwant.plot(syst_honey, site_color=family_colors, site_lw=0.2, 
               colorbar=False)

    plt.scatter(hc_extended_sites[:, 0], hc_extended_sites[:, 1], s=100, marker='o', c='r')
    plt.scatter(hc_sites[:, 0], hc_sites[:, 1], s=100, marker='x')
    plt.show()

    f, ax = plt.subplots(1, figsize=(8, 8))
    ax = voronoi_plot_2d(vor, ax=ax)
    plt.scatter(hc_sites[:, 0], hc_sites[:, 1], s=100, marker='x', c='r')
    plt.show()

    ###### Fill
   
    bigrect = shapes.Rectangle(length=(25, 25), corner=(-12.5, -12.5))
    rectfill_sites = fill_with_lattice(kwant.lattice.honeycomb(1), (0,0), shape=bigrect, distance=3)
    builderfill_sites = fill_with_lattice(kwant.lattice.honeycomb(1), (0,0), shape=syst_honey, distance=3)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(hc_sites[:, 0], hc_sites[:, 1], s=100, alpha=0.5)
    plt.scatter(rectfill_sites[:, 0], rectfill_sites[:, 1], s=100, marker='x', c='r')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(hc_sites[:, 0], hc_sites[:, 1], s=100, alpha=0.5)
    plt.scatter(builderfill_sites[:, 0], builderfill_sites[:, 1], s=100, marker='x', c='r')
    plt.show()
