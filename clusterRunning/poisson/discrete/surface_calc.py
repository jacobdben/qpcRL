'''
   Calculate the surface for polygone.
   Author : Pacome
'''


__all__ = ['surface_vectorized']


import numpy as np
from scipy import linalg as scilinalg

# levi civita for einsum. should be defined somewhere
EIJK = np.zeros((3, 3, 3))
EIJK[0, 1, 2] = EIJK[1, 2, 0] = EIJK[2, 0, 1] = 1
EIJK[0, 2, 1] = EIJK[2, 1, 0] = EIJK[1, 0, 2] = -1


def get_depth(my_list):
    '''get the depth of list or numpy ndarray, starts at 1 for non empty list'''
    if isinstance(my_list, list) or isinstance(my_list, np.ndarray):
        return 1 + max(get_depth(item) for item in my_list)
    else:
        return 0


def _surface(points):
    '''get the surface of a polygon in 2 or 3d.

    Remark:
    -couldnt manage to speed up using tiny array
    -the if statements adds around 1 micro second'''

    # check if vectorial
    np.asarray(points).shape

    N, ndim = np.asarray(points).shape

    assert ndim == 2 or ndim == 3, 'points should have 2 or 3 coordinates, here {}'.format(
        ndim)

    if ndim is 3:  # costs some time but shoud speed up if polygons in  xy, xz, yz plane
        cstxyz = np.max(points, axis=1) == np.min(points, axis=1)  # slow !!!
        if cstxyz.sum() == 1:
            points = points[:, np.invert(cstxyz)]
            ndim = 2

    if ndim is 2:
        x, y = points.T
        ret = 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    elif ndim is 3:
        xp = np.repeat(points, N - 2, axis=0)
        xpp1 = np.roll(xp, -(2*N - 5), axis=0)
        xpp2 = np.roll(xp, -(3*N - 7), axis=0)

        x1 = xpp1 - xp
        x2 = xpp2 - xp
        # vector product (faster)
        cr = np.einsum('ijk,aj,ak->ai', EIJK, x1, x2)

        mat = np.array([cr, x1, x2]).swapaxes(0, 1)

        # compute determinant and triangle area (faster)
        triareas = 0.5 * np.sqrt(np.einsum('ijk,li,lj,lk -> l', EIJK, mat[:, 0],
                                           mat[:, 1], mat[:, 2]))

        ret = triareas.sum() / len(points)

    return ret


def test_surface(func=_surface):
    ''' Test surface calculation '''
    x0 = np.array([0, 0, 1])
    x1 = np.array([0, 1, 1])
    x2 = np.array([1, 1, 1])
    x3 = np.array([1, 0, 1])
    sq3d = np.array([x0, x1, x2, x3])
    tri3d = np.array([x0, x1, x2])

    rands = np.random.rand(1000)
    for r in rands:
        np.testing.assert_allclose(func(sq3d * r), r**2)
        np.testing.assert_allclose(func(tri3d * r), r**2 / 2)
    print('3d tests passed')
    for r in rands:
        np.testing.assert_allclose(func(sq3d[:, :2] * r), r**2)
        np.testing.assert_allclose(func(tri3d[:, :2] * r), r**2 / 2)
    print('2d tests passed')

# for norm use
# (a.dot(a.T))**0.5  almost 2*faster than np.norm


def _surface_vectorized(mylist):
    '''vectorized surface calculation'''
    # assuming mylist is a list polygons with the same shape

    mylist = np.asarray(mylist)
    npoly, N, ndim = mylist.shape

    assert ndim is 2 or ndim is 3, 'points should have 2 or 3 coordinates, here {}'.format(
        ndim)

    if ndim is 2:
        x, y = mylist[:, :, 0], mylist[:, :, 1]

        rolx = np.roll(x, 1, axis=1)
        roly = np.roll(y, 1, axis=1)
        ret = 0.5 * np.abs(np.einsum('ij, ij->i', x, roly) - np.einsum(
            'ij, ij->i', y, rolx))

    if ndim is 3:
        xp = np.repeat(mylist, N - 2, axis=1)
        xpp1 = np.roll(xp, -(2*N - 5), axis=1)
        xpp2 = np.roll(xp, -(3*N - 7), axis=1)

        x1 = xpp1 - xp
        x2 = xpp2 - xp
        cr = np.einsum('ijk,alj,alk->ali', EIJK, x1,
                       x2)  # vector product (faster)

        mat = np.array([cr, x1, x2]).swapaxes(0, 1)

        # compute determinant and triangle area (faster)
        triareas = 0.5 * np.sqrt(np.einsum('ijk,mli,mlj,mlk -> ml', EIJK, mat[:, 0],
                                           mat[:, 1], mat[:, 2]))

        ret = triareas.sum(axis=1) / N

    return ret


def surface_vectorized(listpoly):
    '''vectorized surface calculation
    suppose that all polygon are either 2d or in 3d
    adds huge overhead to _surface_vecor, for sure can be faster
    '''

    if isinstance(listpoly, np.ndarray):
        depth = len(listpoly.shape)

    elif isinstance(listpoly, list):
        depth = get_depth(listpoly)

    else:
        raise TypeError

    ##############################
    # only 1 polygon
    if depth is 2:
        return _surface(listpoly)

    ##############################
    # more than 1 polygon
    elif depth is 3:

        # for the output
        all_surfs = np.zeros(len(listpoly))

        ndim = len(listpoly[0][0])

        # sort the polygons by shapes
        polysizes = np.array([len(poly) for poly in listpoly])  # slow !
        diffsizes = list(set(polysizes))  # faster than np.unique
        msks = np.array([polysizes == i for i in diffsizes])

        sorted_poly = [[a for i, a in enumerate(
            listpoly) if msk[i]] for msk in msks]  # might be slow

        # avoidable ? propably very few elements
        for i, polys in enumerate(sorted_poly):

            polys = np.asarray(polys)

            # check if in xy, xz, yz plane
            if ndim is 3:
                cstxyz = np.max((polys), axis=1) == np.min(
                    (polys), axis=1)  # slow !!!
                msk_2d = np.asarray(cstxyz.sum(axis=1), dtype=bool)  # slow !!!

                if msk_2d.sum() > 0.:  # at least 1 in the plane
                    tmpret = np.zeros(len(msk_2d))
                    points_2d = polys[msk_2d, :, :]
                    points_3d = polys[np.invert(msk_2d), :, :]
                    cstxyz = np.invert(cstxyz[msk_2d])
                    points_2d = np.asarray([(p[:, ms]) for p, ms in zip(
                        points_2d, cstxyz)])  # slow !!!

                    tmpret[msk_2d] = _surface_vectorized(points_2d)
                    tmpret[np.invert(msk_2d)] = _surface_vectorized(points_3d)

                    all_surfs[msks[i]] = tmpret[:]

                else:
                    all_surfs[msks[i]] = _surface_vectorized(polys)
            else:
                all_surfs[msks[i]] = _surface_vectorized(polys)
        return all_surfs
    else:
        raise TypeError('data type not understood, see doc, depth =', depth)


def test_surface_vectorized():
    ''' Test vectorized surface calculation '''

    x0 = np.array([0, 0, 1])
    x1 = np.array([0, 1, 1])
    x2 = np.array([1, 1, 1])
    x3 = np.array([1, 0, 1])
    sq3d = np.array([x0, x1, x2, x3])
    tri3d = np.array([x0, x1, x2])

    rands = np.random.rand(100)

    for r in rands:
        polys = [[sq3d * r] * 100 + [tri3d * r] * 100][0]
        results = [[r**2] * 100 + [r**2 / 2.] * 100][0]  # analytical surfaces
        rng_state = np.random.get_state()
        np.random.shuffle(polys)
        np.random.set_state(rng_state)
        np.random.shuffle(results)
        np.testing.assert_allclose(surface_vectorized(polys), results)
    print('3d tests passed')

    for r in rands:
        polys = [[sq3d[:, :2] * r] * 100 + [tri3d[:, :2] * r] * 100][0]
        results = [[r**2] * 100 + [r**2 / 2.] * 100][0]  # analytical surfaces
        rng_state = np.random.get_state()
        np.random.shuffle(polys)
        np.random.set_state(rng_state)
        np.random.shuffle(results)
        np.testing.assert_allclose(surface_vectorized(polys), results)
    print('2d tests passed')


#test_surface_vectorized
