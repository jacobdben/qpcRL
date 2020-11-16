'''
    Tooblox with functions sent to post_process
'''
import time

from scipy.spatial import qhull, cKDTree

import numpy as np

from poisson import discrete


def value_from_points(x, discret_val, mapping, firstneig,
                      tree, vector=True, interpolate=0, tolerance=1e-15):
    '''x a point or a list of points,
    discret_val a array to be evaluated on mesh points
    tree a cKDTree of the mesh points
    mapping an array that takes a point in vor_struct.point as index and gives
    the point's corresponding index in x and discret_val. For points in
    vor_struct.point no in x, the default value should be len(x).
    '''
    x = np.asarray(x)

    mesh_points = tree.data
    ndim = mesh_points.shape[1]

    mapping_inv = np.ones(len(mesh_points), dtype=int)
    mapping_inv[mapping] = np.arange(len(mapping))

    distances, nearpts = tree.query(x)

    if not interpolate:
        vector = True
    elif interpolate is True:
        interpolate = 1
    else:
        if not isinstance(interpolate, (int, float)):
            raise ValueError

    if np.all(distances == 0):

        return discret_val[mapping[nearpts]]

    elif len(x.shape) < 2:
        # 1 point

        if not interpolate:
            return discret_val[mapping[nearpts]]

        nnearest = firstneig[nearpts]
        nnearest_pt_diff = x - mesh_points[
                firstneig[nearpts]]
        nnearest_distances = np.sqrt(np.einsum('ij,ji->i', nnearest_pt_diff,
                                               nnearest_pt_diff.T))

        all_nearests = np.concatenate(([nearpts], nnearest))

        assert (mapping[all_nearests] >= len(discret_val)).any() == False, \
                ('Mapping, discretefunc and x point data are not '
                 + 'compatible with one another')

        all_vals = discret_val[mapping[all_nearests]]
        all_distances = np.concatenate(([distances], nnearest_distances))
        inv_all_distances = (1/all_distances)**interpolate
        return np.sum(inv_all_distances * all_vals)/inv_all_distances.sum()

    elif len(x.shape) == 2 and vector:
        # list of points
        # Second vectorized version
        vals = np.zeros(len(x))
        discret_val = np.asarray(discret_val)
        if not interpolate:
            return discret_val[nearpts]

#        t0 = time.time()

        # remove points where distance is zero
        near_zero_mask =  abs(distances) < tolerance
        vals[near_zero_mask] = discret_val[mapping[nearpts[near_zero_mask]]]
        mask_not_near_zero = np.delete(np.arange(len(distances)),
                                       np.arange(len(distances))[near_zero_mask])

        distances_WZ = distances[mask_not_near_zero]
        near_pts_Wz = nearpts[mask_not_near_zero]
        x_WZ = x[mask_not_near_zero]
#        t1 = time.time()

        nnearest = [firstneig[nearpt]
                    for nearpt in near_pts_Wz]
        nnearest_shape = np.zeros(len(nnearest), dtype=int)
        for pos, nner in enumerate(nnearest):
            nnearest_shape[pos] = len(nner)

        mapp_sorted = nnearest_shape.argsort()
        uniq_val, uniq_index = np.unique(nnearest_shape[mapp_sorted],
                                          return_index=True)

#        t2 = time.time()
        for pos in range(len(uniq_index)):

            if pos == (len(uniq_index) - 1):
                last_pos = len(nnearest)
            else:
                last_pos = uniq_index[pos + 1]

            nns = np.array([nnearest[mapp_sorted[i]]
                for i in range(uniq_index[pos], last_pos)])

            if nns.shape[1] != 0:
                msk = mapp_sorted[np.arange(uniq_index[pos],
                                            last_pos, dtype=int)]
                shape = uniq_val[pos]

                nnearest_pt = (mesh_points[nns.flat]).reshape(
                        [np.prod(nns.shape)] + [ndim])
                nnearest_pt_diff = ( np.repeat(x_WZ[msk], shape, axis=0)
                                    - nnearest_pt)

                nnearest_distances = np.sqrt(np.einsum('ij,ij->i',
                                                       nnearest_pt_diff,
                                                       nnearest_pt_diff))
                all_nearests = (np.vstack((nns.T, near_pts_Wz[msk])).T).flat

                assert (mapping[all_nearests] >= len(discret_val)).any() == False, \
                        ('Mapping, discretefunc and x point data are not '
                         + 'compatible with one another')

                all_vals = np.reshape(discret_val[mapping[all_nearests]],
                                      (nns.shape[0], shape + 1))
                all_distances = (np.vstack((nnearest_distances.reshape(nns.shape).T,
                                distances_WZ[msk]))).T


                inv_all_distances = (1/all_distances)**interpolate

                norm_inv_all_distances = (inv_all_distances
                                          / np.sum(inv_all_distances, axis=1)[:, None])

                vals[mask_not_near_zero[msk]] = np.einsum('ij, ij->i',
                                                     norm_inv_all_distances,
                                                     all_vals)
#        t3 = time.time()
#        print(t1 - t0)
#        print(t2 - t1)
#        print(t3 - t2)

        return vals

    elif len(x.shape) == 2 and not vector:
        # list of points
        # first non vectorized version
        discret_val = np.asarray(discret_val)

        if not interpolate:
            return discret_val[mapping[nearpts]]

        vals = np.zeros(len(x))
        for i, (subx, distance, nearpt) in enumerate(zip(x, distances, nearpts)):

            if abs(distance) < tolerance:
                vals[i] = discret_val[mapping[nearpt]]
            else:
                nnearest = firstneig[nearpt]
                nnearest_pt_diff = subx - mesh_points[
                        firstneig[nearpt]]
                nnearest_distances = np.sqrt(np.einsum('ij,ij->i',
                                                       nnearest_pt_diff,
                                                       nnearest_pt_diff))

                all_nearests = np.concatenate(([nearpt], nnearest))
                assert (mapping[all_nearests] >= len(discret_val)).any() == False, \
                        ('Mapping, discretefunc and x point data are not '
                         + 'compatible with one another')

                all_vals = discret_val[mapping[all_nearests]]
                all_distances = np.concatenate(([distance], nnearest_distances))
                inv_all_distances = (1 / all_distances)**interpolate
                vals[i] = ((inv_all_distances * all_vals).sum(axis=0) /
                    inv_all_distances.sum())

        return vals


def test_get_value_from_point():

    from numpy.testing import assert_allclose
    # function to test
    def func(x):
        x = np.asarray(x)
        if len(x.shape) < 2:
            x = x.reshape(1, len(x))
        return np.sin(x[:, 0] / 2) * np.cos(x[:, 1] / 3)

    # mesh
    xx = np.linspace(0, 15,50)
    meshsq = np.dstack([x.flatten() for x in np.meshgrid(xx, xx)])[0]
    vor = qhull.Voronoi(meshsq)
    vor_struct = discrete.Voronoi(vor=vor)
    tree = cKDTree(meshsq)
    sample_func = func(meshsq)

    # new_mesh
    new_xx = np.linspace(0, 15, 50)
    new_meshsq = np.dstack([x.flatten() for x in np.meshgrid(new_xx, new_xx)])[0]
    trueval = func(new_meshsq).reshape(len(new_xx), len(new_xx))

    mapping = np.arange(sample_func.shape[0])
    mapping_inv = np.arange(meshsq.shape[0])

    print('TESTING')

    t0 = time.time()
    seq_eval = np.array([[value_from_points([x, y], sample_func,
                                            mapping, mapping_inv,
                                            tree,
                                            interpolate=False, vector=True)
                          for x in new_xx] for y in new_xx])
    t1 = time.time()
    vec_eval = value_from_points(new_meshsq, sample_func,
                                 mapping, mapping_inv,
                                 tree,
                                 interpolate=False, vector=True)
    t2 = time.time()
    interp_eval = value_from_points(new_meshsq, sample_func,
                                    mapping, mapping_inv,
                                    tree,
                                    interpolate=True, vector=False)
    t3 = time.time()
    interp_vec_eval = value_from_points(new_meshsq, sample_func,
                                        mapping, mapping_inv,
                                        tree,
                                        interpolate=True, vector=True)
    t4 = time.time()
    interp_eval_sq = value_from_points(new_meshsq, sample_func,
                                       mapping, mapping_inv,
                                       tree,
                                       interpolate=2, vector=False)
    t5 = time.time()
    interp_vec_eval_sq = value_from_points(new_meshsq, sample_func,
                                           mapping, mapping_inv,
                                           tree,
                                           interpolate=2, vector=True)
    t6 = time.time()

    print('-'*50)
    print('-'*50)
    print('\n average error \n')
    print('sequencial', np.mean(abs(trueval - seq_eval)))

    print('interpolation', np.mean(abs(trueval
                                         - interp_vec_eval.reshape(
                                                 len(new_xx), len(new_xx)))))

    print('interpolation square', np.mean(abs(trueval
                                            - interp_vec_eval_sq.reshape(
                                                    len(new_xx), len(new_xx)))))
    print('-'*50)
    print('\n coherence tests \n')
    print('test not-interpolated -> sequential == vec',
          np.allclose(seq_eval, vec_eval.reshape(len(new_xx), len(new_xx))))
    print('test interpolated -> sequential == vec', np.allclose(interp_eval,
                                                                interp_vec_eval))
    print('test interpolated square -> sequential == vec', np.allclose(interp_eval_sq,
                                                         interp_vec_eval_sq))
    print('-'*50)
    print('\n efficiency test \n')
    print('Time spent during : \n')
    print('     Sequential calculation : \n')
    print('         without interpolation : {0:.5f}s '.format(t1-t0))
    print('         with interpolation : {0:.5f}s '.format(t3-t2))
    print('         with interpolation squared: {0:.5f}s \n'.format(t5-t4))

    print('     Vectorial calculation : \n')
    print('         without interpolation : {0:.5f}s'.format(t2-t1))
    print('         with interpolation : {0:.5f}s '.format(t4-t3))
    print('         with interpolation squared: {0:.5f}s '.format(t6-t5))
    print('-'*50)

    assert_allclose(seq_eval, vec_eval.reshape(len(new_xx), len(new_xx)),
                    err_msg=('Disgraeement between sequential calculation'
                             + ' without interpolation and vectorial '
                             + ' calculation WO interpolation' ))

    assert_allclose(interp_eval, interp_vec_eval,
                err_msg=('Disgraeement between sequential calculation'
                         + ' with interpolation and vectorial '
                         + ' calculation with interpolation' ))

    assert_allclose(interp_eval_sq, interp_vec_eval_sq,
            err_msg=('Disgraeement between sequential calculation'
                     + ' with interpolation squared and vectorial '
                     + ' calculation with interpolation squared' ))
