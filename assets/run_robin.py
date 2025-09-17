## Run ROBIN for outlier pruning compatibility checks

import json
import numpy as np
import cvxpy as cp
from scipy.io import loadmat
import spark_robin as robin_py
import copy
import time
import glob
import h5py


def robin_prune_outliers(tgt, cad_dist_min, cad_dist_max, noise_bound, method='maxclique'):
    '''
    First form a compatibility graph and then 
    Use robin to select inliers
    '''
    N = tgt.shape[1]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    # distances || tgt_j - tgt_i ||
    tgt_dist_ij = np.linalg.norm(
        tgt[:, sj] - tgt[:, si], axis=0)  # shape (n-1)_tri

    allEdges = np.arange(si.shape[0])
    check1 = tgt_dist_ij >= (cad_dist_min - 2 * noise_bound)
    check2 = tgt_dist_ij <= (cad_dist_max + 2 * noise_bound)
    mask_compatible = check1 & check2
    validEdges = allEdges[mask_compatible]
    sdata = np.zeros_like(si)
    sdata[mask_compatible] = 1

    comp_mat = np.zeros((N, N))
    comp_mat[si, sj] = sdata

    # creating a Graph in robin
    g = robin_py.AdjListGraph()
    for i in range(N):
        g.AddVertex(i)

    for edge_idx in validEdges:
        # print(f'Add edge between {si[edge_idx]} and {sj[edge_idx]}.')
        g.AddEdge(si[edge_idx], sj[edge_idx])

    if method == "maxclique":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CLIQUE)
    elif method == "maxcore":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CORE)
    else:
        raise RuntimeError('Prune outliers only support maxclique and maxcore')

    # adj_mat = g.GetAdjMat()

    return inlier_indices, comp_mat

def minimum_distance_to_convex_hull(A):
    '''
    A is shape 3 by K, compute the minimum distance from the origin to the convex hull of A
    '''
    K = A.shape[1]
    P = A.T @ A
    # eigenvalues, eigenvectors = np.linalg.eigh(P)
    # eigenvalues[eigenvalues < 0] = 0
    # P = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    one = np.ones((K, 1))
    # Use CVXPY to solve
    x = cp.Variable(K)
    # prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)),
    #                   [x >= 0,
    #                    one.T @ x == 1])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x)),
                      [x >= 0,
                       one.T @ x == 1])
    # try:
    prob.solve(solver='ECOS', verbose=False)
    x_val = x.value
    min_distance = np.linalg.norm(A @ x_val)
    # except:
    #     print("Failed")
    #     min_distance = 0
    return min_distance

def compute_min_max_distances(cad_kpts):
    print('Computing upper and lower bounds in cad pairwise distances...')

    K = cad_kpts.shape[0]
    N = cad_kpts.shape[2]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    cad_TIMs_ij = cad_kpts[:, :, sj] - cad_kpts[:, :, si]  # shape K by 3 by (n-1)_tri

    # compute max distances
    cad_dist_k_ij = np.linalg.norm(cad_TIMs_ij, axis=1)  # shape K by (n-1)_tri
    cad_dist_max_ij = np.max(cad_dist_k_ij, axis=0)

    # compute min distances
    cad_dist_min_ij = []
    num_pairs = cad_TIMs_ij.shape[2]
    one_tenth = num_pairs / 10
    for i in range(num_pairs):
        tmp = cad_TIMs_ij[:, :, i].T
        min_dist = minimum_distance_to_convex_hull(tmp)
        cad_dist_min_ij.append(min_dist)
        if i % one_tenth == 1:
            print(f'{i}/{num_pairs}.')

    cad_dist_min_ij = np.array(cad_dist_min_ij)

    print('Done')

    return cad_dist_min_ij, cad_dist_max_ij

def compute_min_max_distances_with_idx_maps(cad_kpts):
    print('Computing upper and lower bounds in cad pairwise distances...')

    K = cad_kpts.shape[0]
    N = cad_kpts.shape[2]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    cad_TIMs_ij = cad_kpts[:, :, sj] - cad_kpts[:, :, si]  # shape K by 3 by (n-1)_tri

    # compute max distances
    cad_dist_k_ij = np.linalg.norm(cad_TIMs_ij, axis=1)  # shape K by (n-1)_tri
    cad_dist_max_ij = np.max(cad_dist_k_ij, axis=0)

    # compute min distances
    cad_dist_min_ij = []
    num_pairs = cad_TIMs_ij.shape[2]
    one_tenth = num_pairs / 10
    for i in range(num_pairs):
        tmp = cad_TIMs_ij[:, :, i].T
        min_dist = minimum_distance_to_convex_hull(tmp)
        cad_dist_min_ij.append(min_dist)
        if i % one_tenth == 1:
            print(f'{i}/{num_pairs}.')

    cad_dist_min_ij = np.array(cad_dist_min_ij)

    print('Done')

    return cad_dist_min_ij, cad_dist_max_ij, si, sj


def select_cad_dist_bounds(ids_to_select, original_cad_dist_min, original_cad_dist_max, cad_dist_i_map, cad_dist_j_map):
    """Select CAD db distance bounds by the indices of semantic keypoints provided.

    :param ids_to_select:
    :param original_cad_dist_min:
    :param original_cad_dist_max:
    :param cad_dist_i_map:
    :param cad_dist_j_map:
    :return:
    """
    # obtain the indices of distance bounds array that we want to keep
    counter = 0
    bound_idx = []
    for i, j in zip(list(cad_dist_i_map), list(cad_dist_j_map)):
        # if both (i,j) are in the semantic keypoint list, select
        if i in ids_to_select and j in ids_to_select:
            bound_idx.append(counter)
        counter += 1

    # select
    dist_min = original_cad_dist_min[np.array(bound_idx).astype("int")]
    dist_max = original_cad_dist_max[np.array(bound_idx).astype("int")]

    return dist_min, dist_max

def nocs():
    keypoint_files = glob.glob(f"/home/lorenzo/research/playground/nepv/FastPACE/data/{dataset}/scene_*.json")
    NOISE_BOUND = 0.005

    # load shapes
    shape_file = f"/home/lorenzo/research/playground/nepv/FastPACE/data/{dataset}/shapes.mat"
    try:
        shapes = loadmat(shape_file)
    except:
        shapes = h5py.File(shape_file, 'r')
    shapes = np.array(shapes['shapes']) # 3 x N x K
    shapes = np.transpose(shapes,[2,0,1]) # K x 3 x N

    # precompute cad min/max distances
    cad_dist_min, cad_dist_max = compute_min_max_distances(shapes)

    for keypoint_file in keypoint_files:
        with open(keypoint_file, 'r') as f:
            data = json.load(f)


        new_data = []
        for entry in data:
            d = copy.deepcopy(entry)

            tgt_points = np.array(d['est_world_keypoints']).T # 3 x N

            t_start = time.time()
            clique_inliers, _ = robin_prune_outliers(tgt_points, cad_dist_min, cad_dist_max, NOISE_BOUND,
                                                        method='maxclique')
            t_clique_end = time.time() - t_start

            d['robin_inliers'] = clique_inliers
            d['robin_time'] = t_clique_end
            # use GNC if this is empty, otherwise just use these to start

            new_data.append(d)
        
        parts = keypoint_file.split("/")
        save_file = "/".join(parts[:-1]) + "/robin_" + parts[-1]
        with open(save_file, 'w') as f:
            json.dump(new_data, f)


def apollo():
    keypoint_file = f"/home/lorenzo/research/playground/nepv/FastPACE/data/{dataset}/detections.json"
    NOISE_BOUND = 0.05

    # load shapes
    shapes = np.load("cad_kpts.npy")

    # precompute cad min/max distances
    cad_dist_min, cad_dist_max = compute_min_max_distances(shapes)
    # cad_dist_min, cad_dist_max, i_map, j_map = compute_min_max_distances_with_idx_maps(shapes)

    with open(keypoint_file, 'r') as f:
        data = json.load(f)

    new_data = {}
    for key in data:
        new_data[key] = []
        for entry in data[key]:
            d = copy.deepcopy(entry)

            tgt_points = np.array(d['est_world_keypoints']) # 3 x N
            tgt_points_ids = d["tgt_points_ids"]

            # updated_cad_dist_min, updated_cad_dist_max = select_cad_dist_bounds(tgt_points_ids, cad_dist_min,
            #                                                             cad_dist_max, i_map,
            #                                                             j_map)
            # test = tgt_points[:,tgt_points_ids]

            t_start = time.time()
            clique_inliers, _ = robin_prune_outliers(tgt_points, cad_dist_min, cad_dist_max, NOISE_BOUND,
                                                        method='maxclique')
            # clique_inliers, _ = robin_prune_outliers(test, updated_cad_dist_min, updated_cad_dist_max, NOISE_BOUND,
                                                        # method='maxclique')
            t_clique_end = time.time() - t_start

            d['robin_inliers'] = clique_inliers
            d['robin_time'] = t_clique_end
            # use GNC if this is empty, otherwise just use these to start

            new_data[key].append(d)
    
    parts = keypoint_file.split("/")
    save_file = "/".join(parts[:-1]) + "/robin_" + parts[-1]
    with open(save_file, 'w') as f:
        json.dump(new_data, f)


if __name__ == '__main__':
    dataset = "nocs/mug"
    # dataset = "apolloscape2"
    # dataset = "cast"

    if "nocs" in dataset:
        nocs()
    elif "apollo" in dataset:
        apollo()
    else:
        # cast
        keypoint_files = glob.glob(f"/home/lorenzo/research/playground/nepv/FastPACE/data/{dataset}/cast.json")
        NOISE_BOUND = 0.005

        # load shapes
        shape_file = f"/home/lorenzo/research/playground/nepv/FastPACE/data/{dataset}/racecar_lib.mat"
        try:
            shapes = loadmat(shape_file)
        except:
            shapes = h5py.File(shape_file, 'r')
        shapes = np.array(shapes['shapes']) # 3 x N x K
        shapes = np.transpose(shapes,[2,0,1]) # K x 3 x N

        # precompute cad min/max distances
        cad_dist_min, cad_dist_max = compute_min_max_distances(shapes)

        for keypoint_file in keypoint_files:
            with open(keypoint_file, 'r') as f:
                data = json.load(f)


            new_data = []
            for entry in data:
                d = copy.deepcopy(entry)

                tgt_points = np.array(d['est_world_keypoints']).T # 3 x N

                t_start = time.time()
                clique_inliers, _ = robin_prune_outliers(tgt_points, cad_dist_min, cad_dist_max, NOISE_BOUND,
                                                            method='maxclique')
                t_clique_end = time.time() - t_start

                d['robin_inliers'] = clique_inliers
                d['robin_time'] = t_clique_end
                # use GNC if this is empty, otherwise just use these to start

                new_data.append(d)
            
            parts = keypoint_file.split("/")
            save_file = "/".join(parts[:-1]) + "/robin_" + parts[-1]
            with open(save_file, 'w') as f:
                json.dump(new_data, f)
        pass
