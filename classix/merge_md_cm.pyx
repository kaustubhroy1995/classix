# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport fabs

# 1. Helper functions with noexcept to avoid GIL overhead
cdef int find_root(int* parent, int i) noexcept nogil:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

cdef void union_roots(int* parent, int i, int j) noexcept nogil:
    cdef int root_i = find_root(parent, i)
    cdef int root_j = find_root(parent, j)
    if root_i != root_j:
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

cdef struct DistIdx:
    double dist
    int index

cdef int compare_dist_idx(const void* a, const void* b) noexcept nogil:
    cdef double diff = (<DistIdx*>a).dist - (<DistIdx*>b).dist
    return 1 if diff > 0 else (-1 if diff < 0 else 0)

def merge_manhattan(
    double[:, :] spdata,
    long[:] group_sizes,
    double[:] sort_vals_sp,
    long[:] agg_labels_sp,
    double radius,
    double mergeScale,
    int minPts,
    int mergeTinyGroups,
    int verbose=0
):
    # Variables declared at top to avoid 'redeclared' errors
    cdef int n_groups = spdata.shape[0]
    cdef int dim = spdata.shape[1]
    cdef double thresh = mergeScale * radius
    cdef int i, j, k, last_j, new_lab, target_cluster
    cdef double d, diff
    
    cdef signed char[:, :] Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    cdef int* parent = <int*> malloc(n_groups * sizeof(int))
    cdef DistIdx* dists_arr = <DistIdx*> malloc(n_groups * sizeof(DistIdx))
    
    if not parent or not dists_arr:
        if parent: free(parent)
        if dists_arr: free(dists_arr)
        raise MemoryError()

    for i in range(n_groups):
        parent[i] = i

    # Build Adjacency
    for i in range(n_groups):
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue
        
        last_j = n_groups
        for j in range(i + 1, n_groups):
            if sort_vals_sp[j] > thresh + sort_vals_sp[i]:
                last_j = j
                break

        for j in range(i + 1, last_j):
            if not mergeTinyGroups and group_sizes[j] < minPts:
                continue
            
            d = 0.0
            for k in range(dim):
                diff = spdata[i, k] - spdata[j, k]
                d += fabs(diff)
                if d > thresh: break
            
            if d <= thresh:
                Adj[i, j] = 1
                Adj[j, i] = 1
                union_roots(parent, i, j)

    # Label extraction
    cdef long[:] label_sp = np.empty(n_groups, dtype=np.intp)
    for i in range(n_groups):
        label_sp[i] = find_root(parent, i)

    # Size calculation
    unique_labels = np.unique(label_sp)
    cdef int n_clusters = len(unique_labels)
    cdef long[:] cluster_sizes = np.zeros(n_clusters, dtype=np.intp)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    
    for i in range(n_groups):
        new_lab = label_map[label_sp[i]]
        label_sp[i] = new_lab
        cluster_sizes[new_lab] += group_sizes[i]

    # Reassignment logic
    for i in range(n_groups):
        if cluster_sizes[label_sp[i]] < minPts:
            for j in range(n_groups):
                dists_arr[j].index = j
                d = 0.0
                for k in range(dim):
                    d += fabs(spdata[i, k] - spdata[j, k])
                dists_arr[j].dist = d
            
            qsort(dists_arr, n_groups, sizeof(DistIdx), compare_dist_idx)
            
            for j in range(n_groups):
                target_cluster = label_sp[dists_arr[j].index]
                if cluster_sizes[target_cluster] >= minPts:
                    label_sp[i] = target_cluster
                    Adj[i, dists_arr[j].index] = 2
                    Adj[dists_arr[j].index, i] = 2
                    break

    # Mapping to final output
    final_unique = np.unique(label_sp)
    final_map = {old: new for new, old in enumerate(final_unique)}
    cdef long[:] final_labels = np.zeros(n_groups, dtype=np.intp)
    for i in range(n_groups):
        final_labels[i] = final_map[label_sp[i]]

    free(parent)
    free(dists_arr)

    return {
        'group_cluster_labels': np.asarray(final_labels),
        'Adj': np.asarray(Adj),
        'final_cluster_sizes': np.bincount(np.asarray(final_labels)).astype(np.int64)
    }