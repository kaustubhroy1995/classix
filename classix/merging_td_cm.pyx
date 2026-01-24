# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from tqdm import tqdm
import scipy.sparse as sparse

# DSU find function with path compression
cdef int find_root(int* parent, int i) nogil:
    if parent[i] == i:
        return i
    parent[i] = find_root(parent, parent[i])
    return parent[i]

# DSU union function
cdef void union_sets(int* parent, int i, int j) nogil:
    cdef int root_i = find_root(parent, i)
    cdef int root_j = find_root(parent, j)
    if root_i != root_j:
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

def merge_tanimoto(
    double[:, :] spdata,              
    long[:] group_sizes,         
    double[:] sort_vals_sp,        
    cnp.int32_t[:] agg_labels_sp,       
    double radius,              
    double mergeScale,          
    int minPts,              
    int mergeTinyGroups,     
    verbose=False        
):
    cdef int n_groups = spdata.shape[0]
    cdef int dim = spdata.shape[1]
    cdef double threshold = mergeScale * radius
    
    # Prepare CSR data
    spdatas = sparse.csr_matrix(spdata)
    cdef double[:] data = spdatas.data
    cdef int[:] indices = spdatas.indices
    cdef int[:] indptr = spdatas.indptr
    
    # Outputs
    cdef signed char[:, :] Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    cdef int[:] parent = np.arange(n_groups, dtype=np.int32)
    
    cdef int i, j, k, col, row_start, row_end, last_j
    cdef double xi_val, dot, tan_dist, search_radius
    
    # Phase 1: Adjacency + DSU Merging
    range_iter = tqdm(range(n_groups), desc="Building adjacency", disable=not verbose)
    for i in range_iter:
        if (not mergeTinyGroups) and (group_sizes[i] < minPts):
            continue
            
        search_radius = sort_vals_sp[i] / (1.0 - threshold)
        # Use binary search for last_j
        last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')
        
        for j in range(i + 1, last_j):
            if (not mergeTinyGroups) and (group_sizes[j] < minPts):
                continue
            
            # Sparse dot product: row i and row j
            dot = 0.0
            row_start = indptr[i]
            row_end = indptr[i+1]
            
            # Efficient sparse-sparse or sparse-dense dot can be here
            # Since spdata is dense in input, we use dense-sparse for simplicity & speed
            for k in range(indptr[j], indptr[j+1]):
                col = indices[k]
                dot += spdata[i, col] * data[k]
                
            tan_dist = 1.0 - (dot / (sort_vals_sp[i] + sort_vals_sp[j] - dot))
            
            if tan_dist <= threshold:
                Adj[i, j] = 1
                Adj[j, i] = 1
                union_sets(&parent[0], i, j)

    # Resolve labels from DSU
    cdef int[:] label_sp = np.zeros(n_groups, dtype=np.int32)
    for i in range(n_groups):
        label_sp[i] = find_root(&parent[0], i)

    # Phase 2: minPts redistribution
    # Calculate current cluster sizes
    cdef long[:] cluster_sizes = np.zeros(n_groups, dtype=np.int)
    for i in range(n_groups):
        cluster_sizes[label_sp[i]] += group_sizes[i]

    cdef double[:] ips = np.zeros(n_groups, dtype=np.float64)
    cdef double min_dist, d_val
    cdef int best_gid, target_cluster
    
    # Find small clusters
    for i in range(n_groups):
        if cluster_sizes[label_sp[i]] < minPts:
            # Redistribution logic
            min_dist = 2.0 
            best_gid = -1
            
            # Calculate Tanimoto to all others
            for j in range(n_groups):
                if i == j: continue
                
                dot = 0.0
                for k in range(indptr[j], indptr[j+1]):
                    dot += spdata[i, indices[k]] * data[k]
                
                d_val = 1.0 - (dot / (sort_vals_sp[i] + sort_vals_sp[j] - dot))
                
                # Check if j belongs to a large cluster
                if cluster_sizes[label_sp[j]] >= minPts:
                    if d_val < min_dist:
                        min_dist = d_val
                        best_gid = j
            
            if best_gid != -1:
                label_sp[i] = label_sp[best_gid]
                Adj[i, best_gid] = 2
                Adj[best_gid, i] = 2

    # Final contiguous renumbering
    unique_labels = np.unique(np.asarray(label_sp))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cdef int[:] final_labels = np.zeros(n_groups, dtype=np.int32)
    for i in range(n_groups):
        final_labels[i] = label_map[label_sp[i]]
        
    return {
        'group_cluster_labels': np.asarray(final_labels),
        'Adj': np.asarray(Adj),
        'final_cluster_sizes': np.bincount(np.asarray(final_labels))
    }