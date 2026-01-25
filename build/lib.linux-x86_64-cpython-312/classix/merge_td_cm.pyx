# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
import scipy.sparse as sparse

# Use cnp.intp_t for indexing to handle Windows (32-bit long) vs Linux (64-bit long)
# Use int32 for DSU parent array for memory efficiency

cdef int find_root(int* parent, int i) nogil:
    if parent[i] == i:
        return i
    parent[i] = find_root(parent, parent[i])
    return parent[i]

cdef void union_sets(int* parent, int i, int j) nogil:
    cdef int root_i = find_root(parent, i)
    cdef int root_j = find_root(parent, j)
    if root_i != root_j:
        # To match your Python "minlab" logic, we always point to the smaller index
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

cdef int c_searchsorted(double[:] arr, double target, int n) nogil:
    cdef int low = 0, high = n, mid
    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return low

def merge_tanimoto(
    double[:, :] spdata,              
    long[:] group_sizes,         
    double[:] sort_vals_sp,        
    cnp.intp_t[:] agg_labels_sp,       
    double radius,               
    double mergeScale,          
    int minPts,              
    int mergeTinyGroups,     
    verbose=False        
):
    cdef int n_groups = spdata.shape[0]
    cdef double threshold = mergeScale * radius
    
    # Fast CSR access
    spdatas = sparse.csr_matrix(spdata)
    cdef double[:] data = spdatas.data
    cdef int[:] indices = spdatas.indices
    cdef int[:] indptr = spdatas.indptr
    
    cdef signed char[:, :] Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    # parent array implements the 'label_sp' logic from your Python code via DSU
    cdef int[:] parent = np.arange(n_groups, dtype=np.int32)
    
    cdef int i, j, k, last_j
    cdef double dot, tan_dist, search_radius, norm_i, norm_j
    
    # PHASE 1: Build adjacency + DSU merging (Equivalent to label_sp[mask] = minlab)
    # -------------------------------------------------------------------------
    for i in range(n_groups):
        if (not mergeTinyGroups) and (group_sizes[i] < minPts):
            continue
            
        norm_i = sort_vals_sp[i]
        search_radius = norm_i / (1.0 - threshold)
        last_j = c_searchsorted(sort_vals_sp, search_radius, n_groups)
        
        with nogil:
            for j in range(i + 1, last_j):
                if (not mergeTinyGroups) and (group_sizes[j] < minPts):
                    continue
                
                # Manual Dot Product (equivalent to spsubmatxvec)
                dot = 0.0
                for k in range(indptr[j], indptr[j+1]):
                    dot += spdata[i, indices[k]] * data[k]
                
                norm_j = sort_vals_sp[j]
                # Tanimoto Dist: 1 - (ips / (norm_i + norm_j - ips))
                tan_dist = 1.0 - (dot / (norm_i + norm_j - dot))
                
                if tan_dist <= threshold:
                    Adj[i, j] = 1
                    Adj[j, i] = 1
                    union_sets(&parent[0], i, j)

    # Convert DSU to flat labels for Phase 2
    cdef int[:] label_sp = np.zeros(n_groups, dtype=np.int32)
    for i in range(n_groups):
        label_sp[i] = find_root(&parent[0], i)

    # PHASE 2: minPts redistribution
    # -------------------------------------------------------------------------
    unique_labels, inverse_indices = np.unique(np.asarray(label_sp), return_inverse=True)
    cdef int[:] current_labels = inverse_indices.astype(np.int32)
    cdef int n_clusters = len(unique_labels)
    
    cdef long[:] cluster_sizes = np.zeros(n_clusters, dtype=np.int64)
    for i in range(n_groups):
        cluster_sizes[current_labels[i]] += group_sizes[i]

    # Find labels of clusters that are too small
    # We use a copy of labels to ensure redistribution doesn't cascade mid-loop
    cdef int[:] label_sp_copy = np.copy(current_labels)
    cdef double min_dist, d_val, lower_bound
    cdef int best_gid, target_cluster
    
    for i in range(n_groups):
        if cluster_sizes[label_sp_copy[i]] < minPts:
            min_dist = 2.0 
            best_gid = -1
            norm_i = sort_vals_sp[i]
            
            with nogil:
                for j in range(n_groups):
                    # In your Python: if cs[target_cluster] >= minPts
                    # We check the cluster size of group j
                    target_cluster = label_sp_copy[j]
                    if cluster_sizes[target_cluster] < minPts:
                        continue
                    
                    norm_j = sort_vals_sp[j]
                    
                    # Tanimoto Pruning: Optimization that doesn't change the result
                    if norm_i < norm_j:
                        lower_bound = 1.0 - (norm_i / norm_j)
                    else:
                        lower_bound = 1.0 - (norm_j / norm_i)
                    
                    if lower_bound > min_dist:
                        continue
                        
                    dot = 0.0
                    for k in range(indptr[j], indptr[j+1]):
                        dot += spdata[i, indices[k]] * data[k]
                    
                    d_val = 1.0 - (dot / (norm_i + norm_j - dot))
                    
                    # Stable tie-breaking: if distances are equal, keep the first one found
                    # (Matches np.argsort with kind='stable' behavior)
                    if d_val < min_dist:
                        min_dist = d_val
                        best_gid = j
            
            if best_gid != -1:
                current_labels[i] = label_sp_copy[best_gid]
                Adj[i, best_gid] = 2
                Adj[best_gid, i] = 2

    # Final Renumbering to 0...K-1
    final_unique, final_labels = np.unique(np.asarray(current_labels), return_inverse=True)
        
    return {
        'group_cluster_labels': final_labels.astype(np.int32),
        'Adj': np.asarray(Adj),
        'final_cluster_sizes': np.bincount(final_labels).astype(np.int64)
    }