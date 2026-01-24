import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from spmv import spsubmatxvec  # Sparse submatrix-vector multiplication (C extension)

def merge_tanimoto(
    spdata,              # Group centers in sorted space (dense numpy array, shape: n_groups x dim)
    group_sizes,         # Size of each aggregation group (numpy array, length: n_groups)
    sort_vals_sp,        # sort_vals for group centers (numpy array, length: n_groups)
    agg_labels_sp,       # Initial group labels (typically np.arange(n_groups))
    radius,              # Aggregation radius
    mergeScale,          # Merging scale factor
    minPts,              # Minimum points for a valid cluster
    mergeTinyGroups,     # Whether to ignore tiny groups during initial merging
    verbose=False        # Whether to show progress bars
):
    """
    Tanimoto distance-based group merging with edge-wise merging and minPts redistribution.
    Fully matches the behavior of the original CLASSIX_T implementation.
    """
    n_groups = len(spdata)
    
    # Convert group centers to CSR sparse format for fast inner products
    spdatas = sparse.csr_matrix(spdata)
    spdatas_data = spdatas.data
    spdatas_indices = spdatas.indices
    spdatas_indptr = spdatas.indptr
    
    # Adjacency matrix (for explainability / path finding)
    Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    
    # Current cluster labels for groups (initially independent)
    label_sp = agg_labels_sp.copy()
    
    # Phase 1: Build adjacency + aggressive edge-wise merging
    pbar = tqdm(range(n_groups), desc="Building adjacency", disable=not verbose)
    for i in pbar:
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue
        
        xi = spdata[i]
        # Tanimoto inequality-derived search bound
        search_radius = sort_vals_sp[i] / (1 - mergeScale * radius)
        last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')
        
        if last_j > i:
            # Pre-allocate output array for inner products
            n_rows = last_j - i
            ips = np.zeros(n_rows, dtype=np.float64)
            
            # Sparse inner product: rows [i:last_j) dot xi
            spsubmatxvec(
                spdatas_data.astype(np.float64),
                spdatas_indptr,
                spdatas_indices,
                i,
                last_j,
                xi.astype(np.float64),
                ips
            )
            
            # Tanimoto similarity and distance
            tanimoto_sim = ips / (sort_vals_sp[i] + sort_vals_sp[i:last_j] - ips)
            tanimoto_dist = 1 - tanimoto_sim
            
            # Connected groups
            inds_rel = np.where(tanimoto_dist <= mergeScale * radius)[0]
            inds = i + inds_rel
            
            if not mergeTinyGroups:
                valid = group_sizes[inds] >= minPts
                inds = inds[valid]
            
            # Update adjacency (symmetric)
            Adj[i, inds] = 1
            Adj[inds, i] = 1
            
            # Critical: merge labels immediately upon discovering connections
            connected_labels = np.unique(label_sp[inds])
            if len(connected_labels) > 1:
                minlab = np.min(connected_labels)
                for lbl in connected_labels:
                    label_sp[label_sp == lbl] = minlab
    
    # Phase 2: minPts-based redistribution of tiny clusters
    ul = np.unique(label_sp)
    cs = np.zeros(len(ul), dtype=int)
    group_sizes_arr = np.array(group_sizes)
    
    # Renumber clusters contiguously and compute sizes
    new_label_map = {old: new for new, old in enumerate(ul)}
    for old, new in new_label_map.items():
        mask = (label_sp == old)
        cs[new] = np.sum(group_sizes_arr[mask])
        label_sp[mask] = new
    
    small_clusters = np.where(cs < minPts)[0]
    
    label_sp_copy = label_sp.copy()
    
    pbar_small = tqdm(small_clusters, desc="minPts merging", disable=not verbose or len(small_clusters) == 0)
    for cluster_id in pbar_small:
        group_ids = np.where(label_sp_copy == cluster_id)[0]
        for gid in group_ids:
            xi = spdata[gid]
            
            # Full inner products with all group centers
            ips = np.zeros(n_groups, dtype=np.float64)
            spsubmatxvec(
                spdatas_data.astype(np.float64),
                spdatas_indptr,
                spdatas_indices,
                0,
                n_groups,
                xi.astype(np.float64),
                ips
            )
            
            # Tanimoto distance to all groups
            d = 1 - ips / (sort_vals_sp[gid] + sort_vals_sp - ips)
            order = np.argsort(d, kind='stable')
            
            # Reassign to nearest large cluster
            for nearest_gid in order:
                target_cluster = label_sp_copy[nearest_gid]
                if cs[target_cluster] >= minPts:
                    label_sp[gid] = target_cluster
                    Adj[gid, nearest_gid] = 2
                    Adj[nearest_gid, gid] = 2
                    break
    
    # Final contiguous renumbering
    ul_final = np.unique(label_sp)
    final_map = {old: new for new, old in enumerate(ul_final)}
    label_sp = np.array([final_map[l] for l in label_sp])
    
    return {
        'group_cluster_labels': label_sp,           # Final cluster ID for each group
        'Adj': Adj,                                 # Adjacency matrix (for explainability)
        'final_cluster_sizes': np.bincount(label_sp)  # Sizes of final clusters
    }