# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
import scipy.sparse as sparse
from tqdm import tqdm
from libcpp.vector cimport vector

# Helper for binary search (equivalent to np.searchsorted(side='right'))
cdef Py_ssize_t bisect_right(double[:] a, double x, Py_ssize_t lo, Py_ssize_t hi) nogil:
    cdef Py_ssize_t mid
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

def aggregate_tanimoto(
    double[:, :] data, 
    double radius, 
    bint verbose=False
):
    cdef Py_ssize_t n = data.shape[0]
    
    # --- Pre-processing (Python/Numpy is efficient here) ---
    # 1. Calculate sort_vals (L1 norm)
    cdef double[:] sort_vals = np.sum(data, axis=1)
    
    # 2. Sort indices
    cdef long[:] ind = np.argsort(sort_vals, kind='stable')
    
    # 3. Apply sorting to data
    # Create a dense sorted copy. This is necessary for fast random access in the inner loop.
    cdef double[:, :] data_sorted = np.asarray(data)[ind]
    cdef double[:] sort_vals_sorted = np.asarray(sort_vals)[ind]
    
    # 4. Convert to CSR for fast sparse iteration over candidate neighbors
    # Using scipy to generate the structure, then accessing raw C pointers
    datas = sparse.csr_matrix(np.asarray(data_sorted))
    cdef double[:] csr_data = datas.data
    cdef int[:] csr_indices = datas.indices
    cdef int[:] csr_indptr = datas.indptr

    # --- Initialization ---
    cdef long[:] labels = np.full(n, -1, dtype=np.int64)
    cdef vector[long] splist
    cdef vector[long] group_sizes
    
    # Reserve memory to avoid frequent reallocations (heuristic)
    splist.reserve(n // 10) 
    group_sizes.reserve(n // 10)

    cdef Py_ssize_t i, j, k, last_j
    cdef long lab = 0
    cdef long nr_dist = 0
    cdef double sv_i, search_radius, dot, threshold_val
    
    # Precompute threshold constants
    # Logic: vec >= rhsi  <=>  dot / (val_i + val_j) >= rhsi
    cdef double rhs = 1.0 / (1.0 - radius) + 1.0
    cdef double rhsi = 1.0 / rhs
    
    # Progress bar handling
    cdef object pbar = None
    if verbose:
        pbar = tqdm(total=n, desc="Aggregation")
    cdef int update_iters = 0
    cdef int update_threshold = 1000  # Update python pbar every 1000 iters
    
    # --- Main Loop (Fully Nogil capable logic, except for pbar update) ---
    for i in range(n):
        # Update progress bar occasionally
        if verbose:
            update_iters += 1
            if update_iters >= update_threshold:
                pbar.update(update_iters)
                update_iters = 0

        # Skip if already assigned
        if labels[i] >= 0:
            continue
        
        # Start a new cluster
        labels[i] = lab
        splist.push_back(i)
        group_sizes.push_back(1)
        
        sv_i = sort_vals_sorted[i]
        search_radius = sv_i / (1.0 - radius)
        
        # Find search bound
        last_j = bisect_right(sort_vals_sorted, search_radius, i + 1, n)
        
        if last_j > i + 1:
            nr_dist += (last_j - (i + 1)) # Record search space size (matching original metric)
            
            # Iterate candidates
            for j in range(i + 1, last_j):
                # Critical Optimization: Skip if candidate is already assigned.
                # The original vectorized code computed dot products for ALL j in range.
                # This check saves massive amounts of computation.
                if labels[j] != -1:
                    continue
                
                # Compute Dot Product: Dense (i) dot Sparse (j)
                # data_sorted[i] is the center, datas[j] is the candidate
                dot = 0.0
                for k in range(csr_indptr[j], csr_indptr[j+1]):
                    # data_sorted[i, col] * val
                    dot += data_sorted[i, csr_indices[k]] * csr_data[k]
                
                # Check Tanimoto condition
                # Condition: ips / (sort_vals[i] + sort_vals[j]) >= rhsi
                if (dot / (sv_i + sort_vals_sorted[j])) >= rhsi:
                    labels[j] = lab
                    group_sizes[lab] += 1
        
        lab += 1

    if verbose:
        pbar.update(update_iters)
        pbar.close()

    # --- Convert C++ vectors to Numpy arrays for output ---
    cdef long[:] splist_np = np.zeros(splist.size(), dtype=int)
    cdef long[:] group_sizes_np = np.zeros(group_sizes.size(), dtype=int)
    
    for i in range(splist.size()):
        splist_np[i] = splist[i]
        group_sizes_np[i] = group_sizes[i]

    return {
        'labels': np.asarray(labels),
        'splist': np.asarray(splist_np),
        'group_sizes': np.asarray(group_sizes_np),
        'ind': np.asarray(ind),
        'sort_vals': np.asarray(sort_vals_sorted),
        'data_sorted': np.asarray(data_sorted),
        'nr_dist': nr_dist
    }