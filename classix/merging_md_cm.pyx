# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport fabs

# 定义一个简单的并查集结构用于高效合并 label
cdef int find_root(int* parent, int i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # 路径压缩
        i = parent[i]
    return i

cdef void union_roots(int* parent, int i, int j):
    cdef int root_i = find_root(parent, i)
    cdef int root_j = find_root(parent, j)
    if root_i != root_j:
        if root_i < root_j:
            parent[root_j] = root_i
        else:
            parent[root_i] = root_j

# 用于处理 minPts 重新分配时的结构
cdef struct DistIdx:
    double dist
    int index

cdef int compare_dist_idx(const void* a, const void* b) noexcept:
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
    cdef int n_groups = spdata.shape[0]
    cdef int dim = spdata.shape[1]
    cdef double thresh = mergeScale * radius
    
    # 初始化输出
    cdef signed char[:, :] Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    cdef int* parent = <int*> malloc(n_groups * sizeof(int))
    for i in range(n_groups):
        parent[i] = i

    cdef int i, j, k, last_j
    cdef double d, diff
    cdef int root_i, root_j

    # 第一步：构建 Adjacency 并通过并查集在线合并
    for i in range(n_groups):
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue
        
        # 寻找范围内候选者 (利用 sort_vals_sp 的有序性)
        # search_radius = thresh + sort_vals_sp[i]
        # 使用线性扫描代替 np.searchsorted，在 Cython 中处理局部切片更快
        for j in range(i, n_groups):
            if sort_vals_sp[j] > thresh + sort_vals_sp[i]:
                last_j = j
                break
        else:
            last_j = n_groups

        for j in range(i, last_j):
            if not mergeTinyGroups and group_sizes[j] < minPts:
                continue
            
            # 计算 Manhattan 距离 (L1)
            d = 0.0
            for k in range(dim):
                diff = spdata[i, k] - spdata[j, k]
                d += fabs(diff)
                if d > thresh: break
            
            if d <= thresh:
                Adj[i, j] = 1
                Adj[j, i] = 1
                union_roots(parent, i, j)

    # 导出初步 label
    cdef long[:] label_sp = np.empty(n_groups, dtype=np.int)
    for i in range(n_groups):
        label_sp[i] = find_root(parent, i)

    # 第二步：计算 Cluster 大小并重新映射
    # 使用 Python dict 处理映射以保持逻辑一致，但计算在 C 中完成
    unique_labels = np.unique(label_sp)
    cdef int n_clusters = len(unique_labels)
    cdef long[:] cluster_sizes = np.zeros(n_clusters, dtype=np.int)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    
    for i in range(n_groups):
        new_lab = label_map[label_sp[i]]
        label_sp[i] = new_lab
        cluster_sizes[new_lab] += group_sizes[i]

    # 第三步：处理小簇重新分配 (minPts 逻辑)
    cdef DistIdx* dists_arr = <DistIdx*> malloc(n_groups * sizeof(DistIdx))
    cdef int target_cluster
    
    for i in range(n_groups):
        if cluster_sizes[label_sp[i]] < minPts:
            # 找到最近的且满足 size >= minPts 的 cluster
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

    # 最后映射 label 到连续空间 0~k-1
    final_unique = np.unique(label_sp)
    final_map = {old: new for new, old in enumerate(final_unique)}
    cdef long[:] final_labels = np.zeros(n_groups, dtype=np.int)
    for i in range(n_groups):
        final_labels[i] = final_map[label_sp[i]]

    free(parent)
    free(dists_arr)

    return {
        'group_cluster_labels': np.asarray(final_labels),
        'Adj': np.asarray(Adj),
        'final_cluster_sizes': np.bincount(np.asarray(final_labels))
    }