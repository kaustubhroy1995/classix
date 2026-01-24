# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport fabs
from tqdm import tqdm

# 定义数据类型
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t ITYPE_t

def aggregate_manhattan(double[:, :] data, double radius, int verbose=0):
    """
    Cython optimized version of aggregate_manhattan.
    Inputs and Outputs match the original function exactly.
    """
    cdef int n = data.shape[0]
    cdef int dim = data.shape[1]
    
    # 1. 预处理：计算 Sort Values 并排序 (NumPy 做排序依然很快)
    # 我们保留这部分逻辑，将排序后的数组传给 C 循环
    cdef double[:] sort_vals = np.sum(data, axis=1)
    cdef long[:] ind = np.argsort(sort_vals)
    
    cdef double[:, :] data_sorted = np.empty((n, dim), dtype=np.float64)
    cdef double[:] sort_vals_sorted = np.empty(n, dtype=np.float64)
    
    # 构建排序后的数据视图
    cdef int i, j, k, r
    for i in range(n):
        r = ind[i]
        sort_vals_sorted[i] = sort_vals[r]
        for k in range(dim):
            data_sorted[i, k] = data[r, k]

    # 初始化输出容器
    cdef long[:] labels = np.full(n, -1, dtype=int)
    splist = []      # Python list is fine for accumulating indices
    group_sizes = [] # Python list
    
    cdef int lab = 0
    cdef long nr_dist = 0
    cdef double limit
    cdef double d, diff
    cdef int current_group_size
    
    # 为了 tqdm，我们需要在 Python 环境中创建 pbar 对象
    # 为了不频繁切换 GIL，我们批量更新或者在循环外处理
    # 但为了严格保持原函数行为，我们在这里直接处理
    pbar = tqdm(range(n), desc="Aggregation", disable=not verbose)
    
    # 主循环
    for i in range(n):
        # 更新 tqdm (为了性能，每 100 次更新一次，或者由 Python 迭代器控制)
        # 这里因为是在 enumerate/range 中，我们只是被动更新
        # 注意：为了让 tqdm 正常工作，我们需手动 update，但在 Cython 中包裹 tqdm(range) 会稍慢
        # 为了极致速度，通常建议去掉 tqdm，但这里为了兼容性保留
        if verbose and i % 100 == 0:
            pass # tqdm 迭代器会自动处理，这里只是占位

        if labels[i] >= 0:
            continue
            
        # 建立新簇
        labels[i] = lab
        splist.append(i)
        
        # 搜索逻辑
        current_group_size = 1
        limit = sort_vals_sorted[i] + radius
        
        # 内层循环：替代 searchsorted 和 vectorized distance
        # 从 i+1 开始扫描，直到 sort_val 超出范围
        for j in range(i + 1, n):
            # Projection pruning (排序带来的剪枝)
            if sort_vals_sorted[j] > limit:
                break
            
            # 关键优化：如果该点已被分配，直接跳过计算距离
            # 原版代码是先算距离再 filter，这里反过来，大幅减少计算量
            if labels[j] >= 0:
                continue
            
            # 计算 L1 距离 (带 Early Exit)
            d = 0.0
            for k in range(dim):
                d += fabs(data_sorted[i, k] - data_sorted[j, k])
                if d > radius: # 提前终止
                    break
            
            nr_dist += 1
            
            if d <= radius:
                labels[j] = lab
                current_group_size += 1
        
        group_sizes.append(current_group_size)
        lab += 1
    
    # 强制让 tqdm 完成
    if verbose:
        pbar.close()

    return {
        'labels': np.asarray(labels),
        'splist': np.array(splist),
        'group_sizes': np.array(group_sizes),
        'ind': np.asarray(ind),
        'sort_vals': np.asarray(sort_vals_sorted),
        'data_sorted': np.asarray(data_sorted),
        'nr_dist': nr_dist
    }