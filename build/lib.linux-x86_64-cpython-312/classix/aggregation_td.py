import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from spmv import spsubmatxvec  

def aggregate_tanimoto(data, radius, verbose=False):
    n, _ = data.shape
    
    sort_vals = np.sum(data, axis=1)
    
    # 排序
    ind = np.argsort(sort_vals, kind='stable')
    data_sorted = data[ind]
    sort_vals_sorted = sort_vals[ind]
    
    # 轉為稀疏 CSR 格式（加速內積）
    datas = sparse.csr_matrix(data_sorted)
    
    labels = np.full(n, -1, dtype=int)
    splist = []
    group_sizes = []
    lab = 0
    nr_dist = 0
    
    rhs = 1 / (1 - radius) + 1
    rhsi = 1 / rhs
    
    pbar = tqdm(range(n), desc="Aggregation", disable=not verbose)
    for i in pbar:
        if labels[i] >= 0:
            continue
        
        clustc = data_sorted[i]
        labels[i] = lab
        splist.append(i)
        group_sizes.append(1)
        
        search_radius = sort_vals_sorted[i] / (1 - radius)
        last_j = np.searchsorted(sort_vals_sorted, search_radius, side='right')
        
        if last_j > i + 1:
            n_rows = last_j - (i + 1)
            ips = np.zeros(n_rows, dtype=np.float64)
            
            spsubmatxvec(
                datas.data.astype(np.float64),
                datas.indptr.astype(np.int32),
                datas.indices.astype(np.int32),
                i + 1,
                last_j,
                clustc.astype(np.float64),
                ips
            )
            
            nr_dist += n_rows
            
            # Tanimoto similarity >= rhsi ⇔ distance <= radius
            denom = sort_vals_sorted[i+1:last_j] + sort_vals_sorted[i]
            vec = ips / denom
            vec_mask = vec >= rhsi
            
            notAssigned = labels[i+1:last_j] < 0
            reassignMask = np.logical_and(notAssigned, vec_mask)
            
            if np.any(reassignMask):
                labels[i+1:last_j][reassignMask] = lab
                group_sizes[-1] += np.sum(reassignMask)
        
        lab += 1
    
    return {
        'labels': labels,                     # 在 sorted space 的聚合標籤
        'splist': np.array(splist),           # starting points 在 sorted space 的索引
        'group_sizes': np.array(group_sizes),
        'ind': ind,                           # 原始到 sorted 的排序索引
        'sort_vals': sort_vals_sorted,        # sorted 後的 sort_vals
        'data_sorted': data_sorted,           # sorted 後的數據
        'nr_dist': nr_dist
    }