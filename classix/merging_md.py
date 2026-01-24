import numpy as np
from tqdm import tqdm
from copy import deepcopy

def merge_manhattan(
    spdata,              # group centers (sorted space, shape: n_groups x dim)
    group_sizes,         # 每個 group 的大小 (array, length n_groups)
    sort_vals_sp,        # group centers 對應的 sort_vals (sorted, length n_groups)
    agg_labels_sp,       # 初始 group labels（通常是 np.arange(n_groups)）
    radius,
    mergeScale,
    minPts,
    mergeTinyGroups,
    verbose=False
):
    n_groups = len(spdata)
    Adj = np.zeros((n_groups, n_groups), dtype=np.int8)
    label_sp = agg_labels_sp.copy()  # 複製初始 label

    # 第一步：構建 Adj + 邊構建邊合併（這是造成差異的關鍵）
    pbar = tqdm(range(n_groups), desc="Building adjacency", disable=not verbose)
    for i in pbar:
        if not mergeTinyGroups and group_sizes[i] < minPts:
            continue

        xi = spdata[i]
        search_radius = mergeScale * radius + sort_vals_sp[i]
        last_j = np.searchsorted(sort_vals_sp, search_radius, side='right')

        if last_j > i:
            # L1 距離（用 sum(abs) 更快，等價於 ord=1）
            dists = np.sum(np.abs(spdata[i:last_j] - xi), axis=1)
            inds_rel = np.where(dists <= mergeScale * radius)[0]
            inds = i + inds_rel

            if not mergeTinyGroups:
                valid = group_sizes[inds] >= minPts
                inds = inds[valid]

            # 更新 Adj
            Adj[i, inds] = 1
            Adj[inds, i] = 1

            # 關鍵：邊構建邊合併！
            connected_labels = np.unique(label_sp[inds])
            if len(connected_labels) > 1:
                minlab = np.min(connected_labels)
                for lbl in connected_labels:
                    label_sp[label_sp == lbl] = minlab

    # 第二步：minPts 小簇重分配（完全匹配原版）
    ul = np.unique(label_sp)
    cs = np.zeros(len(ul), dtype=int)
    group_sizes = np.array(group_sizes)

    # 重新編號為 0~k-1，並計算每個 cluster 的大小
    new_label_map = {old: new for new, old in enumerate(ul)}
    for old, new in new_label_map.items():
        mask = (label_sp == old)
        cs[new] = np.sum(group_sizes[mask])
        label_sp[mask] = new

    small_clusters = np.where(cs < minPts)[0]

    label_sp_copy = label_sp.copy()

    pbar_small = tqdm(small_clusters, desc="minPts merging", disable=not verbose or len(small_clusters) == 0)
    for cluster_id in pbar_small:
        group_ids = np.where(label_sp_copy == cluster_id)[0]
        for gid in group_ids:
            xi = spdata[gid]
            dists = np.sum(np.abs(spdata - xi), axis=1)  # L1 到所有 group centers
            order = np.argsort(dists)

            for nearest_gid in order:
                target_cluster = label_sp_copy[nearest_gid]
                if cs[target_cluster] >= minPts:
                    label_sp[gid] = target_cluster
                    Adj[gid, nearest_gid] = 2
                    Adj[nearest_gid, gid] = 2
                    break

    ul_final = np.unique(label_sp)
    final_map = {old: new for new, old in enumerate(ul_final)}
    label_sp = np.array([final_map[l] for l in label_sp])

    return {
        'group_cluster_labels': label_sp, 
        'Adj': Adj,
        'final_cluster_sizes': np.bincount(label_sp)
    }