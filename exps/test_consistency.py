import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt

# Assume the files are in the same directory or importable path
from classix import CLASSIX          # Merged version (supports multiple metrics)
from classix_tm.classix_m import CLASSIX_M      # Original pure Manhattan version
from classix_tm.classix_t import CLASSIX_T      # Original pure Tanimoto version

def compare_labels(labels1, labels2, name=""):
    """Compare two label arrays for consistency (ignore label numbering, only check partitioning)"""
    if len(labels1) != len(labels2):
        print(f"[{name}] Length mismatch!")
        return False
    
    # Convert labels to a partition representation (set of frozensets)
    def to_partition(labels):
        d = {}
        for i, lbl in enumerate(labels):
            if lbl not in d:
                d[lbl] = set()
            d[lbl].add(i)
        return frozenset(frozenset(s) for s in d.values())
    
    part1 = to_partition(labels1)
    part2 = to_partition(labels2)
    
    equal = part1 == part2
    print(f"[{name}] Partitioning fully consistent: {equal}")
    
    if not equal:
        unique1 = np.unique(labels1)
        unique2 = np.unique(labels2)
        print(f"  Merged version clusters: {len(unique1)}, Original version clusters: {len(unique2)}")
    
    return equal

def run_test(X, metric='manhattan', radius=0.5, minPts=3, mergeScale=1.5, mergeTinyGroups=True, verbose=1):
    """
    Run consistency test for a given metric.
    - For 'manhattan': compares merged CLASSIX (metric='manhattan') vs original CLASSIX_M
    - For 'tanimoto': compares merged CLASSIX (metric='tanimoto') vs original CLASSIX_T
    """
    print("\n" + "="*60)
    print(f"Test parameters: metric={metric}, radius={radius}, minPts={minPts}, "
          f"mergeScale={mergeScale}, mergeTinyGroups={mergeTinyGroups}, n_samples={X.shape[0]}")
    print("="*60)
    
    # Merged version with specified metric
    clx_merged = CLASSIX(
        metric=metric,
        radius=radius,
        minPts=minPts,
        mergeScale=mergeScale,
        mergeTinyGroups=mergeTinyGroups,
        group_merging='distance',  # Assuming distance-based merging for both
        verbose=verbose
    )
    clx_merged.fit(X)
    labels_merged = clx_merged.labels_
    
    # Original pure version
    if metric == 'manhattan':
        clx_original = CLASSIX_M(
            radius=radius,
            minPts=minPts,
            mergeScale=mergeScale,
            mergeTinyGroups=mergeTinyGroups,
            verbose=verbose
        )
        title_merged = "Merged version (metric='manhattan')"
        title_original = "Original CLASSIX_M"
    elif metric == 'tanimoto':
        clx_original = CLASSIX_T(
            radius=radius,
            minPts=minPts,
            mergeScale=mergeScale,
            mergeTinyGroups=mergeTinyGroups,
            verbose=verbose
        )
        title_merged = "Merged version (metric='tanimoto')"
        title_original = "Original CLASSIX_T"
    else:
        raise ValueError("Unsupported metric for testing")
    
    clx_original.fit(X)
    labels_original = clx_original.labels
    
    consistent = compare_labels(labels_merged, labels_original, "Consistency check")
    
    print(f"Merged version labels examples (top 10): {labels_merged[:10]}")
    print(f"Original version labels examples (top 10): {labels_original[:10]}")
    
    if X.shape[1] == 2:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X[:,0], X[:,1], c=labels_merged, cmap='tab10', s=30)
        plt.title(title_merged)
        
        plt.subplot(1, 2, 2)
        plt.scatter(X[:,0], X[:,1], c=labels_original, cmap='tab10', s=30)
        plt.title(title_original)
        
        plt.tight_layout()
        plt.show()
    
    return consistent

# ===================================================================
# Generate test datasets
# ===================================================================
np.random.seed(42)

# Test 1: Simple blobs (easy to cluster)
X_blob, _ = make_blobs(n_samples=800, centers=5, cluster_std=0.5, random_state=42)

# Test 2: Moons (non-convex)
X_moon, _ = make_moons(n_samples=600, noise=0.08, random_state=42)

# Test 3: Circles with noise
X_circle, _ = make_circles(n_samples=700, noise=0.07, factor=0.4, random_state=42)
X_circle += np.random.normal(0, 0.5, X_circle.shape) * 0.3  # Add some noise

# ===================================================================
# Run tests
# ===================================================================

# Manhattan consistency tests
print("Test 1: Blob data (Manhattan)")
run_test(X_blob, metric='manhattan', radius=0.3, minPts=4, mergeScale=1)

print("\nTest 2: Moons data (Manhattan)")
run_test(X_moon, metric='manhattan', radius=0.18, minPts=5, mergeScale=1)

print("\nTest 3: Circles + noise (Manhattan)")
run_test(X_circle, metric='manhattan', radius=0.22, minPts=6, mergeScale=1)

# Tanimoto consistency tests 
print("\nTest 4: Blob data (Tanimoto)")
run_test(X_blob, metric='tanimoto', radius=0.4, minPts=5, mergeScale=1.2)

print("\nTest 5: Moons data (Tanimoto)")
run_test(X_moon, metric='tanimoto', radius=0.3, minPts=5, mergeScale=1.1)

print("\nTest 6: Circles + noise (Tanimoto)")
run_test(X_circle, metric='tanimoto', radius=0.35, minPts=6, mergeScale=1.15)