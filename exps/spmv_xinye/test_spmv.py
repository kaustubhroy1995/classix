import numpy as np
from scipy.sparse import csr_matrix
from spmv import spsubmatxvec

A = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [5.0, 0.0, 6.0]])
A_sparse = csr_matrix(A)

v = np.array([1.0, 1.0, 1.0])
result = np.zeros(3, dtype=np.float64)

spsubmatxvec(A_sparse.data, A_sparse.indptr, A_sparse.indices, 0, 3, v, result)
print("Result:", result)  # 應輸出 [3. 7. 11.]