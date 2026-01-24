from setuptools import setup, Extension
import numpy as np

spmv_module = Extension(
    'spmv',
    sources=['spmv.c'],
    include_dirs=[np.get_include()], 
    extra_compile_args=['-O3', '-march=native']
)

setup(
    name='spmv',
    version='2.0',
    description='Sparse submatrix-vector multiplication for CLASSIX_T',
    ext_modules=[spmv_module]
)