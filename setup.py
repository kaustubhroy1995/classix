# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

class CustomBuildExt(build_ext):
    def run(self):
        self.include_dirs.append(np.get_include())
        super().run()

ext_modules = [
    Extension("classix.aggregate_ed_c", ["classix/aggregate_ed_c.pyx"]),
    Extension("classix.aggregate_ed_cm", ["classix/aggregate_ed_cm.pyx"]),
    Extension("classix.merge_ed_cm", ["classix/merge_ed_cm.pyx"]),
    Extension("classix.merge_ed_cm_win", ["classix/merge_ed_cm_win.pyx"]),
    
    Extension("classix.aggregate_md_cm", ["classix/aggregate_md_cm.pyx"]),
    Extension("classix.merge_md_cm", ["classix/merge_md_cm.pyx"]),
    Extension(
            "classix.aggregate_td_cm",
            sources=["classix/aggregate_td_cm.pyx"],
            language="c++",  # This is the critical line
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"]
        ),
    Extension("classix.merge_td_cm", ["classix/merge_td_cm.pyx"]),

    Extension(
        "spmv",                          # 模組名：import spmv
        sources=["classix/spmv.c"],              #（在項目根目錄或 classix/ 下）
        include_dirs=[np.get_include()], # numpy 頭文件（如果需要 array 交互）
        extra_compile_args=['-O3']       # 優化
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
)