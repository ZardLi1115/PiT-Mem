import os
import platform
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile_args = ["-O3", "-std=c++17", "-fopenmp", "-fno-math-errno"]
extra_link_args = ["-fopenmp"]

machine = platform.machine().lower()
if "x86_64" in machine or "amd64" in machine:
    extra_compile_args.extend(["-mavx2", "-mfma"])

ext_modules = [
    Pybind11Extension(
        "pit_mem_core",
        ["src/bindings.cpp"],
        depends=[
            "include/bit_hnsw.hpp",
            "include/mvcc_provenance.hpp",
            "include/rap_compact.hpp",
            "include/baselines_cpp.hpp",
            "include/simd_kernels.hpp",
            "include/pit_types.hpp",
        ],
        include_dirs=["include"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    )
]

setup(
    name="pit_mem_core",
    version="0.1.0",
    description="PiT-Mem C++17 Bi-Temporal Vector-Graph Storage & Indexing Engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
