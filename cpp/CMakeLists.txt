cmake_minimum_required(VERSION 3.16)
project(pit_mem_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(pit_mem_core src/bindings.cpp)
target_include_directories(pit_mem_core PRIVATE include)
target_compile_options(pit_mem_core PRIVATE -O3 -mavx2 -mfma)
target_link_libraries(pit_mem_core PRIVATE OpenMP::OpenMP_CXX)
