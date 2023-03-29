cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(elnino-cuda VERSION 0.1 LANGUAGES CXX)

# Handle the unpleasantness of university deployment
if($ENV{USER} STREQUAL jovyan)
  set(CUDAToolkit_ROOT /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/include)
  set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/bin/nvcc)
endif()

enable_language(CUDA)
include(FindCUDAToolkit)

if (MSVC)
    # warning level 4
    add_compile_options(/W4)
else()
    # additional warnings
    add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(src)
