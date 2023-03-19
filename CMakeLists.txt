cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(elnino-cuda VERSION 0.1 LANGUAGES CXX)


# Handle the unpleasantness of university deployment
if($ENV{USER} STREQUAL jovyan)
  set(CUDAToolkit_ROOT /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/include)
  set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/bin/nvcc)
endif()

enable_language(CUDA)

include(FindCUDAToolkit)

add_executable(sw_cuda src/sw_cuda.cu) 

target_compile_features(sw_cuda PUBLIC cxx_std_14)
set_target_properties(sw_cuda PROPERTIES
  CUDA_ARCHITECTURES 80-real
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sw_cuda PRIVATE -O3)
