cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(elnino-cuda VERSION 0.1 LANGUAGES CXX CUDA)

add_executable(sw_cuda src/sw_cuda.cu) 

target_compile_features(sw_cuda PUBLIC cxx_std_14)
set_target_properties(sw_cuda PROPERTIES
  CUDA_ARCHITECTURES 80-real
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sw_cuda PRIVATE -O3)
