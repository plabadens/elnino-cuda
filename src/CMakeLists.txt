add_executable(sw_cuda sw_cuda.cu) 

target_compile_features(sw_cuda PUBLIC cxx_std_14)
set_target_properties(sw_cuda PROPERTIES
  CUDA_ARCHITECTURES 80-real
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sw_cuda PRIVATE -O3)

add_executable(sw_sequential sw_sequential.cpp)
target_compile_features(sw_sequential PUBLIC cxx_std_14)
