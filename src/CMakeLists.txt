add_executable(sw_cuda sw_cuda.cu) 

target_compile_features(sw_cuda PUBLIC cxx_std_14)
set_target_properties(sw_cuda PROPERTIES
  CUDA_ARCHITECTURES 80-real
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(sw_cuda PRIVATE -O3)

add_executable(elnino_cuda elnino_cuda.cu) 

target_compile_features(elnino_cuda PUBLIC cxx_std_14)
set_target_properties(elnino_cuda PROPERTIES
  CUDA_ARCHITECTURES 80-real
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(elnino_cuda PRIVATE -O3 --ptxas-options=-v)

add_executable(elnino_seq elnino_seq.cpp)
target_compile_features(elnino_seq PUBLIC cxx_std_14)
