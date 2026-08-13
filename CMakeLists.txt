cmake_minimum_required(VERSION 3.25)
project(qwen-3.5-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU Target Architectures: sm_80 (A100), sm_86 (RTX 30-series), sm_89 (RTX 40-series)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
endif()

# Compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

add_executable(qwen-3.5-cuda
    src/main.cpp
    src/loader_gguf.cpp
    src/layer_gdn.cpp
    src/cuda/matmul_q8.cu
    src/cuda/matmul_q4.cu
    src/cuda/ops.cu
    src/cuda/rope.cu
    src/cuda/gdn.cu
    src/cuda/moe.cu
)

target_link_libraries(qwen-3.5-cuda PRIVATE CUDA::cublas CUDA::cudart)
