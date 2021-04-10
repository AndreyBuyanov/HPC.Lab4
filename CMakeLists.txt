cmake_minimum_required (VERSION 3.0)

cmake_policy(SET CMP0054 NEW)

set(CMAKE_CXX_STANDARD 17)

project(HPC.Lab4)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include(FindCUDA)
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA rt library: ${CUDA_CUDART_LIBRARY}")
message(STATUS "CUDA nvcc: ${CUDA_NVCC_EXECUTABLE}")

add_subdirectory(App.2)
