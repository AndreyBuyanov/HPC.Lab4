cmake_minimum_required (VERSION 3.17)

project(kernels.V2 LANGUAGES CUDA)

include(FindCUDAToolkit REQUIRED)

file(GLOB HEADERS "*.hpp")
file(GLOB CUDA_HEADERS "*.cuh")
file(GLOB CUDA_SOURSES "*.cu")

add_library(${PROJECT_NAME} STATIC
	${HEADERS}
	${CUDA_HEADERS}
	${CUDA_SOURSES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
	CUDA_SEPARABLE_COMPILATION OFF
	CUDA_ARCHITECTURES 75
)

target_include_directories(${PROJECT_NAME} PUBLIC
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
