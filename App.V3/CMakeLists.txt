cmake_minimum_required (VERSION 3.0)

project(App.V3 LANGUAGES CXX CUDA)

include(FindCUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

file(GLOB HEADERS "*.hpp")
file(GLOB SOURSES "*.cpp")
file(GLOB CUDA_SOURSES "*.cuh" "*.cu")

add_executable(${PROJECT_NAME}
	${HEADERS}
	${SOURSES}
	${CUDA_SOURSES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
	CUDA_SEPARABLE_COMPILATION OFF
	CUDA_ARCHITECTURES 75
)
