cmake_minimum_required (VERSION 3.17)

project(App.V2 LANGUAGES CXX)

include(FindCUDAToolkit REQUIRED)

find_package(CLI11 CONFIG REQUIRED)
find_package(freeimage CONFIG REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

file(GLOB HEADERS "*.hpp")
file(GLOB SOURSES "*.cpp")

add_executable(${PROJECT_NAME}
	${HEADERS}
	${SOURSES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
	MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	CUDA::cudart
	CLI11::CLI11
	freeimage::FreeImage
	kernels.V2
)

