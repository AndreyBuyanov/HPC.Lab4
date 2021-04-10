cmake_minimum_required (VERSION 3.0)

project(App.2)

enable_language("CUDA")

file(GLOB SOURSES "*.cpp" "*.hpp")
file(GLOB CUDA_SUORCES "*.cu")

find_package(FreeImage)

include_directories(
	${FreeImage_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} ${SOURSES} ${CUDA_SUORCES})

target_link_libraries(${PROJECT_NAME}
	${FreeImage_LIBRARY}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60)
