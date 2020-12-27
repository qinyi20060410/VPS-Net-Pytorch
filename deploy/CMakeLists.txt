cmake_minimum_required(VERSION 3.17)
project(deploy)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("cuda libraries:${CUDA_LIBRARIES}")
    message("cuda include dirs:${CUDA_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "CUDA Not Found!")
endif()
include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARIES})

set(TENSORRT_ROOT "/home/terminal/Software/TensorRT-7.0.0.11")
set(TENSORRT_INCLUDE_DIRS "${TENSORRT_ROOT}/include")
set(TENSORRT_LIBRARIES "${TENSORRT_ROOT}/lib")
include_directories(${TENSORRT_INCLUDE_DIRS})
link_libraries(${TENSORRT_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(deploy main.cpp src/alex.cpp)
target_link_libraries(deploy ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})