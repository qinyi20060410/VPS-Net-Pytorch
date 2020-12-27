cmake_minimum_required(VERSION 3.15)
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

set(TENSORRT_ROOT "/home/terminal/Software/TensorRT-7.2.2.3")
find_path(TENSORRT_INCLUDE_DIRS NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
    message(ERROR "Cannot find TensorRT library.")
endif()

include_directories(${TENSORRT_INCLUDE_DIRS})
link_libraries(${TENSORRT_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(deploy main.cpp include/l2_norm_plugin.h src/l2_norm_plugin.cpp)
target_link_libraries(deploy ${CUDA_LIBRARIES} ${TENSORRT_LIBRARIES})