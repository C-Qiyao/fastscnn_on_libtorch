cmake_minimum_required(VERSION 3.0)
project(torch_demo)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE "Debug")
set(PROJECT_DIR "/home/qiyao/codes/torch_demo")
#set(Torch_DIR "/home/qiyao/LIBS/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch/share/cmake/Torch")
#set(OPTION "-O3 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION} ${TORCH_CXX_FLAGS} ")
message(STATUS"Build type: ${CMAKE_BUILD_TYPE}")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")

set(Torch_DIR  "/home/qiyao/LIBS/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV library status:")
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "libtorch library status:")
message(STATUS "    config: ${Torch_DIR}")
message(STATUS "    version: ${Torch_VERSION}")

link_directories("${PROJECT_DIR}/lib/64")

include_directories(${PROJECT_DIR} )

include_directories("${PROJECT_DIR}/include" "/usr/include/X11" "/usr/include")

link_libraries("${PROJECT_DIR}/lib/64/libMvCameraControl.so")

add_executable(torch_demo main.cpp)

add_library(segnet SHARED network.cpp)
add_library(camera_class SHARED camera_class.cpp)
add_library(algroth SHARED algroth.cpp)

target_link_libraries(torch_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS} )
target_link_libraries(torch_demo libMvCameraControl.so -lX11)
target_link_libraries(torch_demo algroth  segnet camera_class)
target_include_directories(torch_demo PRIVATE CUDA_INCLUDE_DIRS)



