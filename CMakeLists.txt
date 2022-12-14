cmake_minimum_required(VERSION 3.0)
find_package(PythonInterp REQUIRED)
project(torch_demo)
project(testtorch)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE "Debug")
set(PROJECT_DIR "/home/qiyao/codes/torch_demo")

#set(Torch_DIR "/home/qiyao/LIBS/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch/share/cmake/Torch")
#set(OPTION "-O3 -Wall -Wextra")

if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0 ${TORCH_CXX_FLAGS} ")
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -Ofast ${TORCH_CXX_FLAGS} ")
    message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_C_FLAGS_RELEASE}")
    add_executable(test_release ${src_dirs})
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION} ${TORCH_CXX_FLAGS} ")
message(STATUS"Build type: ${CMAKE_BUILD_TYPE}")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")

#set(Torch_DIR  "/home/qiyao/LIBS/libtorch/share/cmake/Torch")
set(Torch_DIR  "/home/qiyao/LIBS/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "    OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libtorch library status:")
message(STATUS "    config: ${Torch_DIR}")
message(STATUS "    version: ${Torch_VERSION}")

link_directories("${PROJECT_DIR}/MVS/lib/64")
include_directories(${PROJECT_DIR})
include_directories("${PROJECT_DIR}/MVS/include" "/usr/include/X11" "/usr/include")

link_libraries("${PROJECT_DIR}/MVS/lib/64/libMvCameraControl.so")

add_executable(torch_demo main.cpp)

add_library(segnet SHARED SIGNET/network.cpp)
add_library(camera_class SHARED CAMERA/camera_class.cpp)
add_library(algroth SHARED CAMERA/algroth.cpp)

target_include_directories(torch_demo PRIVATE CUDA_INCLUDE_DIRS)

target_link_libraries(torch_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS} )
target_link_libraries(torch_demo libMvCameraControl.so -lX11)
target_link_libraries(torch_demo algroth  segnet camera_class)







