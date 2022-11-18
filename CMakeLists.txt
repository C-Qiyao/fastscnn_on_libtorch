cmake_minimum_required(VERSION 3.0)
project(torch_demo)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")
#set(Torch_DIR "/home/qiyao/LIBS/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch/share/cmake/Torch")
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -Wall -Wextra")
message(STATUS"Build type: ${CMAKE_BUILD_TYPE}")



set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CMAKE_PREFIX_PATH "/home/qiyao/LIBS/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "libtorch library status:")
message(STATUS "    config: ${Torch_DIR}")
message(STATUS "    version: ${Torch_VERSION}")



add_executable(torch_demo testtorch.cpp)

target_link_libraries(torch_demo  ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(torch_demo PRIVATE CUDA_INCLUDE_DIRS)

