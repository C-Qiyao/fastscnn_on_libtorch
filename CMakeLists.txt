cmake_minimum_required(VERSION 3.0)

project(torch_demo)

add_executable(torch_demo main.cpp)

install(TARGETS torch_demo RUNTIME DESTINATION bin)
