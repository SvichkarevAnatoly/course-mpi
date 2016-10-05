cmake_minimum_required(VERSION 2.8)
project(03_seidel_method)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(03_seidel_method ${SOURCE_FILES})