cmake_minimum_required(VERSION 2.8)
project(02_simple_iterations)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(02_simple_iterations ${SOURCE_FILES})
