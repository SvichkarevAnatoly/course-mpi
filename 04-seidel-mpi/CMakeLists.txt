cmake_minimum_required(VERSION 2.8)
project(04_seidel_mpi)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(SOURCE_FILES main.c)
add_executable(04_seidel_mpi ${SOURCE_FILES})