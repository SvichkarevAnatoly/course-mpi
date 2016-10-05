cmake_minimum_required(VERSION 3.5)
project(04_seidel_mpi)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_FLAGS mpicxx)

set(SOURCE_FILES main.c)
add_executable(04_seidel_mpi ${SOURCE_FILES})