cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project(mpd-plasma-dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
)

target_compile_options(${PROJECT_NAME} PRIVATE 
    -Wall -Wextra -Wpedantic -O3
)

message(STATUS "Build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler        : ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP flags        : ${OpenMP_CXX_FLAGS}")
message(STATUS "MPI compile flags   : ${MPI_CXX_COMPILE_OPTIONS}")
message(STATUS "MPI include path    : ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI link flags      : ${MPI_CXX_LINK_FLAGS}")
message(STATUS "MPI libraries       : ${MPI_CXX_LIBRARIES}")
