cmake_minimum_required(VERSION 3.25)

set(NAME mpd-plasma-dynamics)
project(${NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI REQUIRED COMPONENTS CXX)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Create executable
add_executable(${NAME}
    src/main.cpp
    src/memory.cpp
    src/geometry.cpp
    src/physics.cpp
    src/mpi_comm.cpp
    src/output/write_vtk.cpp
    src/output/write_plt.cpp
    src/output/write_results.cpp
    src/integrals.cpp
    src/boundary.cpp
    src/solver.cpp
)

target_include_directories(${NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${NAME} PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
)

target_compile_options(${NAME} PRIVATE 
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
