cmake_minimum_required(VERSION 3.25)

project(mpd-plasma-dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(FETCHCONTENT_QUIET ON)
set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)

include(FetchContent)
include(cmake/Dependencies.cmake)

add_executable(${PROJECT_NAME}
    # Core
    src/main.cpp
    src/array2d.cpp
    src/config.cpp
    src/mpi_manager.cpp
    src/grid.cpp
    src/fields.cpp
    src/diagnostics.cpp
    src/io_manager.cpp
    src/solver.cpp

    # BC + geometry framework
    src/face_bc.cpp
    src/bc_registry.cpp
    src/geometry_registry.cpp

    # Built-in BC implementations
    src/bcs/inflow_bc.cpp
    src/bcs/outflow_bc.cpp
    src/bcs/outer_wall_bc.cpp
    src/bcs/solid_wall_bc.cpp
    src/bcs/axis_symmetry_bc.cpp

    # Built-in geometry implementations
    src/geometries/coaxial_nozzle.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE src include)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        yaml-cpp::yaml-cpp
        VTK::IOLegacy
)

message(STATUS "Build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler        : ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard        : ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP flags        : ${OpenMP_CXX_FLAGS}")
message(STATUS "MPI compile flags   : ${MPI_CXX_COMPILE_OPTIONS}")
message(STATUS "MPI include path    : ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI link flags      : ${MPI_CXX_LINK_FLAGS}")
message(STATUS "MPI libraries       : ${MPI_CXX_LIBRARIES}")
message(STATUS "VTK version         : ${VTK_VERSION}")
message(STATUS "yaml-cpp version    : ${yaml-cpp_VERSION}")
