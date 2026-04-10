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
    src/verification.cpp

    # BC + geometry framework
    src/face_bc.cpp
    src/geometry_registry.cpp

    # BC implementation
    src/bcs/per_field_bc.cpp

    # Built-in geometry implementations
    src/geometries/short_cathode_parabolic.cpp
    src/geometries/short_cathode_cosine.cpp

    # IC implementation
    src/ics/expression_ic.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        src
        include
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        yaml-cpp::yaml-cpp
        VTK::IOLegacy
        exprtk::exprtk
)

# Both expression_ic.cpp and per_field_bc.cpp include the large exprtk header.
# Restrict optimisation on these TUs to keep compile times sane while retaining
# full optimisation for the hot numerical loops in solver.cpp / fields.cpp.
set_source_files_properties(
    src/ics/expression_ic.cpp
    src/bcs/per_field_bc.cpp
    PROPERTIES COMPILE_FLAGS "-O1"
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
message(STATUS "exprtk source dir   : ${exprtk_SOURCE_DIR}")
