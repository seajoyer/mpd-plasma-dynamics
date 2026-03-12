cmake_minimum_required(VERSION 3.25)

set(CMAKE_C_COMPILER   gcc)
set(CMAKE_CXX_COMPILER g++)

project(mpd-plasma-dynamics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

find_package(MPI    REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(VTK REQUIRED COMPONENTS
    CommonCore
    CommonDataModel
    IOLegacy
    ChartsCore
    ViewsContext2D
    RenderingContext2D
    IOImage
    RenderingContextOpenGL2
    RenderingOpenGL2
)

find_package(yaml-cpp REQUIRED)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/array2d.cpp
    src/config.cpp
    src/mpi_manager.cpp
    src/grid.cpp
    src/fields.cpp
    src/diagnostics.cpp
    src/io_manager.cpp
    src/solver.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE src include)

target_link_libraries(${PROJECT_NAME} PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${VTK_LIBRARIES}
    yaml-cpp::yaml-cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall -Wextra -Wpedantic -O3
)

# VTK requires this macro to register its factory initializers.
vtk_module_autoinit(
    TARGETS ${PROJECT_NAME}
    MODULES ${VTK_LIBRARIES}
)

# --------------------------------------------------------------------------
# Status messages
# --------------------------------------------------------------------------
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
