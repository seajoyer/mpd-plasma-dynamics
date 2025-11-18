cmake_minimum_required(VERSION 3.16)
project(mpd-plasma-dynamics)

# Enable C++17 (or adjust to your preferred standard)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compile_commands.json generation for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use MPI compilers (adjust if needed for your environment, e.g., mpicxx)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# Help CMake find additional paths if needed (retained from original for compatibility)
if(DEFINED ENV{CPATH})
    string(REPLACE ":" ";" CPATH_LIST $ENV{CPATH})
    foreach(path ${CPATH_LIST})
        include_directories(SYSTEM ${path})
    endforeach()
endif()

if(DEFINED ENV{LIBRARY_PATH})
    string(REPLACE ":" ";" LIBRARY_PATH_LIST $ENV{LIBRARY_PATH})
    foreach(path ${LIBRARY_PATH_LIST})
        link_directories(${path})
    endforeach()
endif()

# Check if MPI was found
if(MPI_CXX_FOUND)
    message(STATUS "MPI found")
    message(STATUS "MPI version: ${MPI_CXX_VERSION}")
    message(STATUS "MPI C++ compiler: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI C++ include dirs: ${MPI_CXX_INCLUDE_PATH}")
    message(STATUS "MPI C++ libraries: ${MPI_CXX_LIBRARIES}")

    message(STATUS "OpenMP found")
    message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP C++ flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP C++ libraries: ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "MPI and/or OpenMP not found")
endif()

# Add your executable (replace 'main.cpp' with your source files)
add_executable(${PROJECT_NAME} main.cpp)

# Link MPI to your executable
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Compiler flags (removed OpenMP-specific, added general ones; adjust as needed for MPI)
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -O2
)

# Optional: Print some useful information during configuration
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "compile_commands.json will be generated at: ${CMAKE_BINARY_DIR}/compile_commands.json")
