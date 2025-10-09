cmake_minimum_required(VERSION 3.16)
project(cpp_openmp_project)

# Enable C++17 (or adjust to your preferred standard)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable compile_commands.json generation for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force use of Clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Help CMake find OpenMP for Clang
# This is important for NixOS environments
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

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Check if OpenMP was found
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP C++ flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP C++ libraries: ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

# Add your executable (replace 'main.cpp' with your source files)
add_executable(${PROJECT_NAME} main.cpp)

# Link OpenMP to your executable
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Clang-specific compiler flags
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
