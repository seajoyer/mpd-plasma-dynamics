include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageFetcher.cmake)

fetch_package(VTK       https://github.com/Kitware/VTK.git            v9.5.2)
fetch_package(yaml-cpp  https://github.com/jbeder/yaml-cpp.git         0.8.0)

find_package(MPI    REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
