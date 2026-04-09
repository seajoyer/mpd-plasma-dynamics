# fetch_package(<name> <repo> <tag>)
#
#   For packages that ship a CMakeLists.txt / CMake config.
#   Tries find_package first; falls back to FetchContent + build.
#
macro(fetch_package target_name repo tag)
    if(NOT TARGET ${target_name} AND NOT TARGET ${target_name}::${target_name})
        find_package(${target_name} QUIET CONFIG)
        if(NOT ${target_name}_FOUND AND NOT TARGET ${target_name}::${target_name})
            message(STATUS "[fetch_package] ${target_name} not found → downloading ${tag}")

            FetchContent_Declare(
                ${target_name}
                GIT_REPOSITORY ${repo}
                GIT_TAG        ${tag}
                GIT_SHALLOW    TRUE
                FIND_PACKAGE_ARGS CONFIG
            )

            # special handling for VTK
            if(${target_name} STREQUAL "VTK")
                message(STATUS "[fetch_package] Configuring minimal VTK build ...")

                set(VTK_BUILD_TESTING       OFF         CACHE BOOL   "")
                set(VTK_WRAP_PYTHON         OFF         CACHE BOOL   "")
                set(VTK_WRAP_JAVA           OFF         CACHE BOOL   "")
                set(VTK_USE_MPI             OFF         CACHE BOOL   "")
                set(VTK_USE_TK              OFF         CACHE BOOL   "")
                set(VTK_BUILD_DOCUMENTATION OFF         CACHE BOOL   "")

                set(VTK_GROUP_ENABLE_Imaging    DONT_WANT CACHE STRING "")
                set(VTK_GROUP_ENABLE_MPI        DONT_WANT CACHE STRING "")
                set(VTK_GROUP_ENABLE_Qt         DONT_WANT CACHE STRING "")
                set(VTK_GROUP_ENABLE_Rendering  DONT_WANT CACHE STRING "")
                set(VTK_GROUP_ENABLE_StandAlone DONT_WANT CACHE STRING "")
                set(VTK_GROUP_ENABLE_Views      DONT_WANT CACHE STRING "")
                set(VTK_GROUP_ENABLE_Web        DONT_WANT CACHE STRING "")

                set(VTK_MODULE_ENABLE_VTK_IOLegacy YES   CACHE STRING "")
            endif()

            FetchContent_MakeAvailable(${target_name})
        else()
            message(STATUS "[fetch_package] Found ${target_name}")
        endif()
    endif()
endmacro()

# fetch_header_only_package(<name> <repo> <tag>)
#
#   For header-only libraries that need no build step.
#   Populates the source tree and creates an INTERFACE imported target
#   <name>::<name> whose include directory is the fetched source root,
#   so callers can use target_link_libraries() uniformly.
#
macro(fetch_header_only_package target_name repo tag)
    if(NOT TARGET ${target_name}::${target_name})
        message(STATUS "[fetch_header_only_package] ${target_name} → downloading ${tag}")

        FetchContent_Declare(
            ${target_name}
            GIT_REPOSITORY ${repo}
            GIT_TAG        ${tag}
            GIT_SHALLOW    TRUE
        )

        FetchContent_GetProperties(${target_name})
        if(NOT ${target_name}_POPULATED)
            FetchContent_Populate(${target_name})
        endif()

        add_library(${target_name}::${target_name} INTERFACE IMPORTED)
        set_target_properties(${target_name}::${target_name} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${${target_name}_SOURCE_DIR}"
        )

        message(STATUS "[fetch_header_only_package] ${target_name} source dir : ${${target_name}_SOURCE_DIR}")
    endif()
endmacro()
