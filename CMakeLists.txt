cmake_minimum_required(VERSION 3.20)

project(OptionResearchEngine
    VERSION 0.1.0
    DESCRIPTION "Modular C++20 quantitative finance research library"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for tooling (clangd, include-what-you-use, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Debug for single-config generators (Ninja, Makefiles).
# Ignored by multi-config generators (Visual Studio).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Windows: use the shared runtime everywhere so GoogleTest links cleanly.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------
option(ORE_BUILD_TESTS    "Build unit tests"          ON)
option(ORE_BUILD_EXAMPLES "Build executable examples" OFF)

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# std::thread requires linking a threading library on POSIX (libpthread).
# CMake's FindThreads is the canonical way to pull it in; MSVC picks it up
# implicitly, so the imported target is a no-op there.
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(CompilerWarnings)

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(ORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION} configuration:")
message(STATUS "  Compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard   : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests    : ${ORE_BUILD_TESTS}")
message(STATUS "  Build examples : ${ORE_BUILD_EXAMPLES}")
message(STATUS "")
