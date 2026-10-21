cmake_minimum_required(VERSION 3.20)
project(HestonFFT VERSION 1.0.0 LANGUAGES CXX)

# ── Standard ──────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type par défaut ──────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Flags ─────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3 -fsanitize=address,undefined -fno-omit-frame-pointer")

# ── Dépendances via FetchContent ───────────────────────────────────────────────
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
    GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ── Bibliothèque principale ────────────────────────────────────────────────────
add_library(heston_lib STATIC
    src/heston.cpp
    src/fft.cpp
    src/Sabr.cpp
)
target_include_directories(heston_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(heston_lib PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion
)

# ── Exécutable principal ───────────────────────────────────────────────────────
add_executable(heston_pricer src/main.cpp)
target_link_libraries(heston_pricer PRIVATE heston_lib)

# ── Module Python ─────────────────────────────────────────────────────────────
pybind11_add_module(heston python/binding.cpp)
target_link_libraries(heston PRIVATE heston_lib)

# ── Tests & Benchmarks ────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)

# ── Résumé ────────────────────────────────────────────────────────────────────
message(STATUS "──────────────────────────────────────────")
message(STATUS " HestonFFT")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++        : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "──────────────────────────────────────────")
