# ── Benchmarks (Google Benchmark) ────────────────────────────────────────────

add_executable(heston_benchmarks
    bench_fft.cpp
    bench_heston.cpp
)

target_link_libraries(heston_benchmarks
    PRIVATE
        heston_lib
        benchmark::benchmark_main   # fournit main() automatiquement
)

target_include_directories(heston_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Les benchmarks tournent toujours en Release pour des mesures fiables
target_compile_options(heston_benchmarks PRIVATE
    -O3 -march=native -ffast-math
)