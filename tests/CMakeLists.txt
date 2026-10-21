# ── Tests unitaires (Google Test) ────────────────────────────────────────────

add_executable(heston_tests
    test_fft.cpp
    test_heston_cf.cpp
    test_heston_price.cpp
    test_sabr.cpp
)

target_link_libraries(heston_tests
    PRIVATE
        heston_lib
        GTest::gtest_main   # fournit main() automatiquement
)

target_include_directories(heston_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(heston_tests PRIVATE
    -Wall -Wextra -Wpedantic
)

# Enregistre chaque test auprès de CTest (découverte automatique)
include(GoogleTest)
gtest_discover_tests(heston_tests)