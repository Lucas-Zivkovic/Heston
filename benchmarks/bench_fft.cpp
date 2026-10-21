#include <benchmark/benchmark.h>
#include <vector>
#include <complex>
#include "fft.hpp"

static void BM_FFT_Forward(benchmark::State& state) {
    const int N = state.range(0);
    std::vector<std::complex<double>> v(N, {1.0, 0.0});
    for (auto _ : state) {
        auto copy = v;
        fft(copy, false);
        benchmark::DoNotOptimize(copy);
    }
    state.SetComplexityN(N);
}
BENCHMARK(BM_FFT_Forward)
    ->RangeMultiplier(2)->Range(256, 1 << 16)
    ->Complexity(benchmark::oNLogN);

static void BM_FFT_Inverse(benchmark::State& state) {
    const int N = state.range(0);
    std::vector<std::complex<double>> v(N, {1.0, 0.0});
    for (auto _ : state) {
        auto copy = v;
        fft(copy, true);
        benchmark::DoNotOptimize(copy);
    }
    state.SetComplexityN(N);
}
BENCHMARK(BM_FFT_Inverse)
    ->RangeMultiplier(2)->Range(256, 1 << 16)
    ->Complexity(benchmark::oNLogN);