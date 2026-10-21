#include <benchmark/benchmark.h>
#include "heston.hpp"

static hestonPrice make_params() {
    HestonParams p{ .kappa=2.0, .v0=0.04, .theta=0.04, .vol=0.3, .cor=-0.7 };
    return hestonPrice{ .S0=100.0, .r=0.05, .T=1.0, .q=0.0, .p=p };
}

// Grille complète — N impair obligatoire pour Simpson
static void BM_HestonCallsGrid(benchmark::State& state) {
    Heston h;
    auto params = make_params();
    size_t N = static_cast<size_t>(state.range(0));
    double eta = 0.25, alpha = 1.5;
    for (auto _ : state) {
        auto [k, calls] = h.heston_calls_price(params, N, eta, alpha);
        benchmark::DoNotOptimize(calls);
    }
    state.SetComplexityN(static_cast<int64_t>(N));
}
BENCHMARK(BM_HestonCallsGrid)
    ->Args({512})->Args({1024})->Args({2048})->Args({4096})->Args({8192})
    ->Complexity(benchmark::oNLogN);

// Prix d'un seul call
static void BM_HestonSingleCall(benchmark::State& state) {
    Heston h;
    auto params = make_params();
    size_t N = 4096;
    double eta = 0.25, alpha = 1.5, K = 100.0;
    for (auto _ : state) {
        double p = h.heston_fft_call_price(params, N, eta, alpha, K);
        benchmark::DoNotOptimize(p);
    }
}
BENCHMARK(BM_HestonSingleCall)->Unit(benchmark::kMillisecond);