#include <gtest/gtest.h>
#include "heston.hpp"
#include <cmath>
#include <iostream>

static hestonPrice make_params() {
    HestonParams p{ .kappa=2.0, .v0=0.04, .theta=0.04, .vol=0.3, .cor=-0.7 };
    return hestonPrice{ .S0=100.0, .r=0.05, .T=1.0, .q=0.0, .p=p };
}

static constexpr size_t N_FFT = 4096;

TEST(HestonPriceTest, ATMCallPositive) {
    Heston h;
    auto params = make_params();
    size_t N = N_FFT; double eta = 0.25, alpha = 1.5, K = 100.0;
    double price = h.heston_fft_call_price(params, N, eta, alpha, K);
    EXPECT_GT(price, 0.0);
}

TEST(HestonPriceTest, OTMlessThanATM) {
    Heston h;
    auto params = make_params();
    size_t N = N_FFT; double eta = 0.25, alpha = 1.5;
    double Katm = 100.0, Kotm = 130.0;
    double atm = h.heston_fft_call_price(params, N, eta, alpha, Katm);
    double otm = h.heston_fft_call_price(params, N, eta, alpha, Kotm);
    EXPECT_LT(otm, atm);
}

TEST(HestonPriceTest, LowerBound) {
    Heston h;
    auto params = make_params();
    size_t N = N_FFT; double eta = 0.25, alpha = 1.5, K = 100.0;
    double price = h.heston_fft_call_price(params, N, eta, alpha, K);
    double lb = std::max(params.S0 - K * std::exp(-params.r * params.T), 0.0);
    EXPECT_GE(price, lb - 1e-4);
}

TEST(HestonPriceTest, GridAllNonNegative) {
    Heston h;
    auto params = make_params();
    size_t N = 512; double eta = 0.25, alpha = 1.5;
    auto [k, calls] = h.heston_calls_price(params, N, eta, alpha);
    for (size_t i = 0; i < N; ++i)
        EXPECT_GE(calls[i], 0.0) << "i=" << i;
}