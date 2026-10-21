#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include "fft.hpp"

static bool near_cx(std::complex<double> a, std::complex<double> b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

TEST(FFTTest, DiracSpectrum) {
    const int N = 8;
    std::vector<std::complex<double>> v(N, 0.0);
    v[0] = 1.0;
    fft(v, false);
    for (int i = 0; i < N; ++i)
        EXPECT_NEAR(std::abs(v[i]), 1.0, 1e-9) << "index " << i;
}

TEST(FFTTest, RoundTrip) {
    const int N = 16;
    std::vector<std::complex<double>> original(N), v(N);
    for (int i = 0; i < N; ++i)
        original[i] = v[i] = std::complex<double>(i * 0.5, i * 0.1);
    fft(v, false);
    fft(v, true);
    for (int i = 0; i < N; ++i)
        EXPECT_TRUE(near_cx(v[i], original[i])) << "index " << i;
}

TEST(FFTTest, ThrowsOnNonPowerOfTwo) {
    std::vector<std::complex<double>> v(6, 1.0);
    EXPECT_THROW(fft(v, false), std::runtime_error);
}

TEST(FFTTest, SizeOne) {
    std::vector<std::complex<double>> v = {{ 3.0, -1.0 }};
    EXPECT_NO_THROW(fft(v, false));
    EXPECT_TRUE(near_cx(v[0], { 3.0, -1.0 }));
}

TEST(SimpsonWeightsTest, SumIsCorrect) {
    const int N = 9;
    const double eta = 0.25;
    auto w = _simpson_weights(N, eta);
    double sum = 0.0;
    for (double wi : w) sum += wi;
    EXPECT_NEAR(sum, eta * (N - 1), 1e-12);
}

TEST(SimpsonWeightsTest, ThrowsOnEvenN) {
    EXPECT_THROW(_simpson_weights(8, 0.25), std::runtime_error);
}

TEST(SimpsonWeightsTest, Pattern) {
    const int N = 5;
    const double eta = 0.3;
    auto w = _simpson_weights(N, eta);
    double x = eta / 3.0;
    EXPECT_NEAR(w[0], x,   1e-12);
    EXPECT_NEAR(w[1], 4*x, 1e-12);
    EXPECT_NEAR(w[2], 2*x, 1e-12);
    EXPECT_NEAR(w[3], 4*x, 1e-12);
    EXPECT_NEAR(w[4], x,   1e-12);
}