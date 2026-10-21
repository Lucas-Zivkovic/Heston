#pragma once
#include <vector>
#include <complex>
#include <cmath>

inline const std::complex<double> I(0.0, 1.0);
inline const double PI = std::acos(-1.0);

void fft(std::vector<std::complex<double>>& vect, bool inverse);
std::vector<double> _simpson_weights(int N, double eta);