#include "fft.hpp"
#include <stdexcept>

void BitReversal(std::vector<std::complex<double>>& vect) {
    size_t n = vect.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) std::swap(vect[i], vect[j]);
    }
}

void fft(std::vector<std::complex<double>>& vect, bool inverse) {
    size_t n = vect.size();
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::runtime_error("La taille n'est pas une puissance de 2");

    BitReversal(vect);

    for (size_t len = 2; len <= n; len <<= 1) {          // ← i <<= 1 correct
        double angle = (inverse ? 1.0 : -1.0) * 2.0 * PI / static_cast<double>(len);
        std::complex<double> wm = std::exp(std::complex<double>(0.0, angle));

        for (size_t k = 0; k < n; k += len) {
            std::complex<double> w = 1.0;
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> t = w * vect[k + j + len / 2];
                std::complex<double> u = vect[k + j];
                vect[k + j]           = u + t;
                vect[k + j + len / 2] = u - t;
                w *= wm;
            }
        }
    }

    if (inverse) {
        for (auto& x : vect)
            x /= static_cast<double>(n);
    }
}

// N doit être impair (nombre de points Simpson composé : intervalles N-1 pair)
std::vector<double> _simpson_weights(int N, double eta) {
    if (N % 2 == 0)
        throw std::runtime_error("N n'est pas impair");
    std::vector<double> w(static_cast<size_t>(N));
    double x = eta / 3.0;
    for (int i = 0; i < N; ++i) {
        if (i == 0 || i == N - 1)  w[static_cast<size_t>(i)] = x;
        else if (i % 2 == 1)       w[static_cast<size_t>(i)] = 4.0 * x;
        else                       w[static_cast<size_t>(i)] = 2.0 * x;
    }
    return w;
}