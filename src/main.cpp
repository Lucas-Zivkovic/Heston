#include <iostream>
#include <iomanip>
#include "../include/heston.hpp"

int main() {
    // ── Paramètres du modèle ────────────────────────────────────────────────
    HestonParams p {
        .kappa = 2.0,    // vitesse de mean-reversion
        .v0    = 0.04,   // variance initiale
        .theta = 0.04,   // variance long-terme
        .vol   = 0.3,    // vol de vol
        .cor   = -0.7,   // corrélation S-v
    };

    hestonPrice params {
        .S0 = 100.0,
        .r  = 0.05,
        .T  = 1.0,
        .p  = p
    };

    // ── Paramètres FFT ──────────────────────────────────────────────────────
    size_t N   = 4096;
    double eta   = 0.25;
    double alpha = 1.5;
    double K     = 100.0;   // strike ATM

    // ── Prix via Carr-Madan FFT ─────────────────────────────────────────────
    Heston heston;
    double price = heston.heston_fft_call_price(params, N, eta, alpha, K);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Heston Call Price (S0=100, K=100, T=1, r=5%): "
              << price << "\n";

    return 0;
}