#pragma once
#include <complex>
#include <cmath>
#include <vector>
#include <tuple>

// ── 1. HestonParams AVANT hestonPrice ────────────────────────────────────────
struct HestonParams {
    double kappa;
    double v0;
    double theta;
    double vol;
    double cor;

    bool verif() const {
        return 2 * kappa * theta > vol * vol;   // condition de Feller
    }
};

// ── 2. hestonPrice utilise HestonParams par valeur (pas const&) ──────────────
struct hestonPrice {
    double S0;
    double r;
    double T;
    double q;
    HestonParams p;   // par valeur, pas const&
};

// ── 3. Constructeur public ────────────────────────────────────────────────────
class Heston {
public:
    Heston()  = default;
    ~Heston() = default;

    std::complex<double> hestoncf(
        const hestonPrice& params,
        std::complex<double> u) const;

    std::complex<double> CarrMadan(
        double alpha,
        const hestonPrice& params,
        double v) const;

    std::tuple<std::vector<double>, std::vector<double>> heston_calls_price(
        const hestonPrice& params,
        size_t N,
        double eta,
        double alpha) const;

    double heston_fft_call_price(
        const hestonPrice& params,
        size_t N,
        double eta,
        double alpha,
        double K) const;
};