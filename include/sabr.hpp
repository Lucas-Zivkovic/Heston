#pragma once
#include <vector>
#include <tuple>

struct SabrParams {
    double alpha;   // volatilité initiale
    double beta;    // élasticité (0 ≤ beta ≤ 1)
    double rho;     // corrélation
    double nu;      // vol de vol

    bool verif() const;
};

struct SabrMarket {
    double F;   // forward
    double T;   // maturité
    double r;   // taux sans risque
};

class Sabr {
public:
    Sabr()  = default;
    ~Sabr() = default;

    // Formule de Hagan et al. (2002) : volatilité implicite Black-Scholes
    double implied_vol(
        const SabrParams& params,
        const SabrMarket& market,
        double K) const;

    // Prix d'un call via la vol implicite SABR + Black-Scholes
    double call_price(
        const SabrParams& params,
        const SabrMarket& market,
        double K) const;

    // Volatilités implicites sur une grille de strikes
    std::tuple<std::vector<double>, std::vector<double>> vol_smile(
        const SabrParams& params,
        const SabrMarket& market,
        const std::vector<double>& strikes) const;
};
