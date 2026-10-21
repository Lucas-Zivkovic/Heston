#include <gtest/gtest.h>
#include "heston.hpp"
#include <complex>
#include <cmath>

// Paramètres de référence (Heston 1993)
static hestonPrice make_params() {
    HestonParams p{ .kappa=2.0, .v0=0.04, .theta=0.04, .vol=0.3, .cor=-0.7 };
    return hestonPrice{ .S0=100.0, .r=0.05, .T=1.0, .p=p };
}

// ── CF(0) = 1 (par définition de toute fonction caractéristique) ──────────────
TEST(HestonCFTest, AtZeroEqualsOne) {
    Heston h;
    auto params = make_params();
    std::complex<double> u = 0.0;
    auto cf = h.hestoncf(params, u);
    EXPECT_NEAR(std::abs(cf), 1.0, 1e-10);
}

// ── |CF(u)| ≤ 1 pour tout u réel ─────────────────────────────────────────────
TEST(HestonCFTest, ModulusLeOne) {
    Heston h;
    auto params = make_params();
    for (double v : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        std::complex<double> u = v;
        auto cf = h.hestoncf(params, u);
        EXPECT_LE(std::abs(cf), 1.0 + 1e-9) << "u=" << v;
    }
}

// ── La CF doit être conjuguée symétrique : CF(-u) = conj(CF(u)) ───────────────
TEST(HestonCFTest, ConjugateSymmetry) {
    Heston h;
    auto params = make_params();
    for (double v : {1.0, 3.0, 7.0}) {
        std::complex<double> up = v, um = -v;
        auto cfp = h.hestoncf(params, up);
        auto cfm = h.hestoncf(params, um);
        EXPECT_NEAR(cfp.real(),  cfm.real(), 1e-9) << "u=" << v;
        EXPECT_NEAR(cfp.imag(), -cfm.imag(), 1e-9) << "u=" << v;
    }
}