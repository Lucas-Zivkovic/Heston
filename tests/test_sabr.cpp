#include <gtest/gtest.h>
#include "sabr.hpp"
#include <cmath>
#include <vector>

static SabrParams make_params() {
    return SabrParams{ .alpha=0.3, .beta=0.5, .rho=-0.3, .nu=0.4 };
}

static SabrMarket make_market() {
    return SabrMarket{ .F=100.0, .T=1.0, .r=0.05 };
}

// ── SabrParams::verif ─────────────────────────────────────────────────────────

TEST(SabrParamsTest, ValidParams) {
    SabrParams p = make_params();
    EXPECT_TRUE(p.verif());
}

TEST(SabrParamsTest, BetaZeroBoundary) {
    SabrParams p = make_params();
    p.beta = 0.0;
    EXPECT_FALSE(p.verif());
}

TEST(SabrParamsTest, BetaOneBoundary) {
    SabrParams p = make_params();
    p.beta = 1.0;
    EXPECT_FALSE(p.verif());
}

// ── Sabr::implied_vol ─────────────────────────────────────────────────────────

TEST(SabrImpliedVolTest, Positive) {
    Sabr s;
    auto p = make_params();
    auto m = make_market();
    EXPECT_GT(s.implied_vol(p, m, 100.0), 0.0);
}

TEST(SabrImpliedVolTest, ATMContinuous) {
    // La vol ATM (F=K) doit être continue : valeurs proches pour K proche de F
    Sabr s;
    auto p = make_params();
    auto m = make_market();
    double vol_atm   = s.implied_vol(p, m, m.F);
    double vol_near  = s.implied_vol(p, m, m.F * 1.001);
    EXPECT_NEAR(vol_atm, vol_near, 1e-3);
}

TEST(SabrImpliedVolTest, SmileSymmetryRho0) {
    // Avec rho=0 la smile doit être symétrique autour du forward
    Sabr s;
    auto p = make_params();
    p.rho  = 0.0;
    auto m = make_market();
    double otm_call = s.implied_vol(p, m, m.F * 1.1);
    double otm_put  = s.implied_vol(p, m, m.F / 1.1);
    // Symétrie approchée : résidu O((1-β)²log²(F/K)) attendu pour β≠1
    EXPECT_NEAR(otm_call, otm_put, 5e-3);
}

TEST(SabrImpliedVolTest, NegativeRhoSkew) {
    // rho < 0 → vol OTM put > vol OTM call (skew négatif)
    Sabr s;
    auto p = make_params();   // rho = -0.3
    auto m = make_market();
    double vol_otm_put  = s.implied_vol(p, m, m.F * 0.9);
    double vol_otm_call = s.implied_vol(p, m, m.F * 1.1);
    EXPECT_GT(vol_otm_put, vol_otm_call);
}

TEST(SabrImpliedVolTest, IncreaseWithNu) {
    // Une vol de vol plus grande doit aplatir/amplifier la smile
    Sabr s;
    auto p1 = make_params();
    auto p2 = make_params();
    p2.nu   = p1.nu * 2.0;
    auto m  = make_market();
    // Pour un strike loin du forward, la vol doit être différente
    double v1 = s.implied_vol(p1, m, m.F * 1.2);
    double v2 = s.implied_vol(p2, m, m.F * 1.2);
    EXPECT_NE(v1, v2);
}

// ── Sabr::call_price ──────────────────────────────────────────────────────────

TEST(SabrCallPriceTest, Positive) {
    Sabr s;
    EXPECT_GT(s.call_price(make_params(), make_market(), 100.0), 0.0);
}

TEST(SabrCallPriceTest, OTMlessThanATM) {
    Sabr s;
    auto p = make_params();
    auto m = make_market();
    double atm = s.call_price(p, m, m.F);
    double otm = s.call_price(p, m, m.F * 1.3);
    EXPECT_LT(otm, atm);
}

TEST(SabrCallPriceTest, LowerBound) {
    // Call >= max(F*e^{-rT} - K*e^{-rT}, 0) (borne inférieure d'arbitrage)
    Sabr s;
    auto p = make_params();
    auto m = make_market();
    double K     = 90.0;
    double price = s.call_price(p, m, K);
    double lb    = std::max((m.F - K) * std::exp(-m.r * m.T), 0.0);
    EXPECT_GE(price, lb - 1e-6);
}

// ── Sabr::vol_smile ───────────────────────────────────────────────────────────

TEST(SabrVolSmileTest, OutputSize) {
    Sabr s;
    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    auto [ks, vols] = s.vol_smile(make_params(), make_market(), strikes);
    EXPECT_EQ(ks.size(),   strikes.size());
    EXPECT_EQ(vols.size(), strikes.size());
}

TEST(SabrVolSmileTest, StrikesPassedThrough) {
    Sabr s;
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    auto [ks, vols] = s.vol_smile(make_params(), make_market(), strikes);
    for (size_t i = 0; i < strikes.size(); ++i)
        EXPECT_DOUBLE_EQ(ks[i], strikes[i]);
}

TEST(SabrVolSmileTest, VolsPositive) {
    Sabr s;
    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    auto [ks, vols] = s.vol_smile(make_params(), make_market(), strikes);
    for (size_t i = 0; i < vols.size(); ++i)
        EXPECT_GT(vols[i], 0.0) << "vols[" << i << "] <= 0";
}

TEST(SabrVolSmileTest, ConsistentWithImpliedVol) {
    // vol_smile doit retourner les mêmes valeurs que implied_vol appelé individuellement
    Sabr s;
    auto p = make_params();
    auto m = make_market();
    std::vector<double> strikes = {90.0, 100.0, 110.0};
    auto [ks, vols] = s.vol_smile(p, m, strikes);
    for (size_t i = 0; i < strikes.size(); ++i)
        EXPECT_DOUBLE_EQ(vols[i], s.implied_vol(p, m, strikes[i]));
}
