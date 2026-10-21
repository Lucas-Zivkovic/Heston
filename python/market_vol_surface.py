"""
Surface de volatilité implicite réelle — SPY (Yahoo Finance)
Calibration Heston + SABR, puis surface des deux modèles vs marché.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from scipy.optimize import minimize

import yfinance as yf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "build"))
import heston as heston_mod
from heston import Sabr, SabrParams, SabrMarket

# ── 1. Données réelles ────────────────────────────────────────────────────────

TICKER  = "SPY"
R       = 0.043          # taux sans risque ~US 3-mois (avril 2026)
BETA    = 0.5            # SABR beta fixé
MIN_VOL = 0.02           # filtre : iv > 2%
MIN_MID = 0.50           # filtre : prix mid > $0.50
MIN_VOL_CONTRACTS = 10   # volume minimum

print("Téléchargement des données de marché…")
tkr   = yf.Ticker(TICKER)
S0    = float(tkr.history(period="1d")["Close"].iloc[-1])
today = datetime.today()
print(f"  {TICKER}  S0 = {S0:.2f}   r = {R:.3f}")

# Maturités cibles en jours
targets = [30, 60, 90, 180, 365]
expirations = tkr.options
chosen = []
for t in targets:
    best = min(expirations,
               key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - t))
    if best not in chosen:
        chosen.append(best)

rows = []
for exp in chosen:
    chain = tkr.option_chain(exp)
    calls = chain.calls.copy()
    T_exp = (datetime.strptime(exp, "%Y-%m-%d") - today).days / 365.0
    calls = calls[
        (calls["impliedVolatility"] > MIN_VOL) &
        (calls["volume"] > MIN_VOL_CONTRACTS)
    ]
    calls = calls[
        (calls["strike"] >= 0.82 * S0) &
        (calls["strike"] <= 1.18 * S0)
    ]
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
    calls = calls[calls["mid"] > MIN_MID]
    for _, row in calls.iterrows():
        rows.append({"T": T_exp, "K": float(row["strike"]),
                     "iv": float(row["impliedVolatility"]),
                     "mid": float(row["mid"])})

import pandas as pd
df = pd.DataFrame(rows).dropna()
print(f"  {len(df)} options retenues sur {len(chosen)} maturités\n")

strikes_mkt    = df["K"].tolist()
maturities_mkt = df["T"].tolist()
prices_mkt     = df["mid"].tolist()

# ── 2. Calibration Heston ─────────────────────────────────────────────────────

h_pricer = heston_mod.Heston()

def objective_heston(params):
    kappa, v0, theta, vol, cor = params
    p   = heston_mod.HestonParams(kappa=kappa, v0=v0, theta=theta, vol=vol, cor=cor)
    err = []
    for K, T, mp in zip(strikes_mkt, maturities_mkt, prices_mkt):
        mkt = heston_mod.HestonMarket(S0=S0, r=R, T=T, q=0.0, params=p)
        try:
            model_p = h_pricer.call_price(mkt, K)
        except Exception:
            return 1e6
        err.append(((mp - model_p) / max(mp, 0.5)) ** 2)
    return float(np.sqrt(np.mean(err)))

print("Calibration Heston…")
x0_h = [2.0, 0.04, 0.04, 0.5, -0.5]
bnd_h = [(0.05, 15), (1e-4, 2.0), (1e-4, 2.0), (0.05, 5), (-0.999, 0.999)]
res_h = minimize(objective_heston, x0_h, method="L-BFGS-B", bounds=bnd_h,
                 options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
kappa_c, v0_c, theta_c, vol_c, cor_c = res_h.x
print(f"  κ={kappa_c:.3f}  v0={v0_c:.4f}  θ={theta_c:.4f}  "
      f"ξ={vol_c:.3f}  ρ={cor_c:.3f}  RMSE={res_h.fun:.4f}\n")

# ── 3. Calibration SABR ───────────────────────────────────────────────────────

sabr_pricer = Sabr()

def objective_sabr(params):
    alpha, nu, rho = params
    p   = SabrParams(alpha=alpha, beta=BETA, rho=rho, nu=nu)
    err = []
    for K, T, mp in zip(strikes_mkt, maturities_mkt, prices_mkt):
        F   = S0 * np.exp(R * T)
        mkt = SabrMarket(F=F, T=T, r=R)
        try:
            model_p = sabr_pricer.call_price(p, mkt, K)
        except Exception:
            return 1e6
        denom = max(abs(mp), 0.5)
        err.append(((mp - model_p) / denom) ** 2)
    return float(np.sqrt(np.mean(err)))

print("Calibration SABR…")
x0_s  = [0.3, 0.4, -0.3]
bnd_s = [(1e-4, 5), (1e-4, 5), (-0.999, 0.999)]
res_s = minimize(objective_sabr, x0_s, method="L-BFGS-B", bounds=bnd_s,
                 options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300})
alpha_c, nu_c, rho_c = res_s.x
print(f"  α={alpha_c:.3f}  β={BETA}  ν={nu_c:.3f}  ρ={rho_c:.3f}  "
      f"RMSE={res_s.fun:.4f}\n")

# ── 4. Grille pour les surfaces ───────────────────────────────────────────────

T_uniq = sorted(df["T"].unique())
K_grid = np.linspace(df["K"].min() * 0.98, df["K"].max() * 1.02, 50)

p_heston = heston_mod.HestonParams(kappa=kappa_c, v0=v0_c, theta=theta_c,
                                   vol=vol_c, cor=cor_c)
p_sabr   = SabrParams(alpha=alpha_c, beta=BETA, rho=rho_c, nu=nu_c)

def bs_implied_vol(price, F, K, T, r):
    """Newton-Raphson pour inverser Black-Scholes."""
    D  = np.exp(-r * T)
    lb = max(D * (F - K), 0.0)
    ub = D * F
    if price <= lb + 1e-8 or price >= ub - 1e-8:
        return np.nan
    from math import erfc, sqrt, log, exp, pi
    sigma = 0.3
    for _ in range(60):
        sqT  = sqrt(T)
        d1   = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
        d2   = d1 - sigma * sqT
        nd1  = 0.5 * erfc(-d1 / sqrt(2))
        nd2  = 0.5 * erfc(-d2 / sqrt(2))
        bs   = D * (F * nd1 - K * nd2)
        vega = D * F * sqT * exp(-0.5 * d1**2) / sqrt(2 * pi)
        diff = bs - price
        if abs(diff) < 1e-10:
            break
        if vega < 1e-12:
            return np.nan
        sigma -= diff / vega
        sigma  = max(sigma, 1e-6)
    return sigma

# Surfaces [len(T_uniq) x len(K_grid)]
TT, KK = np.meshgrid(T_uniq, K_grid, indexing="ij")
vol_heston = np.full_like(TT, np.nan)
vol_sabr   = np.full_like(TT, np.nan)
vol_mkt    = np.full_like(TT, np.nan)

for i, T in enumerate(T_uniq):
    mkt_heston = heston_mod.HestonMarket(S0=S0, r=R, T=T, q=0.0, params=p_heston)
    F          = S0 * np.exp(R * T)
    mkt_sabr   = SabrMarket(F=F, T=T, r=R)

    for j, K in enumerate(K_grid):
        # Heston
        try:
            price = h_pricer.call_price(mkt_heston, K)
            vol_heston[i, j] = bs_implied_vol(price, F, K, T, R)
        except Exception:
            pass

        # SABR
        try:
            vol_sabr[i, j] = sabr_pricer.implied_vol(p_sabr, mkt_sabr, K)
        except Exception:
            pass

    # Marché : interpolation des iv observées à cette maturité
    sub = df[np.abs(df["T"] - T) < 1e-6]
    if not sub.empty:
        for j, K in enumerate(K_grid):
            nearest = sub.iloc[(sub["K"] - K).abs().argsort()[:1]]
            if abs(float(nearest["K"].iloc[0]) - K) < 5.0:
                vol_mkt[i, j] = float(nearest["iv"].iloc[0])

# ── 5. Tracé ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 6))
fig.suptitle(f"Surface de volatilité implicite — {TICKER}  (S₀={S0:.0f})",
             fontsize=14, fontweight="bold")

datasets = [
    (vol_mkt,    f"Marché Yahoo Finance",                          cm.coolwarm),
    (vol_heston, f"Heston calibré  κ={kappa_c:.2f} ρ={cor_c:.2f}", cm.plasma),
    (vol_sabr,   f"SABR calibré  α={alpha_c:.2f} ρ={rho_c:.2f}",  cm.viridis),
]

for idx, (vol, title, cmap) in enumerate(datasets):
    ax   = fig.add_subplot(1, 3, idx + 1, projection="3d")
    surf = ax.plot_surface(KK, TT, vol * 100, cmap=cmap,
                           linewidth=0, antialiased=True, alpha=0.88)
    ax.set_xlabel("Strike K", labelpad=6)
    ax.set_ylabel("Maturité T (ans)", labelpad=6)
    ax.set_zlabel("Vol implicite (%)", labelpad=6)
    ax.set_title(title, fontsize=9)
    fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.1, label="Vol (%)")

plt.tight_layout()
out = "market_vol_surface.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Graphique sauvegardé : {out}")
plt.show()