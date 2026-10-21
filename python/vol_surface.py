"""Visualisation de la surface de volatilité implicite — Heston vs SABR."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "build"))
import heston as heston_mod
from heston import Sabr, SabrParams, SabrMarket

# ── Paramètres communs ─────────────────────────────────────────────────────────

S0      = 100.0
r       = 0.05
q       = 0.0
strikes = np.linspace(70, 135, 40)
mats    = np.linspace(0.1, 2.0, 20)

# ── Paramètres Heston ──────────────────────────────────────────────────────────

heston_p = heston_mod.HestonParams(kappa=2.0, v0=0.04, theta=0.04, vol=0.3, cor=-0.7)
h        = heston_mod.Heston()

# ── Paramètres SABR ───────────────────────────────────────────────────────────

sabr_p = SabrParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
sabr   = Sabr()

# ── Calcul des surfaces ────────────────────────────────────────────────────────

KK, TT = np.meshgrid(strikes, mats)
vol_heston = np.zeros_like(KK)
vol_sabr   = np.zeros_like(KK)

for i, T in enumerate(mats):
    for j, K in enumerate(strikes):
        # Heston : vol implicite via Black-Scholes inverse approché
        mkt   = heston_mod.HestonMarket(S0=S0, r=r, T=T, q=q, params=heston_p)
        price = h.call_price(mkt, K=K)

        # Inversion Black-Scholes (Newton-Raphson)
        F  = S0 * np.exp((r - q) * T)
        D  = np.exp(-r * T)
        lb = max(D * (F - K), 0.0)
        ub = D * F

        if price <= lb + 1e-8 or price >= ub - 1e-8:
            vol_heston[i, j] = np.nan
            continue

        # Newton sur sigma
        sigma = 0.3
        for _ in range(50):
            sqT   = np.sqrt(T)
            d1    = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*sqT)
            d2    = d1 - sigma*sqT
            from math import erfc
            nd1   = 0.5*erfc(-d1/np.sqrt(2))
            nd2   = 0.5*erfc(-d2/np.sqrt(2))
            bs    = D * (F*nd1 - K*nd2)
            vega  = D * F * sqT * np.exp(-0.5*d1**2) / np.sqrt(2*np.pi)
            diff  = bs - price
            if abs(diff) < 1e-10:
                break
            sigma -= diff / vega
            sigma  = max(sigma, 1e-6)
        vol_heston[i, j] = sigma

        # SABR
        vol_sabr[i, j] = sabr.implied_vol(sabr_p, SabrMarket(F=F, T=T, r=r), K)

# ── Tracé ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 6))
fig.suptitle("Surface de volatilité implicite", fontsize=14, fontweight="bold")

for idx, (vol, title, cmap) in enumerate([
    (vol_heston, "Heston  (κ=2, θ=0.04, ξ=0.3, ρ=-0.7)", cm.plasma),
    (vol_sabr,   "SABR  (α=0.3, β=0.5, ρ=-0.3, ν=0.4)",  cm.viridis),
]):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    surf = ax.plot_surface(KK, TT, vol * 100, cmap=cmap,
                           linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturité T (ans)")
    ax.set_zlabel("Vol implicite (%)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label="Vol (%)")

plt.tight_layout()
plt.savefig("vol_surface.png", dpi=150, bbox_inches="tight")
print("Graphique sauvegardé : vol_surface.png")
plt.show()