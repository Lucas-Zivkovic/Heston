"""
multi_calibration.py
====================
Calibration Heston + SABR sur plusieurs actifs au choix.

Actifs disponibles :
    SPY  — S&P 500 ETF (référence)
    AAPL — Apple (large cap, vol modérée)
    NVDA — NVIDIA (semi-conducteurs, vol élevée)
    GLD  — Or ETF (profil de vol différent)
    TSLA — Tesla (vol extrême)

Utilisation
-----------
    python multi_calibration.py              # menu interactif
    python multi_calibration.py SPY          # directement
    python multi_calibration.py SPY NVDA     # plusieurs à la fois

L'objet MultiCalibration peut aussi être importé :
    from multi_calibration import MultiCalibration
    mc = MultiCalibration("NVDA")
    mc.run()
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend sans affichage (headless / no DISPLAY needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from datetime import datetime
from scipy.optimize import minimize, differential_evolution

import yfinance as yf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "build"))
import heston as heston_mod
from heston import Sabr, SabrParams, SabrMarket

# ── Actifs disponibles ────────────────────────────────────────────────────────

AVAILABLE = {
    "SPY":  {"name": "S&P 500 ETF",          "r": 0.043, "beta": 0.5},
    "AAPL": {"name": "Apple Inc.",            "r": 0.043, "beta": 0.7},
    "NVDA": {"name": "NVIDIA Corporation",    "r": 0.043, "beta": 0.5},
    "GLD":  {"name": "Gold ETF (SPDR)",       "r": 0.043, "beta": 0.8},
    "TSLA": {"name": "Tesla Inc.",            "r": 0.043, "beta": 0.5},
}

# ── Utilitaires ───────────────────────────────────────────────────────────────

def bs_implied_vol(price: float, F: float, K: float, T: float, r: float) -> float:
    """Inversion Black-Scholes par Newton-Raphson. Retourne NaN si hors bornes."""
    D  = math.exp(-r * T)
    lb = max(D * (F - K), 0.0)
    ub = D * F
    if price <= lb + 1e-8 or price >= ub - 1e-8:
        return float("nan")
    sigma = 0.3
    for _ in range(80):
        sqT  = math.sqrt(T)
        d1   = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
        d2   = d1 - sigma * sqT
        nd1  = 0.5 * math.erfc(-d1 / math.sqrt(2))
        nd2  = 0.5 * math.erfc(-d2 / math.sqrt(2))
        bs   = D * (F * nd1 - K * nd2)
        vega = D * F * sqT * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        diff = bs - price
        if abs(diff) < 1e-10:
            break
        if vega < 1e-14:
            return float("nan")
        sigma -= diff / vega
        sigma  = max(sigma, 1e-6)
    return sigma


def fetch_options(ticker: str, r: float) -> pd.DataFrame:
    """
    Télécharge les options calls (Yahoo Finance) et retourne un DataFrame filtré
    avec colonnes : T, K, mid, iv.
    """
    tkr    = yf.Ticker(ticker)
    S0     = float(tkr.history(period="1d")["Close"].iloc[-1])
    today  = datetime.today()

    targets = [30, 60, 90, 180, 365]
    exps    = tkr.options
    chosen  = []
    for t in targets:
        best = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - t))
        if best not in chosen:
            chosen.append(best)

    rows = []
    for exp in chosen:
        chain = tkr.option_chain(exp)
        calls = chain.calls.copy()
        T_exp = (datetime.strptime(exp, "%Y-%m-%d") - today).days / 365.0
        if T_exp <= 0:
            continue

        # Filtres de liquidité
        calls = calls[
            (calls["impliedVolatility"] > 0.02) &
            (calls["volume"].fillna(0) > 5)
        ]
        calls = calls[
            (calls["strike"] >= 0.80 * S0) &
            (calls["strike"] <= 1.20 * S0)
        ]
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
        calls = calls[calls["mid"] > 0.30]

        for _, row in calls.iterrows():
            rows.append({
                "T":   T_exp,
                "K":   float(row["strike"]),
                "mid": float(row["mid"]),
                "iv":  float(row["impliedVolatility"]),
            })

    df = pd.DataFrame(rows).dropna()
    return S0, df


# ── Classe principale ─────────────────────────────────────────────────────────

class MultiCalibration:
    """
    Calibration Heston + SABR pour un actif donné.

    Paramètres
    ----------
    ticker : str
        Symbole Yahoo Finance (parmi AVAILABLE).
    verbose : bool
        Affiche les détails de convergence si True.
    """

    def __init__(self, ticker: str, verbose: bool = True):
        ticker = ticker.upper()
        if ticker not in AVAILABLE:
            raise ValueError(
                f"Actif '{ticker}' non disponible. Choisissez parmi : {list(AVAILABLE)}"
            )
        self.ticker  = ticker
        self.meta    = AVAILABLE[ticker]
        self.verbose = verbose
        self.r       = self.meta["r"]
        self.beta    = self.meta["beta"]

        # Résultats (remplis par run())
        self.S0          = None
        self.df          = None
        self.heston_params = None   # dict
        self.sabr_params   = None   # dict
        self.heston_rmse   = None
        self.sabr_rmse     = None

        self._h_pricer  = heston_mod.Heston()
        self._s_pricer  = Sabr()

    # ── Données ───────────────────────────────────────────────────────────────

    def load_data(self):
        print(f"\n{'='*60}")
        print(f"  {self.ticker}  —  {self.meta['name']}")
        print(f"{'='*60}")
        print("  Téléchargement des données de marché…")
        self.S0, self.df = fetch_options(self.ticker, self.r)
        print(f"  S0 = {self.S0:.2f}   r = {self.r:.3f}   "
              f"{len(self.df)} options sur {self.df['T'].nunique()} maturités")

    # ── Calibration Heston ────────────────────────────────────────────────────

    def _objective_heston(self, params):
        kappa, v0, theta, vol, cor = params
        if v0 <= 0 or theta <= 0 or vol <= 0 or not (-1 < cor < 1):
            return 1e6
        try:
            p = heston_mod.HestonParams(kappa=kappa, v0=v0, theta=theta, vol=vol, cor=cor)
        except Exception:
            return 1e6
        err = []
        for _, row in self.df.iterrows():
            mkt = heston_mod.HestonMarket(S0=self.S0, r=self.r, T=row["T"], q=0.0, params=p)
            try:
                mp = self._h_pricer.call_price(mkt, row["K"])
            except Exception:
                return 1e6
            denom = max(row["mid"], 0.5)
            err.append(((row["mid"] - mp) / denom) ** 2)
        return float(np.sqrt(np.mean(err))) if err else 1e6

    def calibrate_heston(self):
        print("\n  Calibration Heston  (differential_evolution → L-BFGS-B)…")
        bounds = [(0.05, 15), (1e-4, 1.5), (1e-4, 1.5), (0.05, 5), (-0.995, 0.995)]

        # Passe globale (robuste aux minima locaux)
        res_de = differential_evolution(
            self._objective_heston, bounds,
            seed=42, maxiter=80, tol=1e-5,
            popsize=8, mutation=(0.5, 1.5), recombination=0.9,
            disp=False,
        )
        # Raffinement local
        res = minimize(
            self._objective_heston, res_de.x, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500},
        )
        k, v, th, xi, rho = res.x
        self.heston_params = dict(kappa=k, v0=v, theta=th, vol=xi, cor=rho)
        self.heston_rmse   = res.fun
        print(f"  κ={k:.3f}  v0={v:.4f}  θ={th:.4f}  ξ={xi:.3f}  ρ={rho:.3f}"
              f"  RMSE={self.heston_rmse:.4f}")

    # ── Calibration SABR ──────────────────────────────────────────────────────

    def _objective_sabr(self, params):
        alpha, nu, rho = params
        if alpha <= 0 or nu <= 0 or not (-1 < rho < 1):
            return 1e6
        try:
            p = SabrParams(alpha=alpha, beta=self.beta, rho=rho, nu=nu)
        except Exception:
            return 1e6
        err = []
        for _, row in self.df.iterrows():
            F   = self.S0 * math.exp(self.r * row["T"])
            mkt = SabrMarket(F=F, T=row["T"], r=self.r)
            try:
                mp = self._s_pricer.call_price(p, mkt, row["K"])
            except Exception:
                return 1e6
            denom = max(abs(row["mid"]), 0.5)
            err.append(((row["mid"] - mp) / denom) ** 2)
        return float(np.sqrt(np.mean(err))) if err else 1e6

    def calibrate_sabr(self):
        print("\n  Calibration SABR  (differential_evolution → L-BFGS-B)…")
        bounds = [(1e-4, 5), (1e-4, 5), (-0.995, 0.995)]

        res_de = differential_evolution(
            self._objective_sabr, bounds,
            seed=42, maxiter=80, tol=1e-5,
            popsize=8, disp=False,
        )
        res = minimize(
            self._objective_sabr, res_de.x, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500},
        )
        alpha, nu, rho = res.x
        self.sabr_params = dict(alpha=alpha, beta=self.beta, nu=nu, rho=rho)
        self.sabr_rmse   = res.fun
        print(f"  α={alpha:.3f}  β={self.beta}  ν={nu:.3f}  ρ={rho:.3f}"
              f"  RMSE={self.sabr_rmse:.4f}")

    # ── Surface et graphique ──────────────────────────────────────────────────

    def plot(self, save: bool = True) -> str:
        """Génère la surface 3D (marché + Heston + SABR) + tableau des paramètres."""
        df    = self.df
        S0    = self.S0
        r     = self.r
        beta  = self.beta

        T_uniq = sorted(df["T"].unique())
        K_min  = df["K"].min() * 0.98
        K_max  = df["K"].max() * 1.02
        K_grid = np.linspace(K_min, K_max, 60)

        p_heston = heston_mod.HestonParams(**self.heston_params)
        p_sabr   = SabrParams(
            alpha=self.sabr_params["alpha"],
            beta=beta,
            rho=self.sabr_params["rho"],
            nu=self.sabr_params["nu"],
        )

        TT, KK  = np.meshgrid(T_uniq, K_grid, indexing="ij")
        vol_h   = np.full_like(TT, np.nan)
        vol_s   = np.full_like(TT, np.nan)
        vol_mkt = np.full_like(TT, np.nan)

        h_pricer = self._h_pricer
        s_pricer = self._s_pricer

        for i, T in enumerate(T_uniq):
            mkt_h = heston_mod.HestonMarket(S0=S0, r=r, T=T, q=0.0, params=p_heston)
            F     = S0 * math.exp(r * T)
            mkt_s = SabrMarket(F=F, T=T, r=r)
            for j, K in enumerate(K_grid):
                try:
                    price     = h_pricer.call_price(mkt_h, K)
                    vol_h[i, j] = bs_implied_vol(price, F, K, T, r)
                except Exception:
                    pass
                try:
                    vol_s[i, j] = s_pricer.implied_vol(p_sabr, mkt_s, K)
                except Exception:
                    pass
            sub = df[abs(df["T"] - T) < 1e-6]
            for j, K in enumerate(K_grid):
                near = sub.iloc[(sub["K"] - K).abs().argsort()[:1]]
                if not near.empty and abs(float(near["K"].iloc[0]) - K) < 8.0:
                    vol_mkt[i, j] = float(near["iv"].iloc[0])

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 13))
        fig.patch.set_facecolor("#0f0f1a")
        fig.suptitle(
            f"Surface de volatilité implicite — {self.ticker}  "
            f"({self.meta['name']})   S₀={S0:.1f}",
            fontsize=15, fontweight="bold", color="white", y=0.98,
        )

        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            height_ratios=[3, 1.2],
            hspace=0.35, wspace=0.3,
            left=0.04, right=0.97, top=0.93, bottom=0.04,
        )

        datasets = [
            (vol_mkt, "Marché Yahoo Finance",
             f"RMSE Heston={self.heston_rmse:.4f}  SABR={self.sabr_rmse:.4f}",
             cm.coolwarm),
            (vol_h,
             f"Heston calibré",
             f"κ={self.heston_params['kappa']:.2f}  v₀={self.heston_params['v0']:.3f}  "
             f"θ={self.heston_params['theta']:.3f}  ξ={self.heston_params['vol']:.3f}  "
             f"ρ={self.heston_params['cor']:.3f}",
             cm.plasma),
            (vol_s,
             f"SABR calibré",
             f"α={self.sabr_params['alpha']:.3f}  β={beta}  "
             f"ν={self.sabr_params['nu']:.3f}  ρ={self.sabr_params['rho']:.3f}",
             cm.viridis),
        ]

        for idx, (vol, title, subtitle, cmap) in enumerate(datasets):
            ax = fig.add_subplot(gs[0, idx], projection="3d")
            ax.set_facecolor("#0f0f1a")

            surf = ax.plot_surface(
                KK, TT, np.where(np.isfinite(vol), vol * 100, np.nan),
                cmap=cmap, linewidth=0, antialiased=True, alpha=0.9,
            )
            ax.set_xlabel("Strike K", color="lightgray", labelpad=6, fontsize=8)
            ax.set_ylabel("Maturité (ans)", color="lightgray", labelpad=6, fontsize=8)
            ax.set_zlabel("Vol impl. (%)", color="lightgray", labelpad=6, fontsize=8)
            ax.tick_params(colors="lightgray", labelsize=7)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_title(title, color="white", fontsize=10, pad=4)
            ax.set_xlabel("Strike K", color="lightgray", labelpad=4, fontsize=8)
            cb = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08, label="Vol (%)")
            cb.ax.yaxis.label.set_color("lightgray")
            cb.ax.tick_params(colors="lightgray")

            ax.text2D(0.5, -0.06, subtitle, transform=ax.transAxes,
                      ha="center", fontsize=7, color="#aaaaaa")

        # ── Tableau des paramètres ─────────────────────────────────────────────
        ax_tab = fig.add_subplot(gs[1, :])
        ax_tab.axis("off")
        ax_tab.set_facecolor("#0f0f1a")

        rows_data = [
            ["Modèle", "Paramètre", "Valeur", "Interprétation"],
            ["Heston", "κ (kappa)",   f"{self.heston_params['kappa']:.4f}",
             "Vitesse de retour à la moyenne de la variance"],
            ["",       "v₀",          f"{self.heston_params['v0']:.4f}",
             f"Vol initiale implicite ≈ {math.sqrt(self.heston_params['v0'])*100:.1f}%"],
            ["",       "θ (theta)",   f"{self.heston_params['theta']:.4f}",
             f"Vol long terme ≈ {math.sqrt(self.heston_params['theta'])*100:.1f}%"],
            ["",       "ξ (vol of vol)", f"{self.heston_params['vol']:.4f}",
             "Volatilité de la variance (courbure du smile)"],
            ["",       "ρ (cor)",     f"{self.heston_params['cor']:.4f}",
             "Corrélation spot-vol (skew)"],
            ["",       "RMSE",        f"{self.heston_rmse:.5f}", "Erreur relative moyenne"],
            ["SABR",   "α (alpha)",   f"{self.sabr_params['alpha']:.4f}",
             "Niveau de vol ATM initial"],
            ["",       "β (beta)",    f"{beta}",
             "Exposant de la lognormalité (fixé)"],
            ["",       "ν (nu)",      f"{self.sabr_params['nu']:.4f}",
             "Vol of vol (courbure du smile)"],
            ["",       "ρ (rho)",     f"{self.sabr_params['rho']:.4f}",
             "Corrélation F-vol (skew)"],
            ["",       "RMSE",        f"{self.sabr_rmse:.5f}", "Erreur relative moyenne"],
        ]

        col_widths = [0.08, 0.12, 0.10, 0.55]
        x_starts   = [0.01, 0.10, 0.22, 0.33]
        y_start    = 0.95
        row_h      = 0.075

        for r_i, row in enumerate(rows_data):
            bg = "#1a1a2e" if r_i % 2 == 0 else "#16213e"
            if r_i == 0:
                bg = "#2a2a4a"
            for c_i, (cell, x) in enumerate(zip(row, x_starts)):
                weight = "bold" if r_i == 0 or c_i == 0 else "normal"
                color  = "#ffffff" if r_i == 0 else ("#ffcc44" if c_i == 0 else "#dddddd")
                ax_tab.text(
                    x, y_start - r_i * row_h, cell,
                    transform=ax_tab.transAxes,
                    fontsize=8, color=color, fontweight=weight,
                    verticalalignment="top",
                )

        out = f"calibration_{self.ticker}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  Graphique sauvegardé : {out}")
        return out

    # ── Méthode principale ────────────────────────────────────────────────────

    def run(self) -> dict:
        """Lance le pipeline complet : données → calibration → graphique."""
        self.load_data()
        self.calibrate_heston()
        self.calibrate_sabr()
        out = self.plot()
        return {
            "ticker":        self.ticker,
            "S0":            self.S0,
            "heston_params": self.heston_params,
            "heston_rmse":   self.heston_rmse,
            "sabr_params":   self.sabr_params,
            "sabr_rmse":     self.sabr_rmse,
            "output_file":   out,
        }


# ── Interface CLI ─────────────────────────────────────────────────────────────

def interactive_menu() -> list[str]:
    print("\n" + "="*60)
    print("  CALIBRATION MULTI-ACTIFS — Heston + SABR")
    print("="*60)
    print("\n  Actifs disponibles :")
    for sym, meta in AVAILABLE.items():
        print(f"    [{sym:4s}]  {meta['name']}")
    print("\n  Entrez un ou plusieurs tickers séparés par des espaces,")
    print("  ou ENTRER pour calibrer tous les actifs :")
    raw = input("  > ").strip().upper()
    if not raw:
        return list(AVAILABLE)
    return [t.strip() for t in raw.split() if t.strip() in AVAILABLE]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:] if t.upper() in AVAILABLE]
        invalid = [t for t in sys.argv[1:] if t.upper() not in AVAILABLE]
        if invalid:
            print(f"Actifs inconnus ignorés : {invalid}")
    else:
        tickers = interactive_menu()

    if not tickers:
        print("Aucun actif valide sélectionné. Fin.")
        sys.exit(1)

    results = []
    for ticker in tickers:
        mc = MultiCalibration(ticker)
        results.append(mc.run())

    # Résumé comparatif si plusieurs actifs
    if len(results) > 1:
        print("\n" + "="*70)
        print("  RÉSUMÉ COMPARATIF")
        print("="*70)
        print(f"  {'Ticker':6s}  {'S0':>7s}  {'Heston RMSE':>11s}  {'SABR RMSE':>9s}  "
              f"{'ρ (Heston)':>10s}  {'Vol LT':>7s}")
        print("  " + "-"*66)
        for r in results:
            vlt = math.sqrt(r["heston_params"]["theta"]) * 100
            print(f"  {r['ticker']:6s}  {r['S0']:>7.1f}  "
                  f"{r['heston_rmse']:>11.5f}  {r['sabr_rmse']:>9.5f}  "
                  f"{r['heston_params']['cor']:>10.3f}  {vlt:>6.1f}%")
