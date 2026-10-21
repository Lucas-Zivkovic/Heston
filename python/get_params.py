import sys, json, math
import numpy as np
from scipy.optimize import minimize, differential_evolution
import yfinance as yf
from datetime import datetime
import pandas as pd

sys.path.insert(0, "../build")
import heston as heston_mod
from heston import Sabr, SabrParams, SabrMarket

ASSETS = {
    "SPY":  {"beta": 0.5, "r": 0.043},
    "AAPL": {"beta": 0.7, "r": 0.043},
    "NVDA": {"beta": 0.5, "r": 0.043},
    "GLD":  {"beta": 0.8, "r": 0.043},
    "TSLA": {"beta": 0.5, "r": 0.043},
}

def fetch(ticker, r):
    tkr = yf.Ticker(ticker)
    S0 = float(tkr.history(period="1d")["Close"].iloc[-1])
    today = datetime.today()
    targets = [30, 60, 90, 180, 365]
    exps = tkr.options
    chosen = []
    for t in targets:
        best = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - t))
        if best not in chosen: chosen.append(best)
    rows = []
    for exp in chosen:
        chain = tkr.option_chain(exp)
        calls = chain.calls.copy()
        T_exp = (datetime.strptime(exp, "%Y-%m-%d") - today).days / 365.0
        if T_exp <= 0: continue
        calls = calls[(calls["impliedVolatility"] > 0.02) & (calls["volume"].fillna(0) > 5)]
        calls = calls[(calls["strike"] >= 0.80 * S0) & (calls["strike"] <= 1.20 * S0)]
        calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
        calls = calls[calls["mid"] > 0.30]
        for _, row in calls.iterrows():
            rows.append({"T": T_exp, "K": float(row["strike"]), "mid": float(row["mid"])})
    return S0, pd.DataFrame(rows).dropna()

results = {}
h_pricer = heston_mod.Heston()
s_pricer = Sabr()

for ticker, meta in ASSETS.items():
    r, beta = meta["r"], meta["beta"]
    print(f"\n=== {ticker} ===", flush=True)
    S0, df = fetch(ticker, r)
    print(f"  S0={S0:.2f}, {len(df)} options", flush=True)

    def obj_h(p):
        kappa, v0, theta, vol, cor = p
        if v0<=0 or theta<=0 or vol<=0 or not(-1<cor<1): return 1e6
        try: hp = heston_mod.HestonParams(kappa=kappa, v0=v0, theta=theta, vol=vol, cor=cor)
        except: return 1e6
        err = []
        for _, row in df.iterrows():
            mkt = heston_mod.HestonMarket(S0=S0, r=r, T=row["T"], q=0.0, params=hp)
            try: mp = h_pricer.call_price(mkt, row["K"])
            except: return 1e6
            err.append(((row["mid"]-mp)/max(row["mid"],0.5))**2)
        return float(np.sqrt(np.mean(err))) if err else 1e6

    def obj_s(p):
        alpha, nu, rho = p
        if alpha<=0 or nu<=0 or not(-1<rho<1): return 1e6
        try: sp = SabrParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
        except: return 1e6
        err = []
        for _, row in df.iterrows():
            F = S0 * math.exp(r * row["T"])
            mkt = SabrMarket(F=F, T=row["T"], r=r)
            try: mp = s_pricer.call_price(sp, mkt, row["K"])
            except: return 1e6
            err.append(((row["mid"]-mp)/max(abs(row["mid"]),0.5))**2)
        return float(np.sqrt(np.mean(err))) if err else 1e6

    bh = [(0.05,15),(1e-4,1.5),(1e-4,1.5),(0.05,5),(-0.995,0.995)]
    rh = differential_evolution(obj_h, bh, seed=42, maxiter=60, popsize=6, tol=1e-4, disp=False)
    rh = minimize(obj_h, rh.x, method="L-BFGS-B", bounds=bh, options={"ftol":1e-10,"maxiter":200})
    k,v,th,xi,rho = rh.x
    print(f"  Heston: k={k:.3f} v0={v:.4f} theta={th:.4f} xi={xi:.3f} rho={rho:.3f} RMSE={rh.fun:.5f}", flush=True)

    bs = [(1e-4,5),(1e-4,5),(-0.995,0.995)]
    rs = differential_evolution(obj_s, bs, seed=42, maxiter=60, popsize=6, tol=1e-4, disp=False)
    rs = minimize(obj_s, rs.x, method="L-BFGS-B", bounds=bs, options={"ftol":1e-10,"maxiter":200})
    al,nu,ro = rs.x
    print(f"  SABR: alpha={al:.3f} beta={beta} nu={nu:.3f} rho={ro:.3f} RMSE={rs.fun:.5f}", flush=True)

    results[ticker] = {
        "S0": round(S0,2),
        "heston": {"kappa":round(k,4),"v0":round(v,4),"theta":round(th,4),"vol":round(xi,4),"cor":round(rho,4),"rmse":round(rh.fun,6)},
        "sabr":   {"alpha":round(al,4),"beta":beta,"nu":round(nu,4),"rho":round(ro,4),"rmse":round(rs.fun,6)},
    }

with open("calibration_results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nOK : calibration_results.json")
