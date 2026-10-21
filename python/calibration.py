import sys
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "build"))
import heston


class calibration:
    def setUp(self):
        self.h   = heston.Heston()
        self.s = heston.Sabr()
        self.beta=0.5

    def objectiveHeston(self,params,strikes,T,marketPrices,r=0.025,S0=100,q=0):
        kappa,v0,theta,vol,cor=params
        p=heston.HestonParams(kappa,v0,theta,vol,cor)
        mkt=heston.HestonMarket(S0=S0,T=1, r=r, q=q, params=p)
        modelPrices=[]
        for i in range(len(strikes)):
            t=T[i]
            K=strikes[i]
            mkt.T=t
            modelPrices.append(self.h.call_price(mkt,K))
        modelPrices=np.array(modelPrices)
        marketPrices=np.array(marketPrices)
        return np.sqrt(np.mean(((marketPrices-modelPrices)/marketPrices)**2))

def calibrationHeston(self,marketPrices,strikes,maturities,r=0.025,S0=100,q=0):
    self.setUp()
    initialParams=[2.0,0.2,0.04,0.3,-0.5]
    bounds=[(0.01,10),(0.01,10),(0.01,10),(0.01,10),(-0.999,0.999)]
    res=minimize(self.objectiveHeston,initialParams,args=(strikes,maturities,marketPrices,r,S0,q),bounds=bounds,method="L-BFGS-B",options={'disp':True})
    if res.success:
        return res.x
    
def CalibrationSabr(self, strikes, marketPrices, maturities, r=0.025, F=100):
    self.setUp()
    initialParams = [0.3, 0.3, -0.2]
    bounds = [(1e-4, 5), (1e-4, 5), (-0.999, 0.999)]
    res = minimize(lambda p, *a: objectiveSabr(self, p, *a), initialParams,
                   args=(strikes, marketPrices, maturities, r, F),
                   bounds=bounds, method="L-BFGS-B",
                   options={'ftol': 1e-12, 'gtol': 1e-8})
    return res.x

def objectiveSabr(self, params, strikes, marketPrices, maturities, r, F):
    alpha, nu, rho = params
    p = heston.SabrParams(alpha, self.beta, rho, nu)
    modelPrices = []
    for i in range(len(strikes)):
        t = maturities[i]
        K = strikes[i]
        mkt = heston.SabrMarket(F=F, T=t, r=r)
        modelPrices.append(self.s.call_price(p, mkt, K))
    modelPrices  = np.array(modelPrices)
    marketPrices = np.array(marketPrices)
    denom = np.maximum(np.abs(marketPrices), 1e-3)
    return float(np.sqrt(np.mean(((marketPrices - modelPrices) / denom) ** 2)))