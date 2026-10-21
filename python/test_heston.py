"""Tests du module Python heston (Carr-Madan FFT)."""
import sys
import math
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "build"))
import heston


def make_market() -> heston.HestonMarket:
    p = heston.HestonParams(kappa=2.0, v0=0.04, theta=0.04, vol=0.3, cor=-0.7)
    return heston.HestonMarket(S0=100.0, r=0.05, T=1.0, q=0.0, params=p)


class TestHestonParams(unittest.TestCase):

    def test_feller_ok(self):
        p = heston.HestonParams(kappa=2.0, v0=0.04, theta=0.04, vol=0.3, cor=-0.7)
        # 2*2*0.04 = 0.16 > 0.09 = 0.3^2
        self.assertTrue(p.feller_ok())

    def test_feller_violated(self):
        p = heston.HestonParams(kappa=0.1, v0=0.04, theta=0.01, vol=0.9, cor=0.0)
        # 2*0.1*0.01 = 0.002 < 0.81 = 0.9^2
        self.assertFalse(p.feller_ok())

    def test_repr(self):
        p = heston.HestonParams(kappa=2.0, v0=0.04, theta=0.04, vol=0.3, cor=-0.7)
        self.assertIn("HestonParams", repr(p))


class TestHestonPrice(unittest.TestCase):

    def setUp(self):
        self.h   = heston.Heston()
        self.mkt = make_market()

    def test_atm_call_positive(self):
        price = self.h.call_price(self.mkt, K=100.0)
        self.assertGreater(price, 0.0)

    def test_otm_less_than_atm(self):
        atm = self.h.call_price(self.mkt, K=100.0)
        otm = self.h.call_price(self.mkt, K=130.0)
        self.assertLess(otm, atm)

    def test_lower_bound(self):
        K     = 100.0
        price = self.h.call_price(self.mkt, K=K)
        lb    = max(self.mkt.S0 - K * math.exp(-self.mkt.r * self.mkt.T), 0.0)
        self.assertGreaterEqual(price, lb - 1e-4)

    def test_intrinsic_deep_itm(self):
        # Call très ITM : prix ≈ S0 - K*e^{-rT}
        K     = 50.0
        price = self.h.call_price(self.mkt, K=K)
        lb    = self.mkt.S0 - K * math.exp(-self.mkt.r * self.mkt.T)
        self.assertGreater(price, lb - 1e-2)

    def test_call_grid_length(self):
        N = 512
        log_strikes, calls = self.h.call_grid(self.mkt, N=N)
        self.assertEqual(len(log_strikes), N)
        self.assertEqual(len(calls), N)

    def test_call_grid_non_negative(self):
        _, calls = self.h.call_grid(self.mkt, N=512)
        for i, c in enumerate(calls):
            self.assertGreaterEqual(c, 0.0, msg=f"calls[{i}] < 0")

    def test_call_grid_consistent_with_single(self):
        # call_price et call_grid doivent donner le même résultat pour K=100
        K            = 100.0
        log_K        = math.log(K)
        price_single = self.h.call_price(self.mkt, K=K, N=4096)
        log_strikes, calls = self.h.call_grid(self.mkt, N=4096)
        # interpolation linéaire au point log(K)=0
        i = next(j for j, s in enumerate(log_strikes) if s >= log_K)
        x1, x2 = log_strikes[i - 1], log_strikes[i]
        y1, y2 = calls[i - 1], calls[i]
        price_grid = y1 + (y2 - y1) * (log_K - x1) / (x2 - x1)
        self.assertAlmostEqual(price_single, price_grid, delta=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
