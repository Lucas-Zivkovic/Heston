"""Tests unitaires de calibration.py."""
import sys
import unittest
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "build"))
import heston
from heston import Sabr, SabrParams, SabrMarket
from calibration import calibration, calibrationHeston, CalibrationSabr, objectiveSabr


def make_synthetic_data(params=(2.0, 0.04, 0.04, 0.3, -0.7),
                        strikes=(90.0, 100.0, 110.0),
                        maturities=(1.0, 1.0, 1.0),
                        r=0.025, S0=100.0, q=0):
    """Génère des prix de marché synthétiques avec des paramètres connus."""
    kappa, v0, theta, vol, cor = params
    p = heston.HestonParams(kappa=kappa, v0=v0, theta=theta, vol=vol, cor=cor)
    h = heston.Heston()
    prices = [
        h.call_price(heston.HestonMarket(S0=S0, r=r, T=T, q=q, params=p), K=K)
        for K, T in zip(strikes, maturities)
    ]
    return prices, list(strikes), list(maturities)


class TestObjectiveHeston(unittest.TestCase):

    def setUp(self):
        self.c = calibration()
        self.c.setUp()
        self.true_params = (2.0, 0.04, 0.04, 0.3, -0.7)
        self.prices, self.strikes, self.maturities = make_synthetic_data(self.true_params)

    def test_returns_float(self):
        """objectiveHeston retourne un float."""
        result = self.c.objectiveHeston(
            self.true_params, self.strikes, self.maturities, self.prices
        )
        self.assertIsInstance(result, float)

    def test_perfect_params_gives_zero_rmse(self):
        """Avec les vrais paramètres, la RMSE relative doit être quasi nulle."""
        result = self.c.objectiveHeston(
            self.true_params, self.strikes, self.maturities, self.prices
        )
        self.assertAlmostEqual(result, 0.0, delta=1e-6)

    def test_wrong_params_gives_positive_rmse(self):
        """Des paramètres différents donnent une RMSE > 0."""
        wrong_params = (1.0, 0.09, 0.09, 0.5, 0.0)
        result = self.c.objectiveHeston(
            wrong_params, self.strikes, self.maturities, self.prices
        )
        self.assertGreater(result, 0.0)

    def test_rmse_non_negative(self):
        """La RMSE est toujours >= 0."""
        for params in [
            (0.5, 0.01, 0.01, 0.1, -0.9),
            (5.0, 0.1, 0.1, 0.8, 0.5),
        ]:
            result = self.c.objectiveHeston(
                params, self.strikes, self.maturities, self.prices
            )
            self.assertGreaterEqual(result, 0.0)

    def test_output_length_independent_of_strike_count(self):
        """La RMSE est un scalaire quelle que soit la taille de la grille."""
        prices_5, strikes_5, mats_5 = make_synthetic_data(
            strikes=(80.0, 90.0, 100.0, 110.0, 120.0),
            maturities=(1.0, 1.0, 1.0, 1.0, 1.0)
        )
        result = self.c.objectiveHeston(
            self.true_params, strikes_5, mats_5, prices_5
        )
        self.assertIsInstance(result, float)


class TestCalibrationHeston(unittest.TestCase):

    def test_returns_params_array(self):
        """calibrationHeston retourne un tableau de 5 paramètres."""
        prices, strikes, maturities = make_synthetic_data()
        c = calibration()
        result = calibrationHeston(c, prices, strikes, maturities)
        self.assertEqual(len(result), 5)

    def test_calibrated_params_in_bounds(self):
        """Les paramètres calibrés doivent respecter les bornes physiques."""
        prices, strikes, maturities = make_synthetic_data()
        c = calibration()
        kappa, v0, theta, vol, cor = calibrationHeston(c, prices, strikes, maturities)
        self.assertGreater(kappa, 0.0)
        self.assertGreater(v0,    0.0)
        self.assertGreater(theta, 0.0)
        self.assertGreater(vol,   0.0)
        self.assertGreater(cor,  -1.0)
        self.assertLess(cor,      1.0)

    def test_calibrated_rmse_small(self):
        """Après calibration, la RMSE relative doit être faible (< 1%)."""
        prices, strikes, maturities = make_synthetic_data()
        c = calibration()
        calibrated = calibrationHeston(c, prices, strikes, maturities)
        rmse = c.objectiveHeston(calibrated, strikes, maturities, prices)
        self.assertLess(rmse, 0.01)


def make_sabr_synthetic_data(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4,
                              strikes=(85.0, 95.0, 100.0, 105.0, 115.0),
                              maturities=(1.0, 1.0, 1.0, 1.0, 1.0),
                              F=100.0, r=0.025):
    """Génère des prix SABR synthétiques avec des paramètres connus."""
    s   = Sabr()
    p   = SabrParams(alpha=alpha, beta=beta, rho=rho, nu=nu)
    prices = [
        s.call_price(p, SabrMarket(F=F, T=T, r=r), K)
        for K, T in zip(strikes, maturities)
    ]
    return prices, list(strikes), list(maturities)


R_SABR = 0.025
F_SABR = 100.0


class TestObjectiveSabr(unittest.TestCase):

    def setUp(self):
        self.c          = calibration()
        self.c.setUp()
        self.true_p     = [0.3, 0.4, -0.3]   # [alpha, nu, rho]
        self.prices, self.strikes, self.maturities = make_sabr_synthetic_data(
            alpha=self.true_p[0], nu=self.true_p[1], rho=self.true_p[2]
        )

    def _obj(self, params):
        return objectiveSabr(self.c, params, self.strikes, self.prices,
                             self.maturities, R_SABR, F_SABR)

    def test_returns_float(self):
        """objectiveSabr retourne un float."""
        self.assertIsInstance(self._obj(self.true_p), float)

    def test_perfect_params_gives_zero_rmse(self):
        """Avec les vrais paramètres, la RMSE doit être quasi nulle."""
        self.assertAlmostEqual(self._obj(self.true_p), 0.0, delta=1e-6)

    def test_wrong_params_gives_positive_rmse(self):
        """Des paramètres différents donnent une RMSE > 0."""
        self.assertGreater(self._obj([0.6, 0.2, 0.1]), 0.0)

    def test_rmse_non_negative(self):
        """La RMSE est toujours >= 0."""
        for p in [[0.1, 0.1, -0.5], [0.5, 0.8, 0.3], [1.0, 0.4, 0.0]]:
            self.assertGreaterEqual(self._obj(p), 0.0)

    def test_rmse_monotone_away_from_truth(self):
        """La RMSE augmente quand alpha s'éloigne de la vraie valeur."""
        close = self._obj([self.true_p[0] * 1.05, self.true_p[1], self.true_p[2]])
        far   = self._obj([self.true_p[0] * 2.0,  self.true_p[1], self.true_p[2]])
        self.assertLess(close, far)


class TestCalibrationSabr(unittest.TestCase):

    def test_returns_three_params(self):
        """CalibrationSabr retourne [alpha, nu, rho]."""
        prices, strikes, maturities = make_sabr_synthetic_data()
        c      = calibration()
        result = CalibrationSabr(c, strikes, prices, maturities)
        self.assertEqual(np.array(result).size, 3)

    def test_calibrated_alpha_positive(self):
        """L'alpha calibré doit être strictement positif."""
        prices, strikes, maturities = make_sabr_synthetic_data()
        c     = calibration()
        alpha, _, _ = CalibrationSabr(c, strikes, prices, maturities)
        self.assertGreater(alpha, 0.0)

    def test_calibrated_alpha_close_to_truth(self):
        """L'alpha calibré doit être proche du vrai alpha (tolérance 5%)."""
        true_alpha = 0.3
        prices, strikes, maturities = make_sabr_synthetic_data(alpha=true_alpha)
        c = calibration()
        alpha, _, _ = CalibrationSabr(c, strikes, prices, maturities)
        self.assertAlmostEqual(alpha, true_alpha, delta=true_alpha * 0.05)

    def test_calibrated_rmse_small(self):
        """Après calibration, la RMSE doit être faible (< 1%)."""
        prices, strikes, maturities = make_sabr_synthetic_data()
        c      = calibration()
        params = CalibrationSabr(c, strikes, prices, maturities)
        rmse   = objectiveSabr(c, params, strikes, prices, maturities, R_SABR, F_SABR)
        self.assertLess(rmse, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
