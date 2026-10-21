#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "heston.hpp"
#include "sabr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(heston, m) {
    m.doc() = "Heston model pricer via Carr-Madan FFT";

    py::class_<HestonParams>(m, "HestonParams")
        .def(py::init<double, double, double, double, double>(),
             py::arg("kappa"), py::arg("v0"), py::arg("theta"),
             py::arg("vol"), py::arg("cor"))
        .def_readwrite("kappa", &HestonParams::kappa)
        .def_readwrite("v0",    &HestonParams::v0)
        .def_readwrite("theta", &HestonParams::theta)
        .def_readwrite("vol",   &HestonParams::vol)
        .def_readwrite("cor",   &HestonParams::cor)
        .def("feller_ok", &HestonParams::verif,
             "Vérifie la condition de Feller : 2*kappa*theta > vol^2")
        .def("__repr__", [](const HestonParams& p) {
            return "HestonParams(kappa=" + std::to_string(p.kappa)
                 + ", v0="    + std::to_string(p.v0)
                 + ", theta=" + std::to_string(p.theta)
                 + ", vol="   + std::to_string(p.vol)
                 + ", cor="   + std::to_string(p.cor) + ")";
        });

    py::class_<hestonPrice>(m, "HestonMarket")
        .def(py::init<double, double, double, double, HestonParams>(),
             py::arg("S0"), py::arg("r"), py::arg("T"),
             py::arg("q"), py::arg("params"))
        .def_readwrite("S0", &hestonPrice::S0)
        .def_readwrite("r",  &hestonPrice::r)
        .def_readwrite("T",  &hestonPrice::T)
        .def_readwrite("q",  &hestonPrice::q)
        .def_readwrite("params", &hestonPrice::p)
        .def("__repr__", [](const hestonPrice& h) {
            return "HestonMarket(S0=" + std::to_string(h.S0)
                 + ", r=" + std::to_string(h.r)
                 + ", T=" + std::to_string(h.T)
                 + ", q=" + std::to_string(h.q) + ")";
        });

    py::class_<Heston>(m, "Heston")
        .def(py::init<>())
        .def("call_price",
             [](const Heston& self, const hestonPrice& market, double K,
                size_t N, double eta, double alpha) {
                 return self.heston_fft_call_price(market, N, eta, alpha, K);
             },
             py::arg("market"), py::arg("K"),
             py::arg("N") = size_t(4096), py::arg("eta") = 0.25,
             py::arg("alpha") = 1.5,
             "Prix d'un call européen pour un strike K")
        .def("call_grid",
             [](const Heston& self, const hestonPrice& market,
                size_t N, double eta, double alpha) {
                 return self.heston_calls_price(market, N, eta, alpha);
             },
             py::arg("market"),
             py::arg("N") = size_t(4096), py::arg("eta") = 0.25,
             py::arg("alpha") = 1.5,
             "Retourne (log_strikes, call_prices) sur toute la grille FFT");

    // ── SABR ──────────────────────────────────────────────────────────────────
    py::class_<SabrParams>(m, "SabrParams")
        .def(py::init<double, double, double, double>(),
             py::arg("alpha"), py::arg("beta"), py::arg("rho"), py::arg("nu"))
        .def_readwrite("alpha", &SabrParams::alpha)
        .def_readwrite("beta",  &SabrParams::beta)
        .def_readwrite("rho",   &SabrParams::rho)
        .def_readwrite("nu",    &SabrParams::nu)
        .def("verif", &SabrParams::verif,
             "Vérifie que 0 < beta < 1")
        .def("__repr__", [](const SabrParams& p) {
            return "SabrParams(alpha=" + std::to_string(p.alpha)
                 + ", beta=" + std::to_string(p.beta)
                 + ", rho="  + std::to_string(p.rho)
                 + ", nu="   + std::to_string(p.nu) + ")";
        });

    py::class_<SabrMarket>(m, "SabrMarket")
        .def(py::init<double, double, double>(),
             py::arg("F"), py::arg("T"), py::arg("r"))
        .def_readwrite("F", &SabrMarket::F)
        .def_readwrite("T", &SabrMarket::T)
        .def_readwrite("r", &SabrMarket::r)
        .def("__repr__", [](const SabrMarket& m) {
            return "SabrMarket(F=" + std::to_string(m.F)
                 + ", T=" + std::to_string(m.T)
                 + ", r=" + std::to_string(m.r) + ")";
        });

    py::class_<Sabr>(m, "Sabr")
        .def(py::init<>())
        .def("implied_vol", &Sabr::implied_vol,
             py::arg("params"), py::arg("market"), py::arg("K"),
             "Volatilité implicite Black-Scholes via Hagan et al. (2002)")
        .def("call_price", &Sabr::call_price,
             py::arg("params"), py::arg("market"), py::arg("K"),
             "Prix d'un call européen")
        .def("vol_smile",
             [](const Sabr& self, const SabrParams& p, const SabrMarket& m,
                const std::vector<double>& strikes) {
                 return self.vol_smile(p, m, strikes);
             },
             py::arg("params"), py::arg("market"), py::arg("strikes"),
             "Retourne (strikes, vols) sur une grille de strikes");
}
