#include "../include/sabr.hpp"
#include <cmath>
#include <stdexcept>

bool SabrParams::verif() const
{
    return beta<1 && beta>0;
}
constexpr double INV_SQRT2 = 0.70710678118654752440;

double normal_cdf(double x){
    return 0.5*(1+std::erf(x*INV_SQRT2));
}

double Sabr::implied_vol(const SabrParams& params,
                         const SabrMarket& market,
                         double K) const
{
    double z=params.nu/params.alpha*std::pow(market.F*K,(1-params.beta)/2)*std::log(market.F/K);
    double x=std::log((std::sqrt(1-2*z*params.rho+z*z)+z-params.rho)/(1-params.rho));
    double y=1+(1-params.beta)*(1-params.beta)/24*std::pow(std::log(market.F/K),2);
    y+=std::pow((1-params.beta),4)/1920*std::pow(std::log(market.F/K),4);
    double v=(1-params.beta)*(1-params.beta)/24*params.alpha*params.alpha/std::pow(K*market.F,1-params.beta);
    v+=0.25*params.rho*params.beta*params.alpha*params.nu/std::pow(market.F*K,(1-params.beta)/2);
    v+=(2-3*params.rho*params.rho)/24*params.nu*params.nu;
    v=v*market.T+1;
    if(std::abs(std::log(market.F/K))<1e-6){
        return params.alpha*v/std::pow(market.F,1-params.beta);
    }
    return params.alpha*z*v/(x*y*std::pow(market.F*K,(1-params.beta)/2));
}

double Sabr::call_price(const SabrParams& params,
                        const SabrMarket& market,
                        double K) const
{
    if (K<0)
    {
        throw std::invalid_argument("K<0");
    }
    if (market.F<0)
    {
        throw std::invalid_argument("F<0");
    }
    double sigma=implied_vol(params,market,K);
    double d1=(std::log(market.F/K)+0.5*sigma*sigma*market.T)/(sigma*std::sqrt(market.T));
    double d2=(std::log(market.F/K)-0.5*sigma*sigma*market.T)/(sigma*std::sqrt(market.T));
    double D=std::exp(-market.r*market.T);
    return D*(market.F*normal_cdf(d1)-K*normal_cdf(d2));
}

std::tuple<std::vector<double>, std::vector<double>> Sabr::vol_smile(
    const SabrParams& params,
    const SabrMarket& market,
    const std::vector<double>& strikes) const
{
    std::vector<double> vols(strikes.size());
    double vol;
    for (size_t i=0;i<strikes.size();++i){
        vol=implied_vol(params,market,strikes[i]);
        vols[i]=vol;
    }
    return std::make_tuple(strikes,vols);
}
