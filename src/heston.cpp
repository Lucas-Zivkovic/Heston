#include "../include/heston.hpp"
#include "fft.hpp"

std::complex<double> Heston::hestoncf(const hestonPrice& params,
                                       std::complex<double> u) const {
    double x0 = std::log(params.S0);
    std::complex<double> a = params.p.vol * params.p.vol * 0.5;
    std::complex<double> b = params.p.kappa 
                           - params.p.cor * params.p.vol * I * u;
    std::complex<double> c = -0.5 * u * (I + u);
    std::complex<double> d = std::sqrt(b*b - (std::complex<double>)4*a*c);
    if (d.real() < 0) d = -d;
    std::complex<double> exp_dT = std::exp(-d * params.T);
    
    std::complex<double> g = (b - d) / (b + d);
    
    std::complex<double> B = (b - d) 
                           * ((std::complex<double>)1 - exp_dT)
                           / (params.p.vol * params.p.vol 
                              * ((std::complex<double>)1 - g * exp_dT));
    std::complex<double> A = I * u * params.r * params.T
        + params.p.kappa * params.p.theta 
          / (params.p.vol * params.p.vol)
        * ((b - d) * params.T
           - (std::complex<double>)2 
             * std::log(
                 ((std::complex<double>)1 - g * exp_dT)
                 / ((std::complex<double>)1 - g)
               )
          );
    return std::exp(x0 * I * u + A + B * params.p.v0);
}

std::complex<double> Heston::CarrMadan(double alpha,const hestonPrice& params,double v) const{
    std::complex<double> x=(std::complex<double>)v-I*(alpha+1);
    return std::exp(-params.r*params.T)/(alpha*alpha+alpha-v*v+I*v*(1+2*alpha))*hestoncf(params,x);
}


std::tuple<std::vector<double>,std::vector<double>> Heston::heston_calls_price(const hestonPrice& params,size_t N,double eta,double alpha) const{
    std::vector<std::complex<double>> y(N);
    std::complex<double> phi_shift,denom,psi;
    double a,b,v,res;
    std::complex<double> x;
    std::vector<double> w(N),k(N),calls(N);
    {
        double z = eta / 3.0;
        for (size_t j = 0; j < N; ++j) {
            double sign = (j % 2 == 0) ? -1.0 : 1.0;
            w[static_cast<size_t>(j)] = z * (3.0 + sign - (j == 0 ? 1.0 : 0.0));
        }
    }
    a=2*PI/(N*eta);
    b=0.5*N*a;
    for (size_t i=0;i<N;++i){
        v=i*eta;
        x=v-(alpha+1)*I;
        phi_shift=hestoncf(params,x);
        denom=(alpha*alpha + alpha - v*v + I*(2*alpha + 1)*v);
        psi= std::exp(-params.r*params.T) * phi_shift / denom;
        y[i]=psi*w[i]*std::exp(b*v*I);
        k[i]=-b+i*a;
    }

    fft(y,false);
    for(size_t i=0;i<N;++i){
        res=y[i].real();
        calls[i]=std::exp(-alpha*k[i])*res/PI;
        calls[i]=calls[i]>0.0?calls[i]:0.0;
    }
    return std::make_tuple(k,calls);
}


double Heston::heston_fft_call_price(const hestonPrice& params,size_t N,double eta,double alpha,double K) const{
    auto [k_grid, calls] = heston_calls_price(params, N, eta, alpha);

    double log_K = std::log(K);

    if (log_K <= k_grid.front()) return calls.front();
    if (log_K >= k_grid.back())  return calls.back();

    auto it = std::lower_bound(k_grid.begin(), k_grid.end(), log_K);
    int  i  = static_cast<int>(std::distance(k_grid.begin(), it));

    double x1 = k_grid[i - 1], x2 = k_grid[i];
    double y1 = calls[i - 1],  y2 = calls[i];
    return y1 + (y2 - y1) * (log_K - x1) / (x2 - x1);
}