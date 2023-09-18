// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// g_simulate
int g_simulate(Rcpp::NumericMatrix portfolio, int n_factor, int n_sim);
RcppExport SEXP _GpuExample_g_simulate(SEXP portfolioSEXP, SEXP n_factorSEXP, SEXP n_simSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type portfolio(portfolioSEXP);
    Rcpp::traits::input_parameter< int >::type n_factor(n_factorSEXP);
    Rcpp::traits::input_parameter< int >::type n_sim(n_simSEXP);
    rcpp_result_gen = Rcpp::wrap(g_simulate(portfolio, n_factor, n_sim));
    return rcpp_result_gen;
END_RCPP
}
// rand_num
void rand_num();
RcppExport SEXP _GpuExample_rand_num() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    rand_num();
    return R_NilValue;
END_RCPP
}
// sim
NumericVector sim(NumericMatrix portfolio, int n_factor, int n_sim);
RcppExport SEXP _GpuExample_sim(SEXP portfolioSEXP, SEXP n_factorSEXP, SEXP n_simSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type portfolio(portfolioSEXP);
    Rcpp::traits::input_parameter< int >::type n_factor(n_factorSEXP);
    Rcpp::traits::input_parameter< int >::type n_sim(n_simSEXP);
    rcpp_result_gen = Rcpp::wrap(sim(portfolio, n_factor, n_sim));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_GpuExample_g_simulate", (DL_FUNC) &_GpuExample_g_simulate, 3},
    {"_GpuExample_rand_num", (DL_FUNC) &_GpuExample_rand_num, 0},
    {"_GpuExample_sim", (DL_FUNC) &_GpuExample_sim, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_GpuExample(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
