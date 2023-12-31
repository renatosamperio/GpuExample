#' Runs a portfolio simulation in C++
#' 
#' 
simulate_cpp <- function() {

    tictoc::tic()
    portfolio <- as.matrix(
        utils::read.table("./inst/data/portfolio.csv",
        sep = ";", dec = ".", header = TRUE))

    n_factor <- max(portfolio[, 1:3]) + 1
    n_sim <- 256e3

    # simulation engine (C++ code)
    system.time(losses <- sim(portfolio, n_factor, n_sim))

    mean(losses)
    mean(losses[losses > stats::quantile(losses, .99)])

    # calculate exact mean
    sum(stats::pnorm(portfolio[, 5]) * portfolio[, 4])

    tictoc::toc()
}

