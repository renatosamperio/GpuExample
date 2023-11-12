#' Runs a portfolio simulation from a GPU
#' 
#' 
simulate_gpu <- function() {

    tictoc::tic()
    portfolio <- as.matrix(
        utils::read.table("./inst/data/portfolio.csv",
        sep = ";", dec = ".", header = TRUE))

    n_factor <- max(portfolio[, 1:3]) + 1
    n_sim <- 256e3

    # simulation engine (C++ code)
    system.time(
        losses <- simulator(portfolio, n_factor, n_sim)
    )

    lossess_avg <- mean(losses)
    lossess_tail_avg <- mean(losses[losses > stats::quantile(losses, .99)])

    message("Loss mean = ", lossess_avg)
    message("ES99 = ", lossess_tail_avg)

    # calculate exact mean
    lossess_truth <- sum(stats::pnorm(portfolio[, 5]) * portfolio[, 4])
    message("Loss truth = ", lossess_truth)

    tictoc::toc()
}

