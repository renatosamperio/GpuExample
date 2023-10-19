#' Runs a portfolio simulation from a GPU
#' 
#' 
simulate_xorshift <- function() {

    tictoc::tic()
    N <- 1000
    numbers <- vector(mode="numeric", length=N)

    # simulation engine (C++ code)
    system.time(res <- xorshift_generator(numbers, N))

    tictoc::toc()
}

