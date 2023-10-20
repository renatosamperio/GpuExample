#' Runs a portfolio simulation from a GPU
#' 
#' @param N `numeric` sample size
#' 
simulate_xorshift <- function(N = 1000000) {
    
    numbers <- vector(mode="numeric", length=N)
    message(paste0("Running random Xorshift for ",N, " samples"))

    # simulation engine (C++ code)
    system.time(res <- xorshift_generator(numbers, N))

    list(mean = mean(res),
    stdev = sd(res),
    length = length(res),
    out = res)
    
}

