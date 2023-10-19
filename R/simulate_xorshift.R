#' Runs a portfolio simulation from a GPU
#' 
#' 
simulate_xorshift <- function(N = 1000000) {
    
    numbers <- vector(mode="numeric", length=N)
    message(paste0("Running random Xorshift for ",N, " samples"))

    # simulation engine (C++ code)
    system.time(res <- xorshift_generator(numbers, N))

    list(mean = mean(res),
    std = sd(res),
    length = length(res))
}

