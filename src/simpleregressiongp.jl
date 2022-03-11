function simpleregressiongp(x, y, σ; iterations = iterations, optimiser = optimiser, seed = seed, numberofrestarts = numberofrestarts, kernel = kernel)

    @assert(length(x) == length(y) == length(σ))

    rg = MersenneTwister(seed)

    N = length(x)



end
