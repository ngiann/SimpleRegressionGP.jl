function simpleregressiongp(x, y; iterations = iterations, seed = 1, numberofrestarts = 1, kernel = kernel)

    @assert(length(x) == length(y))

    rg = MersenneTwister(seed)

    N = length(x)

    JITTER = 1e-8


    function unpack(param)

        @assert(length(param) == 4)

        exp(param[1]), exp(param[2]), exp(param[3]), param[4]

    end


    function objective(param)

        local ρ, α, σ², b = unpack(param)

        local K = Symmetric(calculatekernelmatrix(kernel, ρ, α, x) + σ²*I + JITTER*I)

        -logpdf(MvNormal(zeros(N), K), y .- b)

    end

    safeobj = safewrapper(objective)


    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    solve() = optimize(safeobj, randn(rg, 4)*3, LBFGS(), opt, autodiff=:forward)

    solutions = [solve() for _ in 1:numberofrestarts]

    bestindex = argmin([s.minimum for s in solutions])

    ρopt, αopt, σ²opt, bopt = unpack(solutions[bestindex].minimizer)

    K = Symmetric(calculatekernelmatrix(kernel, ρopt, αopt, x)  + σ²opt*I + JITTER*I)



    function predicttest(xtest)

        # dimensions: N × Ntest
        kB✴ = calculatekernelmatrix(kernel, ρopt, αopt, x, xtest)

        # Ntest × Ntest
        cB = calculatekernelmatrix(kernel, ρopt, αopt, xtest)

        # full predictive covariance
        Σpred = Symmetric(cB - kB✴' * (K \ kB✴) + JITTER*I)

        # predictive mean

        μpred = kB✴' * (K \ (y.-bopt)) .+ bopt

        return μpred, Σpred

    end

    function testloglikelihood(xtest, ytest)

        μpred, Σpred = predicttest(xtest)

        logpdf(MvNormal(μpred, Σpred), ytest)

    end

    return solutions[bestindex].minimum, predicttest, testloglikelihood

end
