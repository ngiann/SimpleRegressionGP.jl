function simpleregressiongp(x, y; iterations = iterations, seed = 1, numberofrestarts = 1, kernel = kernel)

    @assert(length(x) == length(y))

    rg = MersenneTwister(seed)

    N = length(x)

    JITTER = 1e-8


    function unpack(param)

        @assert(length(param) == 3)

        exp.(param)

    end


    function objective(param)

        local ρ, α, σ² = unpack(param)

        local K = calculatekernelmatrix(kernel, ρ, α, x) + σ²*I

        -logpdf(MvNormal(zeros(N), K), y)

    end

    safeobj = safewrapper(objective)


    opt = Optim.Options(iterations = iterations, show_trace = false, show_every = 10)

    solve() = optimize(safeobj, randn(rg, 3)*3, NelderMead(), opt)

    solutions = [solve() for _ in 1:numberofrestarts]

    bestindex = argmin([s.minimum for s in solutions])

    ρopt, αopt, σ²opt = unpack(solutions[bestindex].minimizer)

    K = calculatekernelmatrix(kernel, ρopt, αopt, x) + σ²opt*I



    function predicttest(xtest)

        # dimensions: N × Ntest
        kB✴ = calculatekernelmatrix(kernel, ρopt, αopt, x, xtest)

        # Ntest × Ntest
        cB = calculatekernelmatrix(kernel, ρopt, αopt, xtest)

        # full predictive covariance
        Σpred = cB - kB✴' * (K \ kB✴)+ JITTER*I

        makematrixsymmetric!(Σpred)

        # predictive mean

        μpred = kB✴' * (K \ y)

        return μpred, Σpred

    end

    return solutions[bestindex].minimum, predicttest, (ρopt, αopt, σ²opt)

end
