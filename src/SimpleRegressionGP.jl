module SimpleRegressionGP

    using Optim, Random, Distributions, Printf, LinearAlgebra

    using MiscUtil

    include("kernels.jl")
    include("simpleregressiongp.jl")
    
    export simpleregressiongp
end
