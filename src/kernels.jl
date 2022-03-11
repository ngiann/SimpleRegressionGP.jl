matern12(x1, x2; ρ=ρ, σ²=σ²) = MaternCovariance_12(abs(x1-x2); ρ=ρ, σ²=σ²)

function MaternCovariance_12(d; ρ=ρ, σ²=σ²)

    @assert(d >= 0)

    σ² * exp(-d/ρ)

end


matern32(x1, x2; ρ=ρ, σ²=σ²) = MaternCovariance_32(abs(x1-x2); ρ=ρ, σ²=σ²)

function MaternCovariance_32(d; ρ=ρ, σ²=σ²)

    @assert(d >= 0)

    σ² * (1 + sqrt(3)*d/ρ) * exp(- sqrt(3)*d/ρ)

end


matern52(x1, x2; ρ=ρ, σ²=σ²) = MaternCovariance_52(abs(x1-x2); ρ=ρ, σ²=σ²)

function MaternCovariance_52(d; ρ=ρ, σ²=σ²)

    @assert(d >= 0)

    σ² * (1 + sqrt(5)*d/ρ + 5*d^2/(3*ρ^2)) * exp(- sqrt(5)*d/ρ)

end


rbf(x1, x2; ρ=ρ, σ²=σ²) = RBFCovariance(abs(x1-x2); ρ=ρ, σ²=σ²)

function RBFCovariance(d; ρ=ρ, σ²=σ²)

    @assert(d >= 0)

    σ² * exp(-d^2 / (2ρ))

end


linear(x1, x2; c=c, σ²=σ²) = σ² * (x1 - c) * (x2 - c)
