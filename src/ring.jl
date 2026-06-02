using Random

# --- Theta-neuron right-hand side ---
# θ̇ = (1 - cos θ) + (1 + cos θ) * η
thetadot(θ, η) = (1-cos(θ)) + (1 + cos(θ)) * η

# Neuron positions on the ring [0, 2π]
function make_positions(N)
    return [2π*(j-1)/N for j in 1:N]
end

# Heterogeneous excitabilities via deterministic Cauchy (lorentian) quantiles,
# shuffled so excitability is uncorrelated with ring position
function make_excitabilities(N, η̄, Δ; rng=Random.default_rng())
    u = [(2j - 1) / (2N) for j in 1:N]          # bin midpoints in (0,1)
    η = η̄ .+ Δ .* tan.(π .* (u .- 0.5))          # inverse-CDF of Cauchy
    shuffle!(rng, η)
    return η
end

# Symmetric coupling kernel matrix K[j,k] = 0.1 + 0.3 cos(x_j - x_k)
function make_kernel(x)
    N = length(x)
    K = Matrix{Float64}(undef, N, N)
    for k in 1:N, j in 1:N
        K[j, k] = 0.1 + 0.3 * cos(x[j] - x[k])
    end
    return K
end

# --- One RK4 step of size dt ---
function rk4_step(θ, η, dt)
    k1 = thetadot(θ,            η)
    k2 = thetadot(θ + 0.5dt*k1, η)
    k3 = thetadot(θ + 0.5dt*k2, η)
    k4 = thetadot(θ + dt*k3,    η)
    return θ + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# --- Simulate one neuron, return time, θ-trace, and spike times ---
function simulate(η; T=100.0, dt=0.01, θ0=0.0)
    nt = round(Int, T/dt)
    θ = θ0
    ts     = Vector{Float64}(undef, nt)
    θtrace = Vector{Float64}(undef, nt)
    spikes = Float64[]

    for i in 1:nt
        θ_new = rk4_step(θ, η, dt)
        # spike if θ crossed π upward during this step
        if θ < π && θ_new >= π
            push!(spikes, i*dt)
        end
        θ = mod(θ_new, 2π)
        ts[i]     = i*dt
        θtrace[i] = θ
    end
    return ts, θtrace, spikes
end