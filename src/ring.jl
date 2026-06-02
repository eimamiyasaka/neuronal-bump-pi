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

# Pulse signal emitted by a single neuron (elementwise)
# a_n - regulator so pulse has unit area over a period (standard theta-neuron normalisation is "a_n = 2^n (n!)^2 / (2n)!)"; to be tuned later
# n - pulse sharpness, i.e. positive integer which controls the width of the pulse
pulse(θ, a_n, n) = a_n .* (1 .- cos.(θ)).^n

# Coupling - turns the pulse field into the drive felt by every neuron
coupling(P, K) = (K * P) ./ length(P)

# Drive - composes pulse and coupling (called by dynamics)
drive(θ, K, a_n, n) = coupling(pulse(θ, a_n, n), K)

# --- One RK4 step of size dt ---
function rk4_step(θ, η, dt)
    k1 = thetadot.(θ,            η)
    k2 = thetadot.(θ + 0.5dt*k1, η)
    k3 = thetadot.(θ + 0.5dt*k2, η)
    k4 = thetadot.(θ + dt*k3,    η)
    return θ + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# --- One population RK4 step: advance all N phases by dt ---
# I (coupling drive) is computed once from θ and held fixed across the 4 RK4 stages
function step_population(θ, η, K, a_n, n, κ, dt)
    I = drive(θ, K, a_n, n)         # synaptic input felt by each neuron (length N) - drive I depends on θ through the pulses, so it should later be recomputed inside each stage (!!!)
    θ_new = rk4_step(θ, η .+ (κ .* I), dt) # total drive per neuron = intrinsic η + synaptic I
    return mod.(θ_new, 2π)          # wrap each phase into [0, 2π)
end

# --- Simulate one neuron, return time, θ-trace, and spike times ---
function simulate_single(η; T=100.0, dt=0.01, θ0=0.0)
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

# --- Simulate population of neurons ---
function simulate_population(θ0, η, K, a_n, n, κ; T=100.0, dt=0.01)
    nt = round(Int, T/dt)
    N = length(θ0)
    Θ_agg = zeros(N, nt)        # storage: row = neuron, col = timestep
    θ = copy(θ0)            # evolving state vector

    for i in 1:nt
        θ = step_population(θ, η, K, a_n, n, κ, dt)
        Θ_agg[:, i] = θ
    end
    return Θ_agg
end