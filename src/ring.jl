using Random

# Shared theta-neuron model: `thetadot` (and the single-neuron helpers) live in
# single.jl. Path is resolved relative to this file's directory, so it works no
# matter which script includes ring.jl.
include("single.jl")

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
coupling(P, K) = (K * P) .* (2π / length(P))

# Drive - composes pulse and coupling (called by dynamics)
drive(θ, K, a_n, n) = coupling(pulse(θ, a_n, n), K)

# --- Full population right-hand side: dθ/dt for every neuron ---
# The coupling I is recomputed from whatever state θ is handed in (it depends on
# the whole state through the pulses), which is what makes the RK4 below a TRUE
# RK4 for the coupled network. Iext is an optional external drive (scalar or
# length-N), e.g. a transient bump-seeding kick.
function population_rhs(θ, η, K, a_n, n, κ; Iext=0.0)
    I = drive(θ, K, a_n, n) .+ Iext      # synaptic input felt by each neuron (length N)
    return thetadot.(θ, η .+ κ .* I)     # total drive per neuron = intrinsic η + κ·I
end

# --- One population RK4 step: advance all N phases by dt ---
# Coupling is recomputed at every stage (not frozen at the stage-1 value). θ is
# NOT wrapped between stages — thetadot/pulse use only cos θ (2π-periodic), so
# unwrapped intermediate states give the correct RHS; only the final state wraps.
function step_population(θ, η, K, a_n, n, κ, dt; Iext=0.0)
    k1 = population_rhs(θ,              η, K, a_n, n, κ; Iext=Iext)
    k2 = population_rhs(θ .+ 0.5dt.*k1, η, K, a_n, n, κ; Iext=Iext)
    k3 = population_rhs(θ .+ 0.5dt.*k2, η, K, a_n, n, κ; Iext=Iext)
    k4 = population_rhs(θ .+ dt .* k3,  η, K, a_n, n, κ; Iext=Iext)
    θ_new = θ .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
    return mod.(θ_new, 2π)               # wrap each phase into [0, 2π)
end

# Smoothed local order parameter z(x) = <e^{-iθ}>, averaged over a window of ±half
# neighbouring neurons on the ring (neurons are position-ordered). Matches the
# field's convention z = <e^{-iθ}>, so |z| and rate(z) are comparable to field.jl.
function local_order_parameter(θ, half)
    N = length(θ)
    e = exp.(-im .* θ)
    z = similar(e)
    for j in 1:N
        acc = 0.0 + 0.0im
        for d in -half:half
            acc += e[mod1(j + d, N)]      # mod1 wraps the index around the ring
        end
        z[j] = acc / (2half + 1)
    end
    return z
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
