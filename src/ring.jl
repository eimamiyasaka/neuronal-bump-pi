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

# First-harmonic coupling kernel K(x-y) = a0 + a1·cos(x-y) + B·sin(x-y)
# (Laing & Omel'chenko Eq. 4: a0=0.1, a1=0.3; B is the velocity-asymmetry knob,
# B=0 ⇒ symmetric Mexican-hat). Stored as the precomputed cos/sin of the ring
# positions plus the three coefficients — NOT a dense N×N matrix — so coupling()
# can evaluate the kernel sum in O(N) via projections (see coupling). This is the
# same rank-3 structure field.jl exploits via FFT.
struct RingKernel
    cosx::Vector{Float64}
    sinx::Vector{Float64}
    a0::Float64
    a1::Float64
    B::Float64
end
make_kernel(x; a0=0.1, a1=0.3, B=0.0) = RingKernel(cos.(x), sin.(x), a0, a1, B)

# Pulse signal emitted by a single neuron when it fires (Laing & Omel'chenko Eq. 2,
# P_n(θ) = a_n(1-cosθ)^n). Elementwise over the population.
#   a_n = 2^n (n!)^2 / (2n)!  fixes the pulse area to 2π over a period (their
#         normalisation; computed in the run scripts and consistent with field.jl's
#         meanpulse, where n=2 ⇒ a_2 = 2/3).
#   n   - pulse sharpness: positive integer controlling the pulse width.
pulse(θ, a_n, n) = a_n .* (1 .- cos.(θ)).^n

# Coupling: synaptic drive felt by every neuron, I_j = (2π/N) Σ_k K(x_j-x_k) P_k
# (Laing & Omel'chenko Eq. 2; the 2π/N is the ring integration measure, matching
# field.jl). Because the kernel is first-harmonic it is rank-3: expanding
#   cos(x_j-x_k) = cos x_j cos x_k + sin x_j sin x_k
#   sin(x_j-x_k) = sin x_j cos x_k − cos x_j sin x_k
# shows every I_j depends on the population only through three scalar projections
#   S0 = Σ P_k ,  Sc = Σ P_k cos x_k ,  Ss = Σ P_k sin x_k .
# This is an EXACT rewriting of the dense Σ_k K_jk P_k (identical to round-off),
# evaluated in O(N) instead of O(N²). The B (sin) term reuses the same Sc, Ss, so
# the asymmetric / moving-bump coupling costs nothing extra. It is an instantaneous
# identity — recomputed each RK4 stage — so it holds for a moving bump too.
function coupling(P, ker::RingKernel)
    N = length(P)
    S0 = 0.0; Sc = 0.0; Ss = 0.0
    @inbounds for k in 1:N
        S0 += P[k]; Sc += P[k] * ker.cosx[k]; Ss += P[k] * ker.sinx[k]
    end
    m = 2π / N
    return m .* (ker.a0 * S0 .+ ker.a1 .* (ker.cosx .* Sc .+ ker.sinx .* Ss)
                              .+ ker.B  .* (ker.sinx .* Sc .- ker.cosx .* Ss))
end

# Drive - composes pulse and coupling (called by dynamics)
drive(θ, ker, a_n, n) = coupling(pulse(θ, a_n, n), ker)

# --- Full population right-hand side: dθ/dt for every neuron ---
# The coupling I is recomputed from whatever state θ is handed in (it depends on
# the whole state through the pulses), which is what makes the RK4 below a TRUE
# RK4 for the coupled network. Iext is an optional external drive (scalar or
# length-N), e.g. a transient bump-seeding kick.
function population_rhs(θ, η, K, a_n, n, κ; Iext=0.0)
    I = drive(θ, K, a_n, n) .+ Iext      # synaptic input felt by each neuron (length N)
    return thetadot.(θ, η .+ κ .* I)     # total drive per neuron = intrinsic η + κ·I
end

# --- RK4 phase increment Δθ for one step (the UNWRAPPED change in θ) ---
# Coupling is recomputed at every stage (not frozen at the stage-1 value). θ is
# NOT wrapped between stages — thetadot/pulse use only cos θ (2π-periodic), so
# unwrapped intermediate states give the correct RHS. Returned increment is the
# true phase advance over dt; the caller decides whether to wrap (stepping) or
# accumulate it (mean-frequency measurement). Single source of truth for the
# stage math, shared by step_population and mean_frequencies.
function rk4_increment(θ, η, K, a_n, n, κ, dt; Iext=0.0)
    k1 = population_rhs(θ,              η, K, a_n, n, κ; Iext=Iext)
    k2 = population_rhs(θ .+ 0.5dt.*k1, η, K, a_n, n, κ; Iext=Iext)
    k3 = population_rhs(θ .+ 0.5dt.*k2, η, K, a_n, n, κ; Iext=Iext)
    k4 = population_rhs(θ .+ dt .* k3,  η, K, a_n, n, κ; Iext=Iext)
    return (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

# --- One population RK4 step: advance all N phases by dt, wrapping into [0,2π) ---
function step_population(θ, η, K, a_n, n, κ, dt; Iext=0.0)
    return mod.(θ .+ rk4_increment(θ, η, K, a_n, n, κ, dt; Iext=Iext), 2π)
end

# --- Mean firing frequency of each neuron: Laing & Omel'chenko Eq. (10) ---
# f_k = (1/2π) ⟨dθ_k/dt⟩_t , the time-averaged phase velocity over a settled
# window of duration T. We accumulate the UNWRAPPED per-step RK4 increments
# (rk4_increment) — their sum is the total phase advanced — so f_k = (total
# advance)/(2π·T) is the neuron's mean rotation rate = cycles (spikes) per unit
# time. This is the paper's row-2 measurement and is the MICROSCOPIC counterpart
# of the field's rate(z) (Eq. 9): it is measured directly from the neurons,
# independent of any order parameter, so overlaying it on rate(z) is a genuine
# micro/macro test rather than re-checking the same formula. It is also smoother
# than counting θ=π crossings, which quantizes into integer spikes over a finite
# window. Run on a SETTLED state (e.g. after kick_then_release) so the average is
# taken on the stationary bump, not the transient.
function mean_frequencies(θ0, η, K, a_n, n, κ; T=200.0, dt=0.01)
    nt = round(Int, T/dt)
    θ = copy(θ0)
    advance = zeros(length(θ0))           # ∫ dθ/dt dt accumulated per neuron
    for _ in 1:nt
        inc = rk4_increment(θ, η, K, a_n, n, κ, dt)
        advance .+= inc                   # accumulate UNWRAPPED phase advance
        θ = mod.(θ .+ inc, 2π)            # wrap only the evolving state
    end
    return advance ./ (2π * T)            # mean frequency f_k (cycles per unit time)
end

# --- Time-averaged order parameter of each neuron over a settled window ---
# z̄_k = ⟨e^{-iθ_k}⟩_t. The mean field z(x) = <e^{-iθ}> is stationary for a static
# bump, but a SINGLE snapshot of the spatial average (local_order_parameter) is a
# noisy finite-sample estimate: in the firing band the cycling phases nearly cancel
# at any instant, driving |z|→0 and making arg z meaningless. Averaging e^{-iθ}
# over the settled window — the order-parameter analogue of mean_frequencies — adds
# independent temporal samples, converging to the stationary field. Spatially smooth
# the result (ring_smooth) for the per-position z(x); time and space averages commute.
# Run on a SETTLED state, exactly as for mean_frequencies.
function mean_order_parameter(θ0, η, K, a_n, n, κ; T=200.0, dt=0.01)
    nt = round(Int, T/dt)
    θ = copy(θ0)
    zacc = zeros(ComplexF64, length(θ0))  # Σ e^{-iθ}
    for _ in 1:nt
        zacc .+= exp.(-im .* θ)           # sample order parameter at current θ
        θ = step_population(θ, η, K, a_n, n, κ, dt)
    end
    return zacc ./ nt                     # time-averaged ⟨e^{-iθ}⟩ per neuron
end

# Windowed average of a per-neuron quantity over ±half neighbours on the ring
# (neurons are position-ordered; the ring wraps via mod1). Works for real or
# complex vectors — used both for the empirical z below and to draw a smooth
# f_k profile line over the raw per-neuron dots (Laing-style row 2).
function ring_smooth(v, half)
    N = length(v)
    out = similar(v)
    for j in 1:N
        acc = zero(eltype(v))
        for d in -half:half
            acc += v[mod1(j + d, N)]      # mod1 wraps the index around the ring
        end
        out[j] = acc / (2half + 1)
    end
    return out
end

# Smoothed local order parameter z(x) = <e^{-iθ}>, averaged over ±half neighbours.
# This is the CONJUGATE of Laing & Omel'chenko's z = <e^{+iθ}> (see field.jl
# header): |z| and rate(z) match theirs directly; use arg_laing for the argument.
local_order_parameter(θ, half) = ring_smooth(exp.(-im .* θ), half)

# --- Simulate population of neurons, storing the FULL trajectory ---
# Returns the N×nt phase history. Iext is an optional external drive held fixed
# across the run (scalar, or a length-N vector for a localized bump-seeding kick);
# passed through to the stepper. Use when the time course is needed (e.g. a space-
# time map). When only the final state matters, prefer evolve_population — the
# history is N×nt floats (e.g. ~3.3 GB at N=4096, T=1000, dt=0.01).
function simulate_population(θ0, η, K, a_n, n, κ; T=100.0, dt=0.01, Iext=0.0)
    nt = round(Int, T/dt)
    N = length(θ0)
    Θ_agg = zeros(N, nt)        # storage: row = neuron, col = timestep
    θ = copy(θ0)            # evolving state vector

    for i in 1:nt
        θ = step_population(θ, η, K, a_n, n, κ, dt; Iext=Iext)
        Θ_agg[:, i] = θ
    end
    return Θ_agg
end

# --- Evolve the population for duration T, returning ONLY the final state ---
# Identical dynamics to simulate_population but stores no history (O(N) memory
# instead of O(N·nt)). Use for seeding/settling at large N, where the full
# trajectory would be many GB and only the settled state is needed.
function evolve_population(θ0, η, K, a_n, n, κ; T=100.0, dt=0.01, Iext=0.0)
    nt = round(Int, T/dt)
    θ = copy(θ0)
    for _ in 1:nt
        θ = step_population(θ, η, K, a_n, n, κ, dt; Iext=Iext)
    end
    return θ
end
