# ============================================================================
# Moving-bump measurement + B-sweep layer (Laing & Omel'chenko 2020, Figs. 2–3).
#
# This file is the Phase-2 extension of the project: it adds the machinery needed
# to *move* the bump and *measure how fast it moves*, on top of the unchanged
# Phase-1 models. The asymmetry knob B (the `B sin x` term of the kernel) is
# already plumbed through both models — `field_kernel(x; B=...)` in field.jl and
# `make_kernel(x; B=...)` in ring.jl — so nothing in src/field.jl or src/ring.jl
# changes; the only new physics is reading out the bump's lateral speed s(B).
#
# Layer convention (mirrors the existing split): this file is MODEL/MEASUREMENT
# logic only — no Plots. Presentation lives in scripts/plotting.jl, parameters in
# the run scripts (scripts/run_moving.jl, scripts/run_sweep.jl).
#
# Contents:
#   bump_centroid        — bump centre = phase of the 1st spatial Fourier mode
#   unwrap / lateral_speed — turn a centroid time series into a mean speed s
#   track_field_centroid — evolve the field, record only the centroid each step
#   track_ring_centroid  — evolve the spiking net, record only the centroid
#   seed_field_bump / seed_ring_bump — settle a motionless B=0 bump (sweep IC)
#   bsweep_field / bsweep_ring — warm-started forward/backward B-sweeps → s(B)
# ============================================================================

include("field.jl")     # rate, meanpulse, field_kernel, field_step, simulate_field, rest_state, angular_distance
include("ring.jl")      # pulse, make_kernel, step_population, evolve_population, RingKernel

# --- Bump centre: phase of the first spatial Fourier mode of the activity ---
# For a localized activity profile a(x_k) ≥ 0 on the ring, Σ_k a_k e^{i x_k} is the
# population vector; its argument is the bump's angular position. This is the same
# rank-1 projection the kernel already uses (cosx, sinx), so it is O(N) and needs
# no x stored — pass the precomputed cos/sin of the ring positions (the field grid,
# or a RingKernel's own ker.cosx/ker.sinx). Returns an angle in (−π, π]; the speed
# code unwraps it, so the branch does not matter here.
bump_centroid(activity, cosx, sinx) = angle(sum(activity .* complex.(cosx, sinx)))

# --- Phase unwrap: remove 2π jumps so a wrapping series becomes continuous ---
# Base Julia has no unwrap; this is the standard one (add/subtract 2π so each step
# is the shortest arc). Used to turn a centroid that wraps around the 0/2π seam
# into a monotone displacement whose slope is the lateral speed.
function unwrap(p)
    out = copy(float.(p))
    for i in 2:length(out)
        d = out[i] - out[i-1]
        out[i] = out[i-1] + (d - 2π*round(d/(2π)))   # shortest-arc increment
    end
    return out
end

# --- Mean lateral speed from a centroid time series ---
# s = (net unwrapped displacement)/(elapsed time) over the LAST `frac` of the run.
# Measuring on the tail discards the settling transient, so s is the speed of the
# settled travelling wave — exactly the paper's protocol ("mean lateral speed from
# the last 1000 of 2000 time units"; there frac = 0.5). dt is the integrator step.
function lateral_speed(xc, dt; frac=0.5)
    u  = unwrap(xc)
    nt = length(u)
    i0 = max(1, floor(Int, (1 - frac) * nt))     # first index of the measurement tail
    return (u[nt] - u[i0]) / ((nt - i0) * dt)
end

# --- Field: evolve and record ONLY the bump centroid each step ---
# Same dynamics as simulate_field, but instead of the O(Nx·nt) trajectory it keeps
# just the scalar centroid per step (O(nt)) — all the sweep needs. Returns the
# centroid series and the final field z (for warm-starting the next B). cosx/sinx
# are the field grid's cos/sin, precomputed once by the caller.
function track_field_centroid(z0, η̄, Δ, κ, Khat, cosx, sinx; T=200.0, dt=0.02)
    nt = round(Int, T/dt)
    z  = copy(z0)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        z = field_step(z, η̄, Δ, κ, Khat, dt)
        xc[i] = bump_centroid(rate.(z), cosx, sinx)   # centre of the firing-rate profile
    end
    return xc, z
end

# --- Spiking net: evolve and record ONLY the bump centroid each step ---
# The microscopic counterpart of track_field_centroid. The bump centre is read from
# the instantaneous pulse activity P_k (large in the firing band, ~0 at rest), so it
# needs no order parameter. Reuses the kernel's stored ker.cosx/ker.sinx for the
# projection. O(N) per step, O(nt) memory — safe at N = 8192 over long sweeps.
function track_ring_centroid(θ0, η, K::RingKernel, a_n, n, κ; T=200.0, dt=0.02)
    nt = round(Int, T/dt)
    θ  = copy(θ0)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        θ = step_population(θ, η, K, a_n, n, κ, dt)
        xc[i] = bump_centroid(pulse(θ, a_n, n), K.cosx, K.sinx)
    end
    return xc, θ
end

# --- Seed a settled, motionless bump at B = 0 (the sweep's initial condition) ---
# These mirror the kick_then_release protocol in run_field.jl / run_ring.jl, lifted
# into the model layer so the sweep scripts are self-contained: relax to rest →
# transient localized Gaussian kick to carve a bump → release and settle. At Δ = 0.1
# the field damps at rate ~Δ, so T_free ≈ 200 (~20/Δ) settles cleanly (vs ~1000 at
# the Δ = 0.01 static point). Return the settled state used to start the B-sweep.
function seed_field_bump(x, η̄, Δ, κ, Khat;
                         A=0.6, σ=0.6, x0=π, T_kick=20.0, T_free=200.0, dt=0.02)
    z0   = fill(rest_state(η̄, Δ), length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    Zk = simulate_field(z0, η̄, Δ, κ, Khat; T=T_kick, dt=dt, Iext=kick)   # carve bump
    Zf = simulate_field(Zk[:, end], η̄, Δ, κ, Khat; T=T_free, dt=dt)       # release/settle
    return Zf[:, end]
end

function seed_ring_bump(η, K::RingKernel, a_n, n, κ, x;
                        A=0.6, σ=0.6, x0=π, T_relax=30.0, T_kick=20.0, T_free=200.0, dt=0.02)
    θ0   = zeros(length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    θr = evolve_population(θ0, η, K, a_n, n, κ; T=T_relax, dt=dt)            # → rest
    θk = evolve_population(θr, η, K, a_n, n, κ; T=T_kick,  dt=dt, Iext=kick) # carve bump
    θf = evolve_population(θk, η, K, a_n, n, κ; T=T_free,  dt=dt)            # release/settle
    return θf
end

# --- Field B-sweep: speed s for each B in Bs, warm-started along the list ---
# For each B (in the given order) rebuild the kernel transform and evolve from the
# PREVIOUS B's settled field — this continuation is what exposes the hysteresis:
# pass an ASCENDING Bs for the forward sweep, a DESCENDING Bs for the backward one,
# and the two trace different branches wherever the system is multistable. Only the
# scalar speed per B is kept; the field state is carried internally for the next B.
# Returns (speeds aligned with Bs, final field state) — the final state lets a
# backward sweep continue from where a forward sweep ended (the natural way to
# close the hysteresis loop). Cheap: 256-point grid, so a fine Bs is fine.
function bsweep_field(Bs, x, η̄, Δ, κ, z_start; T=200.0, dt=0.02, frac=0.5)
    cosx, sinx = cos.(x), sin.(x)
    s = Vector{Float64}(undef, length(Bs))
    z = copy(z_start)
    for (i, B) in enumerate(Bs)
        Khat = fft(field_kernel(x; B=B))                       # kernel changes only via B
        xc, z = track_field_centroid(z, η̄, Δ, κ, Khat, cosx, sinx; T=T, dt=dt)
        s[i] = lateral_speed(xc, dt; frac=frac)
    end
    return s, z
end

# --- Spiking-net B-sweep: the microscopic counterpart of bsweep_field ---
# Same warm-started continuation, but the only B-dependence is the kernel's B
# coefficient, so we rebuild the (cheap, rank-3) RingKernel each step rather than
# refitting anything. Carries the phase state θ between B values. Returns (s aligned
# with Bs, final phase state) — same chaining contract as bsweep_field. This is the
# expensive side (N neurons × nt steps × |Bs|) — keep Bs coarse unless reproducing
# the paper exactly.
function bsweep_ring(Bs, x, η, a_n, n, κ, θ_start; T=200.0, dt=0.02, frac=0.5)
    s = Vector{Float64}(undef, length(Bs))
    θ = copy(θ_start)
    for (i, B) in enumerate(Bs)
        K = make_kernel(x; B=B)                                # rank-3 kernel, O(N) to build
        xc, θ = track_ring_centroid(θ, η, K, a_n, n, κ; T=T, dt=dt)
        s[i] = lateral_speed(xc, dt; frac=frac)
    end
    return s, θ
end
