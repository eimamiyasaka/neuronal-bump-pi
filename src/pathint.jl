# ============================================================================
# Phase-3 velocity-driven path-integration layer (project.md §15, Step 3.1).
#
# This is to Phase 3 what src/moving.jl is to Phase 2: a DRIVE + READOUT layer on
# top of the unchanged Phase-1/2 models — no new physics. Phase 2 swept the kernel
# asymmetry B as a static bifurcation parameter; Phase 3 makes B a *time-varying
# self-motion command*, B(t) = β·Ω(t), where Ω(t) is the commanded angular velocity
# (written Ω, NOT γ — "gamma" is reserved for the Phase-5 gamma-oscillation regime).
# See notes/writeupAssist/step3_modeling_note.md for the Ω↔B calibration (β=1/3.91,
# unity small-signal gain), the quasi-static-vs-dynamic gain split, and scope.
#
# Layer convention (as in moving.jl): MODEL/MEASUREMENT only — no Plots. Parameters
# live in scripts/run_pathint.jl.
#
# THE ONE CORRECTNESS POINT — stage-time RK4. With B(t) the field/ring systems are
# NON-AUTONOMOUS, so a true RK4 must evaluate the kernel at each stage's own time
# (t, t+dt/2, t+dt/2, t+dt). This is the explicit-time extension of the existing
# rule "recompute coupling at every RK4 stage" (ring.jl). Freezing B at the step
# start would drop the integrator to ~1st order in the command. The field exploits
# FFT linearity so NO per-stage FFT is needed: Khat(B) = Khat0 + B·Shat exactly
# (Shat = fft(sin x)); the ring rebuilds its rank-3 RingKernel (O(1)) per stage.
#
# Contents:
#   field_step_tv / step_population_tv — stage-time RK4 steppers with B(t)
#   drive_field / drive_ring           — evolve under B(t), record the centroid
#   omega_const/step/ramp/realistic    — angular-velocity command library Ω(t)
#   Bfun_from_omega                    — compose the Ω→B calibration B(t)=β·Ω(t)
#   heading_estimate / commanded_heading / instantaneous_speed — PI readouts
# ============================================================================

include("moving.jl")    # unwrap, bump_centroid, lateral_speed, track_*_centroid, seed_*_bump
                        # (+ field.jl: field_rhs, rate, field_kernel, field_positions, …)
                        # (+ ring.jl:  population_rhs, pulse, RingKernel, make_positions, …)

# --- Field: one stage-time RK4 step under a time-varying B(t) -----------------
# Khat0 = fft(0.1 + 0.3cos x), Shat = fft(sin x) are precomputed ONCE by the caller;
# the stage kernel transform is the exact linear combination Khat0 .+ B(t).*Shat
# (fft is linear, so this equals fft(field_kernel(x; B=B(t))) to round-off). Bfun is
# the kernel-level command t ↦ B(t). Reuses the unchanged field_rhs.
function field_step_tv(z, t, η̄, Δ, κ, Khat0, Shat, Bfun, dt; Iext=0.0)
    Kh(tt) = Khat0 .+ Bfun(tt) .* Shat
    k1 = field_rhs(z,              η̄, Δ, κ, Kh(t);          Iext=Iext)
    k2 = field_rhs(z .+ 0.5dt.*k1, η̄, Δ, κ, Kh(t + 0.5dt); Iext=Iext)
    k3 = field_rhs(z .+ 0.5dt.*k2, η̄, Δ, κ, Kh(t + 0.5dt); Iext=Iext)
    k4 = field_rhs(z .+ dt  .*k3,  η̄, Δ, κ, Kh(t + dt);    Iext=Iext)
    return z .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

# --- Ring: one stage-time RK4 step under a time-varying B(t) ------------------
# cosx/sinx of the ring positions are precomputed once; only B changes per stage,
# so the rank-3 RingKernel is rebuilt in O(1) (no make_kernel — that recomputes
# cos/sin). Reuses the unchanged population_rhs. a0,a1 are the kernel constants.
function step_population_tv(θ, t, η, cosx, sinx, a0, a1, a_n, n, κ, Bfun, dt; Iext=0.0)
    K(tt) = RingKernel(cosx, sinx, a0, a1, Bfun(tt))
    k1 = population_rhs(θ,              η, K(t),          a_n, n, κ; Iext=Iext)
    k2 = population_rhs(θ .+ 0.5dt.*k1, η, K(t + 0.5dt),  a_n, n, κ; Iext=Iext)
    k3 = population_rhs(θ .+ 0.5dt.*k2, η, K(t + 0.5dt),  a_n, n, κ; Iext=Iext)
    k4 = population_rhs(θ .+ dt  .*k3,  η, K(t + dt),     a_n, n, κ; Iext=Iext)
    return mod.(θ .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4), 2π)
end

# --- Field: evolve under B(t), recording the bump centroid each step ----------
# The time-varying-B analogue of track_field_centroid. Returns the time grid, the
# centroid series (centre of the firing-rate profile), and the final field z (for
# chaining). cosx/sinx default to the field grid's. Bfun is the kernel-level B(t).
function drive_field(z0, η̄, Δ, κ, x, Bfun; T=200.0, dt=0.02)
    Khat0 = fft(field_kernel(x; B=0.0))            # symmetric part transform
    Shat  = fft(sin.(x))                           # antisymmetric part transform
    cosx, sinx = cos.(x), sin.(x)
    nt = round(Int, T/dt)
    z  = copy(z0)
    ts = Vector{Float64}(undef, nt)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        t = (i - 1) * dt
        z = field_step_tv(z, t, η̄, Δ, κ, Khat0, Shat, Bfun, dt)
        ts[i] = i * dt
        xc[i] = bump_centroid(rate.(z), cosx, sinx)
    end
    return ts, xc, z
end

# --- Spiking net: evolve under B(t), recording the bump centroid each step -----
# Microscopic counterpart of drive_field. Centre read from the instantaneous pulse
# activity (as in track_ring_centroid). a0,a1 are the kernel constants. Returns the
# time grid, centroid series, and final phase state θ.
function drive_ring(θ0, η, x, a_n, n, κ, Bfun; a0=0.1, a1=0.3, T=200.0, dt=0.02)
    cosx, sinx = cos.(x), sin.(x)
    nt = round(Int, T/dt)
    θ  = copy(θ0)
    ts = Vector{Float64}(undef, nt)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        t = (i - 1) * dt
        θ = step_population_tv(θ, t, η, cosx, sinx, a0, a1, a_n, n, κ, Bfun, dt)
        ts[i] = i * dt
        xc[i] = bump_centroid(pulse(θ, a_n, n), cosx, sinx)
    end
    return ts, xc, θ
end

# ============================================================================
# Profile-recording drivers (for the visualization in scripts/animate_ring.jl).
#
# Identical dynamics to drive_field/drive_ring above — same stage-time RK4 steppers,
# no new physics — but they additionally save the full SPATIAL ACTIVITY PROFILE every
# `stride` steps so the moving bump can be drawn, not just its centroid. The scalar
# outputs (ts, centroid) match drive_field/drive_ring sampled on the frame grid, so an
# animation built from these is the same run the CSVs came from.
# ============================================================================

# Field: record the firing-rate profile rate(z) at each frame. Returns the frame time
# grid, an (Nx × nframes) matrix of rate profiles, and the centroid series.
function drive_field_record(z0, η̄, Δ, κ, x, Bfun; T=200.0, dt=0.02, stride=25)
    Khat0 = fft(field_kernel(x; B=0.0))
    Shat  = fft(sin.(x))
    cosx, sinx = cos.(x), sin.(x)
    nt = round(Int, T/dt)
    nf = fld(nt, stride)
    z  = copy(z0)
    ts = Vector{Float64}(undef, nf)
    xc = Vector{Float64}(undef, nf)
    R  = Matrix{Float64}(undef, length(x), nf)      # rate profile per frame
    f  = 0
    for i in 1:nt
        t = (i - 1) * dt
        z = field_step_tv(z, t, η̄, Δ, κ, Khat0, Shat, Bfun, dt)
        if i % stride == 0
            f += 1
            r = rate.(z)
            R[:, f] = r
            ts[f] = i * dt
            xc[f] = bump_centroid(r, cosx, sinx)
        end
    end
    return ts, R, xc
end

# Spiking net: record a SMOOTHED pulse-activity profile (the firing-intensity bump),
# downsampled to `ndisp` display points so 8192 neurons render cheaply. `half` is the
# ring_smooth half-window. Returns the frame times, the display angle grid, an
# (ndisp × nframes) activity matrix, and the centroid (from the full, unsmoothed pulse).
function drive_ring_record(θ0, η, x, a_n, n, κ, Bfun;
                           a0=0.1, a1=0.3, T=200.0, dt=0.02, stride=25, ndisp=256, half=256)
    cosx, sinx = cos.(x), sin.(x)
    N  = length(θ0)
    idx = collect(1:max(1, fld(N, ndisp)):N)        # display neuron indices
    xdisp = x[idx]
    nt = round(Int, T/dt)
    nf = fld(nt, stride)
    θ  = copy(θ0)
    ts = Vector{Float64}(undef, nf)
    xc = Vector{Float64}(undef, nf)
    A  = Matrix{Float64}(undef, length(idx), nf)    # smoothed pulse activity per frame
    f  = 0
    for i in 1:nt
        t = (i - 1) * dt
        θ = step_population_tv(θ, t, η, cosx, sinx, a0, a1, a_n, n, κ, Bfun, dt)
        if i % stride == 0
            f += 1
            P = pulse(θ, a_n, n)
            A[:, f] = ring_smooth(P, half)[idx]
            ts[f] = i * dt
            xc[f] = bump_centroid(P, cosx, sinx)
        end
    end
    return ts, xdisp, A, xc
end

# ============================================================================
# Angular-velocity command library — each returns a closure Ω: t ↦ Ω(t).
# Compose with Bfun_from_omega to get the kernel-level B(t) the drivers consume.
# ============================================================================

# Constant self-motion: a steady turn at angular velocity Ω0.
omega_const(Ω0) = t -> Ω0

# Piecewise-constant command (step changes). `t_edges` are the switch times (sorted
# ascending); `levels` holds the value on each interval and must have length
# length(t_edges)+1 (the first level applies before t_edges[1]). Tests the bump's
# response to abrupt velocity changes (Step 3.1 case b; the lag/branch-jump probe).
function omega_step(t_edges, levels)
    @assert length(levels) == length(t_edges) + 1 "levels must be one longer than t_edges"
    return t -> levels[searchsortedlast(t_edges, t) + 1]
end

# Linear ramp from Ω0 (for t≤t0) to Ω1 (for t≥t1), linear in between. A smooth
# acceleration — the cleanest probe of where the quasi-static gain starts to lag.
function omega_ramp(Ω0, Ω1, t0, t1)
    return t -> t <= t0 ? Ω0 : t >= t1 ? Ω1 : Ω0 + (Ω1 - Ω0) * (t - t0) / (t1 - t0)
end

# Realistic band-limited command: an Ornstein–Uhlenbeck angular-velocity trace
# (mean Ωbar, std σ, correlation time τ), precomputed on the dt grid and returned
# as a linearly-interpolating closure (clamped past the ends). Reproducible via the
# supplied rng. Keep σ,Ωbar inside the monostable band for the clean PI demo
# (modeling note §6); a larger-amplitude trace is the branch-trapping probe.
function omega_realistic(T, dt; τ=20.0, σ=0.05, Ωbar=0.0, rng=Random.MersenneTwister(2))
    nt = round(Int, T/dt) + 1
    Ω  = Vector{Float64}(undef, nt)
    Ω[1] = Ωbar
    a = exp(-dt/τ)                                  # OU exact discrete update
    b = σ * sqrt(1 - a^2)
    for i in 2:nt
        Ω[i] = Ωbar + a * (Ω[i-1] - Ωbar) + b * randn(rng)
    end
    return function (t)
        u = clamp(t/dt, 0.0, nt - 1 - 1e-9)         # fractional grid index
        j = floor(Int, u); f = u - j
        return (1 - f) * Ω[j + 1] + f * Ω[j + 2]    # linear interpolation (1-based)
    end
end

# Compose the Ω→B calibration: kernel asymmetry B(t) = β·Ω(t) (modeling note §1–2).
Bfun_from_omega(Ωfun, β) = t -> β * Ωfun(t)

# ============================================================================
# Path-integration readouts.
# ============================================================================

# Decoded heading φ(t): the bump centroid unwrapped over time into a continuous
# angle (the encoded heading the network reports). Reuses moving.jl's unwrap.
heading_estimate(xc) = unwrap(xc)

# Commanded ("true") heading Φ(t) = φ0 + ∫₀ᵗ Ω dt', by the trapezoidal rule on the
# time grid ts. This is the ideal a unity-gain integrator would reproduce.
function commanded_heading(ts, Ωfun; φ0=0.0)
    Φ = similar(ts)
    Φ[1] = φ0
    for i in 2:length(ts)
        Φ[i] = Φ[i-1] + 0.5 * (Ωfun(ts[i]) + Ωfun(ts[i-1])) * (ts[i] - ts[i-1])
    end
    return Φ
end

# Instantaneous bump angular velocity dφ/dt from the centroid series. The unwrapped
# centroid is box-smoothed over ±hw points (a few t.u.) before central differencing
# — the field centroid is clean but the spiking one is noisy, so smoothing keeps the
# microscopic speed readable. Endpoints use one-sided differences. dt is the step.
function instantaneous_speed(ts, xc; window=2.0)
    u  = unwrap(xc)
    dt = ts[2] - ts[1]
    hw = max(1, round(Int, window / dt))
    n  = length(u)
    us = similar(u)                                 # box-smoothed heading
    for i in 1:n
        lo = max(1, i - hw); hi = min(n, i + hw)
        us[i] = sum(@view u[lo:hi]) / (hi - lo + 1)
    end
    v = similar(u)
    for i in 2:n-1
        v[i] = (us[i+1] - us[i-1]) / (2dt)
    end
    v[1] = (us[2] - us[1]) / dt
    v[n] = (us[n] - us[n-1]) / dt
    return v
end
