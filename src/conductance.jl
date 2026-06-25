# ============================================================================
# Phase-4 conductance-based, Dale-compliant synapse layer (project.md §15, Step 4.1).
#
# This is to Phase 4 what src/moving.jl is to Phase 2 and src/pathint.jl is to Phase 3:
# a new layer ON TOP of the unchanged Phase-1/2/3 models. Unlike those two it adds
# real new *physics* — conductance synapses g·(E_syn − v) replacing the current-based
# coupling η + κI — but the entire change collapses to ONE extra term per model (the
# derivation + all modeling decisions are in notes/writeupAssist/step4_modeling_note.md):
#
#   ring (theta):  θ̇ = (1−cosθ) + (1+cosθ)(η + J) − G·sinθ          (adds  − G·sinθ)
#   field (z):     ż = (i/2)(z−1)² − (1/2)(z+1)²[Δ + i(η̄+J)] + (G/2)(z²−1)
#                                                                     (adds +(G/2)(z²−1))
#
# Dale split of the (unchanged) Mexican-hat kernel K = 0.1 + 0.3cos + B sin into two
# NON-NEGATIVE conductance channels (note §2):
#   K_E = 0.4 + 0.3cos + B sin   (excitation + the velocity-shift; rides on excitation)
#   K_I = 0.3                     (uniform inhibition)        K_E − K_I = K exactly
#
# STRUCTURED INHIBITION (Phase-5 divisive-shunting extension; default OFF). The flat
# K_I above makes the inhibitory conductance g_I = g0∫K_I·P spatially UNIFORM, so the
# shunt is subtractive (it holds the silent surround but cannot reshape the bump) and
# cranking g0 floods/collapses the bump rather than dividing its gain. Giving K_I a
# cosine structure (a1=aI>0) makes g_I(x) TRACK the bump, so the shunt G(x) is localized
# and the inhibition acts DIVISIVELY at the bump. To keep the NET kernel (hence the
# Phase-1–4 bump) unchanged, the caller must preserve K_E − K_I = 0.1 + 0.3cos + B sin,
# i.e. cE−cI = 0.1 and aE−aI = 0.3 (so structuring I by aI means aE = 0.3+aI). Dale
# non-negativity caps aI: K_E = cE+aE cos+B sin ≥ 0 ⇒ aI ≲ 0.1 at B=0. aI defaults to
# 0.0 so every existing call is bit-identical to the flat-K_I model. See
# scripts/run_structured_inhibition.jl and notes/writeupAssist/step5_divisive_gate.md.
#   g_E = κ·∫K_E·P ≥ 0 ,  g_I = κ·∫K_I·P ≥ 0     (instantaneous — no extra ODEs)
#   J = E_E·g_E + E_I·g_I   (reversal-weighted current → enters η̄+J)
#   G = g_E + g_I           (total conductance → the new shunting term)
#
# Why closure survives (Hinge A): the conductance term is LINEAR in v, so the OA
# reduction stays of the tolerated A+Bv+Cv² form; the closed field equation above is
# the single analytic term (G/2)(z²−1) — confirmed numerically by the micro/macro gate
# in scripts/run_conductance.jl, NOT assumed.
#
# Layer convention (as in moving.jl / pathint.jl): MODEL/MEASUREMENT only — no Plots.
# field.jl and ring.jl are NOT modified, so every Phase 1–3 result stays bit-identical;
# the current-based model is recovered as the E_syn→∞, G→0 limit (note §1).
#
# The `shunt` keyword (default true) toggles the new (G/2)(z²−1) / −G·sinθ term OFF —
# used by the run script's debug staircase: with shunt=false and E_E=1, E_I=−1 one has
# E_E·K_E + E_I·K_I = K and no shunting, so the conductance code must reproduce the
# current-based bump exactly, isolating the reversal-weighting from the new term.
# ============================================================================

include("pathint.jl")   # → moving.jl → field.jl + ring.jl (single include chain). Brings the
                        # core (coupling_field, coupling, rate, meanpulse, pulse, make_kernel,
                        # field_kernel, rest_state, angular_distance, thetadot, ring_smooth,
                        # bump_centroid, unwrap, lateral_speed, …) AND the Phase-3 command
                        # library + PI readouts reused by Step 4.2 (omega_*, Bfun_from_omega,
                        # heading_estimate, commanded_heading, instantaneous_speed).

# --- Dale-split synapse configuration -----------------------------------------
# Precompute the per-side kernels once; both structs are tiny (the field stores two
# FFT transforms, the ring two rank-3 RingKernels). E_E/E_I are the reversal potentials.
# `g0` is the CONDUCTANCE GAIN — the new physics knob (note §1). The conductances are
# g_E = g0·∫K_E·P, g_I = g0·∫K_I·P, so g0 sets the size of the shunt term G=g_E+g_I.
# The current-based model is the EXACT limit g0→0 with the reversals sent to ±κ/g0→±∞
# (E_syn→∞, g→0, the product g·E_syn = current held fixed): then the drive
# J = E_E g_E + E_I g_I → κ∫(K_E−K_I)P = κI stays put while the shunt G·g0 → 0. So a
# SMALL g0 with `matched_reversals` is a weak-shunt perturbation of the validated bump;
# turning g0 up introduces shunting. (κ is no longer a separate coupling multiplier —
# the current scale is carried by E_E·g0 / E_I·g0; κ survives only as the kick gain.)
struct FieldSyn
    KhatE::Vector{ComplexF64}     # fft(K_E) for the FFT convolution in coupling_field
    KhatI::Vector{ComplexF64}     # fft(K_I)
    g0::Float64                   # conductance gain (shunt strength)
    E_E::Float64
    E_I::Float64
end

struct RingSyn
    KE::RingKernel                # rank-3 excitatory kernel (carries cosx/sinx of positions)
    KI::RingKernel                # rank-3 inhibitory kernel (flat ⇒ a1=B=0; structured ⇒ a1=aI>0)
    g0::Float64                   # conductance gain (shunt strength)
    E_E::Float64
    E_I::Float64
end

# Build the field/ring Dale configs (note §2 constants). B rides on the excitatory
# channel (the velocity-shift); with the defaults K_E − K_I = 0.1 + 0.3cos + B sin = the
# original kernel. `aI` is the inhibitory cosine structure (default 0.0 ⇒ flat K_I, the
# validated Phase-4 model, bit-identical); set aI>0 for the divisive structured-inhibition
# extension, and set aE=0.3+aI at the call site to hold the net kernel fixed (see note above).
function dale_field(x, E_E, E_I; g0=1.0, B=0.0, cE=0.4, aE=0.3, cI=0.3, aI=0.0)
    KhatE = fft(cE .+ aE .* cos.(x) .+ B .* sin.(x))
    KhatI = fft(cI .+ aI .* cos.(x))
    return FieldSyn(KhatE, KhatI, g0, E_E, E_I)
end

function dale_ring(x, E_E, E_I; g0=1.0, B=0.0, cE=0.4, aE=0.3, cI=0.3, aI=0.0)
    KE = make_kernel(x; a0=cE, a1=aE, B=B)
    KI = make_kernel(x; a0=cI, a1=aI, B=0.0)
    return RingSyn(KE, KI, g0, E_E, E_I)
end

# Reversals that hold the drive J equal to the current-based κI at a given shunt gain
# g0 (so g0→0 ⇒ reversals→±∞ ⇒ exact current-based limit). Use as
# `E_E, E_I = matched_reversals(κ, g0)` to start a re-tune near the validated bump.
matched_reversals(κ, g0) = (κ / g0, -κ / g0)

# ============================================================================
# Field side (macroscopic). Mirrors field_rhs/field_step; reuses coupling_field
# (which already carries the 2π/N integration measure — κ multiplies it to give the
# conductance). Iext enters as κ·Iext, matching the current-based field's kick scaling.
# ============================================================================

function field_rhs_cond(z, η̄, Δ, κ, syn::FieldSyn; Iext=0.0, shunt::Bool=true)
    gE = syn.g0 .* coupling_field(z, syn.KhatE)            # excitatory conductance ≥ 0
    gI = syn.g0 .* coupling_field(z, syn.KhatI)            # inhibitory conductance ≥ 0
    f  = η̄ .+ syn.E_E .* gE .+ syn.E_I .* gI .+ κ .* Iext  # η̄ + J (+ external kick)
    ż  = (im/2) .* (z .- 1).^2 .- 0.5 .* (z .+ 1).^2 .* (Δ .+ im .* f)
    G  = gE .+ gI                                          # total conductance
    return shunt ? ż .+ 0.5 .* G .* (z.^2 .- 1) : ż        # + (G/2)(z²−1) shunting term
end

function field_step_cond(z, η̄, Δ, κ, syn, dt; Iext=0.0, shunt::Bool=true)
    k1 = field_rhs_cond(z,                η̄, Δ, κ, syn; Iext=Iext, shunt=shunt)
    k2 = field_rhs_cond(z .+ 0.5dt .* k1, η̄, Δ, κ, syn; Iext=Iext, shunt=shunt)
    k3 = field_rhs_cond(z .+ 0.5dt .* k2, η̄, Δ, κ, syn; Iext=Iext, shunt=shunt)
    k4 = field_rhs_cond(z .+ dt   .* k3,  η̄, Δ, κ, syn; Iext=Iext, shunt=shunt)
    return z .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

# Evolve the field, returning ONLY the final z (O(Nx) memory — used by seeding).
function evolve_field_cond(z0, η̄, Δ, κ, syn; T=200.0, dt=0.02, Iext=0.0, shunt::Bool=true)
    nt = round(Int, T/dt)
    z  = copy(z0)
    for _ in 1:nt
        z = field_step_cond(z, η̄, Δ, κ, syn, dt; Iext=Iext, shunt=shunt)
    end
    return z
end

# ============================================================================
# Ring side (microscopic). Mirrors population_rhs/rk4_increment/step_population.
# thetadot is reused for the (1∓cosθ) intrinsic+drive part; the only addition is the
# −G·sinθ shunting term. coupling carries the 2π/N measure; κ gives the conductance.
# ============================================================================

function population_rhs_cond(θ, η, syn::RingSyn, a_n, n, κ; Iext=0.0, shunt::Bool=true)
    P    = pulse(θ, a_n, n)
    gE   = syn.g0 .* coupling(P, syn.KE)                   # excitatory conductance ≥ 0
    gI   = syn.g0 .* coupling(P, syn.KI)                   # inhibitory conductance ≥ 0
    f    = η .+ syn.E_E .* gE .+ syn.E_I .* gI .+ κ .* Iext
    base = thetadot.(θ, f)                                 # (1−cosθ) + (1+cosθ)(η + J)
    return shunt ? base .- (gE .+ gI) .* sin.(θ) : base    # − G·sinθ shunting term
end

function rk4_increment_cond(θ, η, syn, a_n, n, κ, dt; Iext=0.0, shunt::Bool=true)
    k1 = population_rhs_cond(θ,                η, syn, a_n, n, κ; Iext=Iext, shunt=shunt)
    k2 = population_rhs_cond(θ .+ 0.5dt .* k1, η, syn, a_n, n, κ; Iext=Iext, shunt=shunt)
    k3 = population_rhs_cond(θ .+ 0.5dt .* k2, η, syn, a_n, n, κ; Iext=Iext, shunt=shunt)
    k4 = population_rhs_cond(θ .+ dt   .* k3,  η, syn, a_n, n, κ; Iext=Iext, shunt=shunt)
    return (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function step_population_cond(θ, η, syn, a_n, n, κ, dt; Iext=0.0, shunt::Bool=true)
    return mod.(θ .+ rk4_increment_cond(θ, η, syn, a_n, n, κ, dt; Iext=Iext, shunt=shunt), 2π)
end

function evolve_population_cond(θ0, η, syn, a_n, n, κ; T=200.0, dt=0.02, Iext=0.0, shunt::Bool=true)
    nt = round(Int, T/dt)
    θ  = copy(θ0)
    for _ in 1:nt
        θ = step_population_cond(θ, η, syn, a_n, n, κ, dt; Iext=Iext, shunt=shunt)
    end
    return θ
end

# ============================================================================
# Seeding (mirror seed_field_bump/seed_ring_bump from moving.jl, under conductance):
# relax to rest → transient localized Gaussian kick to carve a bump → release/settle.
# ============================================================================

function seed_field_bump_cond(x, η̄, Δ, κ, syn;
                              A=0.6, σ=0.6, x0=π, T_kick=20.0, T_free=200.0, dt=0.02, shunt::Bool=true)
    z0   = fill(rest_state(η̄, Δ), length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    zk = evolve_field_cond(z0, η̄, Δ, κ, syn; T=T_kick, dt=dt, Iext=kick, shunt=shunt)  # carve
    zf = evolve_field_cond(zk, η̄, Δ, κ, syn; T=T_free, dt=dt,           shunt=shunt)  # settle
    return zf
end

function seed_ring_bump_cond(η, syn, a_n, n, κ, x;
                             A=0.6, σ=0.6, x0=π, T_relax=30.0, T_kick=20.0, T_free=200.0, dt=0.02, shunt::Bool=true)
    θ0   = zeros(length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    θr = evolve_population_cond(θ0, η, syn, a_n, n, κ; T=T_relax, dt=dt,           shunt=shunt)  # → rest
    θk = evolve_population_cond(θr, η, syn, a_n, n, κ; T=T_kick,  dt=dt, Iext=kick, shunt=shunt)  # carve
    θf = evolve_population_cond(θk, η, syn, a_n, n, κ; T=T_free,  dt=dt,           shunt=shunt)  # settle
    return θf
end

# ============================================================================
# Conductance micro readouts (mirror mean_frequencies/mean_order_parameter): the
# spiking 3-panel for the micro/macro gate, measured directly from the neurons under
# the conductance dynamics. Run on a SETTLED state (after seed_ring_bump_cond).
# ============================================================================

# Mean firing frequency f_k = ⟨dθ/dt⟩_t / 2π (Laing Eq. 10) — micro counterpart of rate(z).
function mean_frequencies_cond(θ0, η, syn, a_n, n, κ; T=200.0, dt=0.02, shunt::Bool=true)
    nt = round(Int, T/dt)
    θ  = copy(θ0)
    advance = zeros(length(θ0))
    for _ in 1:nt
        inc = rk4_increment_cond(θ, η, syn, a_n, n, κ, dt; shunt=shunt)
        advance .+= inc
        θ = mod.(θ .+ inc, 2π)
    end
    return advance ./ (2π * T)
end

# Time-averaged order parameter z̄_k = ⟨e^{-iθ}⟩_t — micro counterpart of the field z.
function mean_order_parameter_cond(θ0, η, syn, a_n, n, κ; T=200.0, dt=0.02, shunt::Bool=true)
    nt   = round(Int, T/dt)
    θ    = copy(θ0)
    zacc = zeros(ComplexF64, length(θ0))
    for _ in 1:nt
        zacc .+= exp.(-im .* θ)
        θ = step_population_cond(θ, η, syn, a_n, n, κ, dt; shunt=shunt)
    end
    return zacc ./ nt
end

# ============================================================================
# Moving-bump centroid tracking (mirror track_field_centroid/track_ring_centroid):
# evolve under a fixed asymmetric B (baked into syn) and record only the bump centre,
# for the Step-4.1 moving-bump speed check via lateral_speed.
# ============================================================================

function track_field_centroid_cond(z0, η̄, Δ, κ, syn, cosx, sinx; T=200.0, dt=0.02)
    nt = round(Int, T/dt)
    z  = copy(z0)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        z = field_step_cond(z, η̄, Δ, κ, syn, dt)
        xc[i] = bump_centroid(rate.(z), cosx, sinx)
    end
    return xc, z
end

function track_ring_centroid_cond(θ0, η, syn::RingSyn, a_n, n, κ; T=200.0, dt=0.02)
    nt = round(Int, T/dt)
    θ  = copy(θ0)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        θ = step_population_cond(θ, η, syn, a_n, n, κ, dt)
        xc[i] = bump_centroid(pulse(θ, a_n, n), syn.KE.cosx, syn.KE.sinx)
    end
    return xc, θ
end

# ============================================================================
# STEP 4.2 — PI gain under conductance (project.md §15, Step 4.2).
#
# Two pieces, both thin wrappers over the conductance RHS above + the Phase-2/3
# machinery (now reachable via the pathint.jl include): (A) warm-started s(B) sweeps
# for the quasi-static gain curve k(Ω)=s(βΩ)/Ω, and (B) time-varying-B(t) steppers
# for the dynamic path-integration demo (heading φ tracks ∫Ω dt). B always rides the
# EXCITATORY channel K_E (note §2): the field's antisymmetric transform adds linearly
# (Khat_E(t)=Khat_E0 + B(t)·Shat), the ring rebuilds only K_E's rank-3 coefficients.
# ============================================================================

# --- (A) Warm-started conductance B-sweeps → s(B) (mirror bsweep_field/bsweep_ring) -
# For each B (in order) rebuild the Dale config at that B and evolve from the previous
# B's settled state; keep only the scalar speed. ASCENDING Bs = forward sweep,
# DESCENDING = backward (exposes any hysteresis). Returns (speeds, final state).
function bsweep_field_cond(Bs, x, η̄, Δ, κ, E_E, E_I, g0, z_start; T=100.0, dt=0.02, frac=0.5, tag="field", every=0)
    cosx, sinx = cos.(x), sin.(x)
    s = Vector{Float64}(undef, length(Bs))
    z = copy(z_start)
    for (i, B) in enumerate(Bs)
        syn = dale_field(x, E_E, E_I; g0=g0, B=B)
        xc, z = track_field_centroid_cond(z, η̄, Δ, κ, syn, cosx, sinx; T=T, dt=dt)
        s[i] = lateral_speed(xc, dt; frac=frac)
        if every > 0 && (i % every == 0 || i == length(Bs))
            println("  [$tag] $i/$(length(Bs))  B=", round(B, digits=4), "  s=", round(s[i], digits=4)); flush(stdout)
        end
    end
    return s, z
end

function bsweep_ring_cond(Bs, x, η, a_n, n, κ, E_E, E_I, g0, θ_start; T=100.0, dt=0.02, frac=0.5, tag="ring", every=0)
    s = Vector{Float64}(undef, length(Bs))
    θ = copy(θ_start)
    for (i, B) in enumerate(Bs)
        syn = dale_ring(x, E_E, E_I; g0=g0, B=B)
        xc, θ = track_ring_centroid_cond(θ, η, syn, a_n, n, κ; T=T, dt=dt)
        s[i] = lateral_speed(xc, dt; frac=frac)
        if every > 0 && (i % every == 0 || i == length(Bs))
            println("  [$tag] $i/$(length(Bs))  B=", round(B, digits=4), "  s=", round(s[i], digits=4)); flush(stdout)
        end
    end
    return s, θ
end

# --- (B) Time-varying-B(t) conductance steppers (mirror field_step_tv/step_population_tv) -
# The system is NON-AUTONOMOUS under B(t), so each RK4 stage uses the kernel at its own
# stage time (t, t+dt/2, t+dt/2, t+dt) — the explicit-time extension of the per-stage
# coupling rule. `syn0` is the Dale config built at B=0; only K_E's asymmetric part moves.
function field_step_cond_tv(z, t, η̄, Δ, κ, syn0::FieldSyn, Shat, Bfun, dt; Iext=0.0, shunt::Bool=true)
    synt(tt) = FieldSyn(syn0.KhatE .+ Bfun(tt) .* Shat, syn0.KhatI, syn0.g0, syn0.E_E, syn0.E_I)
    k1 = field_rhs_cond(z,              η̄, Δ, κ, synt(t);          Iext=Iext, shunt=shunt)
    k2 = field_rhs_cond(z .+ 0.5dt.*k1, η̄, Δ, κ, synt(t + 0.5dt); Iext=Iext, shunt=shunt)
    k3 = field_rhs_cond(z .+ 0.5dt.*k2, η̄, Δ, κ, synt(t + 0.5dt); Iext=Iext, shunt=shunt)
    k4 = field_rhs_cond(z .+ dt  .*k3,  η̄, Δ, κ, synt(t + dt);    Iext=Iext, shunt=shunt)
    return z .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

function step_population_cond_tv(θ, t, η, syn0::RingSyn, a_n, n, κ, Bfun, dt; Iext=0.0, shunt::Bool=true)
    synt(tt) = RingSyn(RingKernel(syn0.KE.cosx, syn0.KE.sinx, syn0.KE.a0, syn0.KE.a1, Bfun(tt)),
                       syn0.KI, syn0.g0, syn0.E_E, syn0.E_I)
    k1 = population_rhs_cond(θ,              η, synt(t),          a_n, n, κ; Iext=Iext, shunt=shunt)
    k2 = population_rhs_cond(θ .+ 0.5dt.*k1, η, synt(t + 0.5dt),  a_n, n, κ; Iext=Iext, shunt=shunt)
    k3 = population_rhs_cond(θ .+ 0.5dt.*k2, η, synt(t + 0.5dt),  a_n, n, κ; Iext=Iext, shunt=shunt)
    k4 = population_rhs_cond(θ .+ dt  .*k3,  η, synt(t + dt),     a_n, n, κ; Iext=Iext, shunt=shunt)
    return mod.(θ .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4), 2π)
end

# --- Drive under B(t), recording the bump centroid each step (mirror drive_field/drive_ring) -
# syn0 = Dale config at B=0; Bfun = kernel-level command t↦B(t) (compose Bfun_from_omega(Ω, β)).
function drive_field_cond(z0, η̄, Δ, κ, x, syn0::FieldSyn, Bfun; T=200.0, dt=0.02)
    Shat = fft(sin.(x))
    cosx, sinx = cos.(x), sin.(x)
    nt = round(Int, T/dt)
    z  = copy(z0)
    ts = Vector{Float64}(undef, nt)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        t = (i - 1) * dt
        z = field_step_cond_tv(z, t, η̄, Δ, κ, syn0, Shat, Bfun, dt)
        ts[i] = i * dt
        xc[i] = bump_centroid(rate.(z), cosx, sinx)
    end
    return ts, xc, z
end

function drive_ring_cond(θ0, η, x, syn0::RingSyn, a_n, n, κ, Bfun; T=200.0, dt=0.02)
    nt = round(Int, T/dt)
    θ  = copy(θ0)
    ts = Vector{Float64}(undef, nt)
    xc = Vector{Float64}(undef, nt)
    for i in 1:nt
        t = (i - 1) * dt
        θ = step_population_cond_tv(θ, t, η, syn0, a_n, n, κ, Bfun, dt)
        ts[i] = i * dt
        xc[i] = bump_centroid(pulse(θ, a_n, n), syn0.KE.cosx, syn0.KE.sinx)
    end
    return ts, xc, θ
end
