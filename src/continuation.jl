# ============================================================================
# Pseudo-arclength continuation of MOVING-BUMP (travelling-wave) solutions of the
# macroscopic field — the SOLID CURVE of Laing & Omel'chenko Fig. 3 (field travelling-
# wave speed s vs asymmetry B; that solid curve is their Fig. 13, Δ=0.1). A direct-
# simulation B-sweep can only sample *stable* arcs with markers and jumps across folds;
# continuation instead Newton-solves the travelling-wave equation directly and follows a
# branch through its turning points, drawing each branch as a continuous line.
#
# WHY PSEUDO-ARCLENGTH (and not natural-parameter continuation). A travelling-wave branch
# folds back in B (and at Δ=0.1 the lower branch has such a fold near B≈0.1, s≈0.5).
# Continuation in a single control parameter cannot round a turning point in that same
# parameter: stepping B fails at a B-fold, stepping s fails at an s-fold. Pseudo-arclength
# removes this by parametrising the branch by its own arclength — it augments the (a, s, B)
# system with the constraint
#
#     ⟨u − u_prev, τ⟩_w − Δarc = 0 ,                                   (Keller)
#
# where τ is the unit branch tangent (the null vector of the core Jacobian), Δarc the step,
# and ⟨·,·⟩_w a weighted inner product that paces the march by the physical (s,B) (see
# warc_dot). The augmented Jacobian [J_core; ∇constraint′] stays nonsingular AT a fold
# (that is the whole point), so each branch is traced as one continuous arc through its
# folds. This is the fix over the previous march-in-s version, which stalled at folds and
# broke a branch into pieces.
#
# WHAT IS AND IS NOT CONTINUOUS. At Δ=0.1 there are TWO travelling-wave branches — both
# leave the origin and run to large B with similar speeds (s→~1.27), Laing's Fig. 12/13
# scenario. They are genuinely SEPARATE within the B∈[0,0.2] plot window (verified: the
# lower branch does not connect to the upper one at finite B in-window). So tw_continuation
# uses ONE seed per branch and NaN-joins them: each branch is continuous (the win here),
# but the gap BETWEEN the two branches is real physics — the same coexistence the forward/
# backward marker hysteresis shows — not a continuation artifact.
#
# THE EQUATION. A right-moving wave z(x,t) = a(x − s t) is a steady state of the
# co-moving field (substitute into field.jl's ż = F(z); since ∂_t z = −s a′):
#
#     G(a, s) ≡ F(a) + s·a′(x) = 0 ,
#     F(a) = (i/2)(a−1)² − (1/2)(a+1)²·[Δ + i(η̄ + κ·I[a])]            (Laing Eq. 12)
#
# with I[a] the SAME kernel convolution field.jl uses (so the macro model is
# untouched — this is just its steady travelling-wave form). a′ is computed
# spectrally. Translational invariance (a → a(·+c) is also a solution) is removed
# by the pinning condition
#
#     ∫₀^{2π} Re[a(x)] cos x dx = 0 .                                  (Laing Eq. 13)
#
# Unknowns: the complex profile a(x) on N grid points plus the speed s and asymmetry B
# (packed u = [Re a; Im a; s; B], length 2N+2). res_core gives the 2N+1 physics+pinning
# residual — one short of square, leaving the 1-D branch; the arclength constraint
# closes it. Newton with a central-difference numerical Jacobian (numjac) corrects each
# step; the tangent predictor and arclength corrector (tw_tangent / arclength_corrector)
# are the only pieces beyond the seed machinery (initial_wave / seed_branch).
#
# DEPENDENCIES. Uses field.jl symbols (coupling_field, field_kernel, field_positions)
# and moving.jl symbols (seed_field_bump, track_field_centroid, lateral_speed) that
# are already in scope when this file is included after moving.jl (as run_sweep.jl
# does). No Plots. This file carries the only LinearAlgebra dependency in src/.
# ============================================================================

using LinearAlgebra

# Integer wavenumbers m for the spectral derivative on [0, 2π) with N points: a
# function a(x) = Σ_m c_m e^{imx} has a′ = Σ_m (im) c_m e^{imx}, i.e. multiply the
# FFT by i·m. Ordering matches FFTW: 0,1,…,N/2−1, −N/2,…,−1.
wavenumbers(N) = Float64.(vcat(0:(N ÷ 2 - 1), -(N ÷ 2):-1))

# Spatial derivative a′(x) via FFT (periodic, spectrally accurate).
deriv_x(a, kvec) = ifft(im .* kvec .* fft(a))

# Travelling-wave residual G = F(a) + s·a′ (N complex numbers; zero on a wave).
function tw_residual(a, s, B, η̄, Δ, κ, kvec, x)
    Khat = fft(field_kernel(x; B=B))            # kernel transform (cheap; B-dependent)
    I = coupling_field(a, Khat)                 # (2π/N) Σ K·meanpulse(a) — same as field.jl
    f = η̄ .+ κ .* I
    F = (im / 2) .* (a .- 1).^2 .- 0.5 .* (a .+ 1).^2 .* (Δ .+ im .* f)
    return F .+ s .* deriv_x(a, kvec)
end

# Reconstruct the complex profile a from the packed real unknown vector
# u = [Re a (N); Im a (N); s; B].
geta(u, N) = complex.(u[1:N], u[N+1:2N])

# Core residual [Re G; Im G; pinning] (length 2N+1), as a function of the full
# packed u (length 2N+2). This is the physics + phase condition; the arclength row
# is appended separately by the continuation corrector.
function res_core(u, N, η̄, Δ, κ, kvec, x, cosx)
    a = geta(u, N); s = u[2N + 1]; B = u[2N + 2]
    G = tw_residual(a, s, B, η̄, Δ, κ, kvec, x)
    pin = sum(real.(a) .* cosx) * (2π / N)      # ∫ Re a cos x dx (discretised)
    return vcat(real.(G), imag.(G), pin)
end

# Central-difference Jacobian of a residual closure res(u). For this problem the
# system is ~2N+2 ≈ 514-dimensional at N=256, so a numerical Jacobian is simplest;
# CENTRAL differences (2n residual evals, each a handful of length-256 FFTs) are
# worth the 2× cost over forward differences because the O(h²) truncation error
# pushes the Jacobian accuracy to ~1e-10, which is what lets Newton drive the
# 514-component residual NORM below tolerance (forward differences floor out around
# a norm of ~2e-6 here — the sqrt(514) amplification of the per-component ~1e-7 floor
# — which silently stalls the arclength corrector). r0 is unused but kept for a
# uniform call signature with the forward-difference form.
function numjac(res, u, r0; ε=1e-6)
    m = length(r0); n = length(u)
    J = Matrix{Float64}(undef, m, n)
    u = copy(u)
    @inbounds for j in 1:n
        h = ε * (abs(u[j]) + 1.0)
        uj = u[j]
        u[j] = uj + h; rp = res(u)
        u[j] = uj - h; rm = res(u)
        J[:, j] = (rp .- rm) ./ (2h)
        u[j] = uj
    end
    return J
end

# Newton's method for a SQUARE residual system res(u) = 0. With the central-difference
# numjac (Jacobian accuracy ~1e-10) the 514-component residual norm reaches ~2e-8, so
# tol=1e-7 is comfortably attainable while still demanding a genuine solution.
function newton(res, u0; tol=1e-7, maxit=30)
    u = copy(u0)
    for _ in 1:maxit
        r = res(u)
        norm(r) < tol && return u, true
        J = numjac(res, u, r)
        u = u .- (J \ r)
    end
    return u, norm(res(u)) < tol
end

# Converge an initial travelling wave at FIXED B from a direct-simulation guess
# (profile a, speed s). Returns the full packed u (with B appended) and a success
# flag. Here the unknowns are just [Re a; Im a; s] (2N+1) and B is held constant,
# so the system is square and Newton lands exactly on the branch.
function initial_wave(aguess, sguess, B, N, η̄, Δ, κ, kvec, x, cosx)
    # PRE-ALIGN to ~satisfy pinning. The direct-sim snapshot is an excellent wave
    # (residual of the physics G ≈ 0) but sits at an arbitrary phase, so its pinning
    # term ∫Re a cos x dx is O(1). Zeroing it requires translating the bump — a large,
    # highly nonlinear move that makes Newton diverge if left to it. A translate is
    # itself a wave (same s), so we first circularly shift the profile to the integer
    # grid offset that minimises |pinning|; Newton then only has to polish it.
    pin_of(sh) = abs(sum(real.(circshift(aguess, sh)) .* cosx))
    sh = argmin(pin_of.(0:N-1)) - 1
    aguess = circshift(aguess, sh)

    res(v) = res_core(vcat(v, B), N, η̄, Δ, κ, kvec, x, cosx)   # hold B fixed
    v0 = vcat(real.(aguess), imag.(aguess), sguess)
    v, ok = newton(res, v0)
    return vcat(v, B), ok
end

# --- Weighted arclength inner product (paces the march by the (s,B) parameters) ---
# The branch lives in the 2N+2 vector u = [Re a; Im a; s; B], but only the last two
# components are the physical (s, B) we want to plot. Measuring arclength in the full
# space makes the unit tangent DOMINATED by the 512 profile components, so each Δarc
# step barely advances (s, B) and the branch needs thousands of tiny steps. We instead
# DOWN-WEIGHT the profile by wprof≪1 in the arclength inner product ⟨p,q⟩_w, so Δarc is
# (to leading order) a step in the physical (s,B) arclength. The profile is kept in the
# product (not dropped) so the augmented Jacobian stays nonsingular at a fold.
warc_dot(p, q, N, wprof) = wprof * dot(p[1:2N], q[1:2N]) + dot(p[2N+1:2N+2], q[2N+1:2N+2])
warc_grad(τ, N, wprof)   = vcat(wprof .* τ[1:2N], τ[2N+1:2N+2])   # ∇_u ⟨u−u_prev, τ⟩_w

# --- Branch tangent τ: the (weighted-)unit null vector of the core Jacobian -------
# res_core gives 2N+1 equations in the 2N+2 unknowns u, so its Jacobian J (2N+1 × 2N+2)
# has a 1-D null space — the branch direction. We pick the tangent by appending one row
# (the weighted previous tangent) to make the system square and solving
#     [J; warc_grad(τ_prev)′] τ = [0…0; 1]   (so J τ = 0),
# then normalising τ in the WEIGHTED norm. Flipping the sign when ⟨τ,τ_prev⟩_w < 0 keeps
# the marching direction continuous. `fcore` is the closure u ↦ res_core(u, …). This is
# the predictor's direction and the corrector's constraint normal.
function tw_tangent(fcore, u, τ_prev, N, wprof)
    J = numjac(fcore, u, fcore(u))                  # 2N+1 × 2N+2 core Jacobian
    τ = vcat(J, reshape(warc_grad(τ_prev, N, wprof), 1, :)) \ vcat(zeros(length(u) - 1), 1.0)
    τ ./= sqrt(warc_dot(τ, τ, N, wprof))
    return warc_dot(τ, τ_prev, N, wprof) < 0 ? -τ : τ
end

# First tangent at the seed, where there is no previous direction. Same null-vector
# solve but with the normalising row = the unit vector on the SPEED component (index
# 2N+1), which is well-conditioned on the lower branch (ds/darc ≠ 0 there). The sign is
# fixed so the speed increases — i.e. the forward march heads UP the branch toward the
# knot; the backward march (−τ0) heads down toward the origin.
function initial_tangent(fcore, u, N, wprof)
    J = numjac(fcore, u, fcore(u))
    e_s = zeros(length(u)); e_s[2N + 1] = 1.0       # select the s-component
    τ = vcat(J, reshape(e_s, 1, :)) \ vcat(zeros(length(u) - 1), 1.0)
    τ ./= sqrt(warc_dot(τ, τ, N, wprof))
    return τ[2N + 1] < 0 ? -τ : τ                   # march toward increasing s
end

# --- Pseudo-arclength corrector: Newton on the AUGMENTED square system ------------
# Given a predictor u_pred, the previous on-branch point u_prev, the tangent τ, and the
# step Δarc, solve the 2N+2 square system
#     R(u) = [ res_core(u) ;  ⟨u − u_prev, τ⟩_w − Δarc ] = 0 ,   A(u) = [ J_core(u) ; ∇g′ ]
# by Newton, where g is the (weighted) arclength constraint and ∇g = warc_grad(τ). The
# bottom row fixes the arclength advance along τ; A stays NONSINGULAR at a fold (where
# J_core alone drops rank), which is exactly why this rounds turning points in both s and
# B. Returns (u, converged?).
function arclength_corrector(fcore, u_pred, u_prev, τ, Δarc, N, wprof; tol=1e-7, maxit=20)
    grow = warc_grad(τ, N, wprof)                   # constant row (τ fixed during corrector)
    u = copy(u_pred)
    for _ in 1:maxit
        rc = fcore(u)
        R  = vcat(rc, warc_dot(u .- u_prev, τ, N, wprof) - Δarc)
        norm(R) < tol && return u, true
        A = vcat(numjac(fcore, u, rc), reshape(grow, 1, :))
        u = u .- (A \ R)
    end
    R = vcat(fcore(u), warc_dot(u .- u_prev, τ, N, wprof) - Δarc)
    return u, norm(R) < tol
end

# --- March one branch from a seed in ONE direction by pseudo-arclength ------------
# Predictor–corrector loop: predict u + Δarc·τ along the tangent, correct back onto the
# branch, recompute the tangent for the next step. The step Δarc is HALVED (up to
# `ntry` times) whenever the corrector fails — automatic refinement through the tight
# turns at folds (this is what lets it ROUND a fold in B, e.g. the knot at Δ=0.1, where
# the old march-in-s stalled). Stops when: the corrector fails even at the smallest step,
# the branch leaves the [Blo,Bhi]×[slo,shi] window (Bhi = the 0.2 plot edge, so the in-
# window arc is captured and the run ends as the branch heads off to large B), `max_steps`
# is hit, or it returns near the seed (a closed loop). Records (B, s) per accepted point
# (the seed is the first entry) and a `closed` flag. `fcore = u ↦ res_core(u, …)`.
function arclength_branch(fcore, u0, τ0, N, wprof; Δarc=0.02, max_steps=2000, ntry=6,
                          Blo=-0.02, Bhi=0.21, slo=-0.05, shi=1.4, closetol=1e-3)
    Bs = [u0[2N + 2]]; ss = [u0[2N + 1]]
    u = copy(u0); τ = copy(τ0); closed = false
    for step in 1:max_steps
        unew = u; ok = false; h = Δarc
        for _ in 1:ntry                              # adaptive: shrink the step on failure
            unew, ok = arclength_corrector(fcore, u .+ h .* τ, u, τ, h, N, wprof)
            ok && break
            h /= 2
        end
        ok || break                                  # corrector gave up → branch end
        B = unew[2N + 2]; s = unew[2N + 1]
        push!(Bs, B); push!(ss, s)
        (B < Blo || B > Bhi || s < slo || s > shi) && break   # left the physical window
        if step > 5 && hypot(B - u0[2N + 2], s - u0[2N + 1]) < closetol
            closed = true; break                     # came back to the seed → closed loop
        end
        τ = tw_tangent(fcore, unew, τ, N, wprof)     # tangent for the next predictor
        u = unew
    end
    return Bs, ss, closed
end

# --- Trace one whole branch (both directions from its seed) -----------------------
# A seed sits in the middle of its branch; we march FORWARD (+τ0, increasing s) and, if
# that march did not already close the loop, BACKWARD (−τ0) too, then stitch them into
# one ordered polyline reverse(backward)→seed→forward. Returns (B, s) for the branch.
function trace_component(fcore, u0, τ0, N, wprof; kw...)
    Bf, sf, closed = arclength_branch(fcore, u0,  τ0, N, wprof; kw...)
    closed && return Bf, sf                          # closed loop already fully traced
    Bb, sb, _ = arclength_branch(fcore, u0, -τ0, N, wprof; kw...)
    return vcat(reverse(Bb)[1:end-1], Bf), vcat(reverse(sb)[1:end-1], sf)  # drop duplicated seed
end

# Seed a branch by ramping B from 0 up to Btarget (warm-started, as the marker sweep
# does — this lands on the moving branch reachable from the static bump), measuring
# the speed, and Newton-polishing onto the travelling-wave branch. Returns (a, s, B).
function seed_branch(Btarget, N, η̄, Δ, κ, kvec, x, cosx, sinx; dt=0.02)
    zstat = seed_field_bump(x, η̄, Δ, κ, fft(field_kernel(x; B=0.0)); dt=dt)     # B=0 static bump
    _, z  = bsweep_field(collect(0.0:0.005:Btarget), x, η̄, Δ, κ, zstat; T=120.0, dt=dt, frac=0.5)
    KhatT = fft(field_kernel(x; B=Btarget))
    xc, z = track_field_centroid(z, η̄, Δ, κ, KhatT, cosx, sinx; T=150.0, dt=dt)
    s0    = lateral_speed(xc, dt; frac=0.5)
    u0, ok = initial_wave(z, s0, Btarget, N, η̄, Δ, κ, kvec, x, cosx)
    ok || @warn "seed_branch(B=$Btarget): initial Newton did not converge"
    return geta(u0, N), u0[2N + 1], u0[2N + 2]
end

# Top-level: trace the field travelling-wave curve s(B) of Laing Fig. 3, each branch
# CONTINUOUS through its folds, by pseudo-arclength continuation. At Δ=0.1 the system has
# TWO travelling-wave branches that both leave the origin and run off to large B with
# similar speeds (s→~1.27) — Laing's Fig. 12/13 scenario ("two branches ... with similar
# speeds for large B"). They are genuinely SEPARATE within the B∈[0,0.2] plot window, so
# we use one seed per branch (each found by the direct-sim ramp, which jumps to the locally
# stable branch — seed_branch):
#   • LOWER branch — seed_branch(0.05) lands at (0.05, ~0.20); trace_component rounds its
#     fold near (B≈0.1, s≈0.5) and follows it up to the B=0.2 edge and down to the origin;
#   • UPPER branch — seed_branch(0.20) lands at (0.2, ~1.24); trace_component follows it
#     across the window.
# The two are NaN-joined (a GENUINE gap between two real branches, not a method artifact —
# the markers' forward/backward hysteresis is exactly this coexistence). What pseudo-
# arclength fixes versus the old march-in-s is the breaks WITHIN each branch: the fold in B
# is now rounded, so each branch is a single continuous arc, matching Laing's Fig. 3.
# Returns (B, s). Further disjoint isolas (if any) would need their own seeds — out of scope.
function tw_continuation(N, η̄, Δ, κ; dt=0.02, Δarc=0.02, wprof=0.01, max_steps=2000)
    x = field_positions(N); cosx = cos.(x); sinx = sin.(x); kvec = wavenumbers(N)
    fcore(u) = res_core(u, N, η̄, Δ, κ, kvec, x, cosx)

    function branch(Bseed)
        a0, s0, B0 = seed_branch(Bseed, N, η̄, Δ, κ, kvec, x, cosx, sinx; dt=dt)
        u0 = vcat(real.(a0), imag.(a0), s0, B0)
        τ0 = initial_tangent(fcore, u0, N, wprof)
        return trace_component(fcore, u0, τ0, N, wprof; Δarc=Δarc, max_steps=max_steps)
    end

    Blo, slo = branch(0.05)        # lower branch (origin → fold → B=0.2 edge)
    Bup, sup = branch(0.20)        # upper branch (fast, s≈1.24 at B=0.2)

    B = vcat(Blo, NaN, Bup)        # NaN = genuine gap between two coexisting branches
    s = vcat(slo, NaN, sup)
    return B, s
end
