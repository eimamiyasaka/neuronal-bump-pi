# ============================================================================
# Phase-5, Step 5.2 — the Nakao (2014) adjoint / phase-sensitivity drift law for
# the next-generation bump. A READOUT/ANALYSIS layer on top of the unchanged
# conductance field (like phase5.jl); the only new dependency is LinearAlgebra
# (already used in src/continuation.jl). No Plots. Closes the item DEFERRED in
# step5_gate.md: "Nakao adjoint analytic drift — the per-realization drift law for
# heavy-tailed disorder."
#
# WHY AN ADJOINT. A static bump z₀(x) of the translation-invariant field is one
# member of a continuous family z₀(x−φ) (the Goldstone/translation mode). The field
# linearization L = ∂F/∂z about z₀ therefore has a zero eigenvalue with right
# eigenvector the translation mode
#       v_g(x) = ∂ₓ z₀(x)                     (L v_g = 0),
# and a corresponding LEFT zero eigenvector w(x) (the adjoint mode, L† w = 0). Nakao,
# Yanagita & Kawamura (2014) show w IS the phase-sensitivity function: a small
# perturbation p(x,t) added to the field RHS drifts the bump phase at
#       dφ/dt = ⟨ w(·−φ), p(·) ⟩ ,            ⟨w, ∂_φ z₀⟩ = 1     (phase reduction).
# (∂_φ z₀(x−φ) = −v_g, so we normalise ⟨w, v_g⟩ = 1 and carry the sign explicitly.)
# This converts "how does disorder move the bump?" from a brute-force simulation into a
# single projection against w — the exact tool the plan (§7, Nakao et al.) calls for.
#
# WHY IT EXPLAINS THE Cauchy DRIFT. The finite-N quenched disorder enters the field as a
# frozen, spatially-structured shift of the local drive, δη(x). Its first-order effect on
# the RHS is p(x) = ∂F/∂η̄ · δη(x) = −(i/2)(z₀(x)+1)²·δη(x), so the drift is LINEAR in the
# disorder,
#       dφ/dt ≈ Σ_x S_η(x)·δη(x) ,   S_η(x) = ⟨ w(x), −(i/2)(z₀(x)+1)² ⟩_ℝ²       (Eq. ★)
# (the real 2-vector dot of w and the η̄-response at each x). Because η is Cauchy, a linear
# functional Σ S_η δη of Cauchy variables is ITSELF Cauchy (a stable law): the drift inherits
# the heavy tails directly, with a scale set by Δ·Σ|S_η| that does NOT shrink as 1/√N. This is
# the mechanistic reason behind the empirical finding in run_drift5.jl ("drift is heavy-tail,
# outlier-dominated, NOT 1/√N") — now derived, not just observed.
#
# WHAT IS RIGOROUS vs MODELLED (stated honestly):
#   • Rigorous: the adjoint w / sensitivity S_η of the exact field bump, and the controlled
#     LINEAR-RESPONSE validation (a known δη tilt drives the bump at exactly ⟨w, p⟩) — this is
#     the Nakao method, validated to first order with no fitting (only an overall ± sign
#     convention, fixed once from one perturbation).
#   • Modelled: the micro→field map δη(x) ← the realized η_j (a binned local-drive deviation).
#     The STRUCTURAL consequence (Cauchy inheritance, N-independent scale) is robust; the
#     magnitude depends on this map, so we report the structure + a finite-N correlation check
#     rather than claim a calibrated drift constant.
#
# Real representation: the complex field z (length Nx) is stacked as the real vector
# u = [Re z; Im z] (length 2Nx); L is the real 2Nx×2Nx Jacobian, so the SVD null-vector
# machinery is plain real linear algebra. Operating point matches run_drift5.jl
# (Δ=0.1, κ=2, η̄=−0.4, n=2, conductance g0=0.1, matched reversals), B=0 (the kernel must be
# translation-invariant for the Goldstone mode to be exact).
# ============================================================================

using LinearAlgebra

include("phase5.jl")    # → conductance → pathint → moving → field+ring (rate, field_rhs_cond,
                        # dale_field, seed_field_bump_cond, track_field_centroid_cond, unwrap, …)

# --- real ⇄ complex stacking u = [Re z; Im z] ------------------------------------
stack_ri(z) = vcat(real.(z), imag.(z))
unstack_ri(u) = (Nx = length(u) ÷ 2; complex.(u[1:Nx], u[Nx+1:2Nx]))

# Spectral spatial derivative ∂ₓ on [0,2π) with Nx points (same convention as
# continuation.jl wavenumbers/deriv_x; duplicated locally since that file is not in scope).
function deriv_periodic(z)
    Nx = length(z)
    kvec = Float64.(vcat(0:(Nx ÷ 2 - 1), -(Nx ÷ 2):-1))
    return ifft(im .* kvec .* fft(z))
end

# --- Field Jacobian L = ∂F/∂u about the bump, central differences --------------------
# F(u) = stack( field_rhs_cond(z, …) ). 2Nx residual evals per column (≈512 length-256
# FFTs total at Nx=256 — sub-second). Central differences give the O(h²) accuracy the
# null-space SVD needs (a forward-difference Jacobian floors the smallest singular value
# near the FD error and muddies the Goldstone gate). `syn` carries g0 / reversals; pass the
# B=0 (translation-invariant) Dale config so v_g is an exact null mode.
function field_jacobian(z0, η̄, Δ, κ, syn::FieldSyn; ε=1e-6, shunt::Bool=true)
    Nx = length(z0)
    m  = 2Nx
    F(u) = stack_ri(field_rhs_cond(unstack_ri(u), η̄, Δ, κ, syn; shunt=shunt))
    u0 = stack_ri(z0)
    J  = Matrix{Float64}(undef, m, m)
    u  = copy(u0)
    @inbounds for j in 1:m
        h = ε * (abs(u0[j]) + 1.0)
        u[j] = u0[j] + h; rp = F(u)
        u[j] = u0[j] - h; rm = F(u)
        J[:, j] = (rp .- rm) ./ (2h)
        u[j] = u0[j]
    end
    return J
end

# Translation (Goldstone) mode v_g = ∂ₓ z₀, real-stacked (length 2Nx). This is the right
# null vector of L; goldstone_residual below reports ‖L v_g‖/‖v_g‖ as the correctness gate.
goldstone_mode(z0) = stack_ri(deriv_periodic(z0))
goldstone_residual(L, vg) = norm(L * vg) / norm(vg)

# --- Adjoint phase-sensitivity w: the LEFT null vector of L, normalised ⟨w, v_g⟩ = 1 ---
# SVD L = U S Vᵀ ⇒ Lᵀ u_k = σ_k v_k, so the left singular vector u_min at the smallest
# singular value is the (numerical) null vector of Lᵀ = the adjoint mode. We also return the
# RIGHT null vector V[:,end] so the caller can check it aligns with v_g (an independent
# verification that the smallest singular triple is the translation mode and not spurious).
# Returns (w, σ_min, v_right). w is normalised so ⟨w, v_g⟩ = 1 (phase-reduction convention;
# the overall sign is a convention fixed empirically by the linear-response test).
function phase_sensitivity(L, vg)
    F = svd(L)
    w  = F.U[:, end]            # left null vector  (Lᵀ w ≈ 0)
    vr = F.V[:, end]            # right null vector (L vr ≈ 0) — should ∥ v_g
    σmin = F.S[end]
    w = w ./ dot(w, vg)         # ⟨w, v_g⟩ = 1
    return w, σmin, vr
end

# --- Drift sensitivity to the local drive, S_η(x) (Eq. ★) ----------------------------
# Per-position real kernel with dφ/dt = Σ_x S_η(x)·δη(x) for a static drive perturbation
# δη(x) (a shift of the local η̄). The η̄-response of the field RHS is
# ∂F/∂η̄ = −(i/2)(z₀+1)², and S_η(x) is its real 2-vector dot with w(x). `wadj_sign` (±1)
# carries the phase-reduction sign convention (set once by calibrate_sign! / the
# linear-response test); default +1.
function eta_drift_sensitivity(z0, w; wadj_sign=1.0)
    Nx = length(z0)
    dFdη = -(im / 2) .* (z0 .+ 1).^2                 # ∂(ż)/∂η̄ at each x
    return wadj_sign .* (w[1:Nx] .* real.(dFdη) .+ w[Nx+1:2Nx] .* imag.(dFdη))
end

# Predicted bump drift rate dφ/dt for a static local-drive perturbation field δη(x).
predict_drift(Sη, δη) = dot(Sη, δη)

# --- Controlled linear-response measurement (the validation ground truth) -------------
# Impose a small static drive tilt η̄(x) = η̄ + ε·cos(x − x_p) on the EXACT field, evolve the
# settled bump, and read the LATE-TIME centroid velocity — the asymptotic translation rate the
# adjoint predicts. Subtlety: the phase-reduction drift law is an ASYMPTOTIC statement. At t=0+
# the centroid velocity is the geometric response to the instantaneous push p (it includes the
# bump's deformation transient, which relaxes at rate ~Δ over ~10 t.u.); only after that
# transient does the centroid track the pure translation mode = ⟨w,p⟩. So we run to T≳ several/Δ
# and measure the speed over the tail (frac), with ε small enough that the bump barely drifts
# (projection stays at the seed phase). Returns dφ/dt. (Averaging over [0,T] instead, as a naive
# first cut did, undershoots the adjoint by ~7% because it folds in the slow-start transient.)
function measured_drift_tilt(z0, η̄, Δ, κ, syn, x, x_p, ε; T=40.0, dt=0.02, frac=0.4)
    η̄vec = η̄ .+ ε .* cos.(x .- x_p)
    xc, _ = track_field_centroid_cond(z0, η̄vec, Δ, κ, syn, cos.(x), sin.(x); T=T, dt=dt)
    return lateral_speed(xc, dt; frac=frac)
end

# Predicted drift for the same cos(x − x_p) tilt of amplitude ε (the analytic counterpart).
predict_drift_tilt(Sη, x, x_p, ε) = predict_drift(Sη, ε .* cos.(x .- x_p))

# --- Finite-N micro → field disorder map (the MODELLED bridge) ------------------------
# Bin the N spiking neurons (at positions xr, excitabilities η) onto the Nx field grid and
# return the per-bin local-drive deviation δη(x_b) = ⟨η_j⟩_{j∈bin b} − η̄. This is the
# heuristic quenched perturbation fed to S_η for the per-realization drift prediction; honest
# caveat (see header): the OA field already carries the Lorentzian width Δ, so a bin-mean of
# Cauchy η over-weights extreme neurons relative to their bounded dynamical effect — hence we
# test the STRUCTURE (Cauchy inheritance, N-independence) and a correlation, not a magnitude.
function binned_drive_deviation(η, xr, Nx, η̄)
    δη = zeros(Nx); cnt = zeros(Int, Nx)
    @inbounds for j in eachindex(xr)
        b = mod1(round(Int, xr[j] / (2π) * Nx) + 1, Nx)
        δη[b] += η[j]; cnt[b] += 1
    end
    for b in 1:Nx
        δη[b] = cnt[b] > 0 ? δη[b] / cnt[b] - η̄ : 0.0
    end
    return δη
end
