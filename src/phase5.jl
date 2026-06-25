# ============================================================================
# Phase-5 measurement layer (project.md §15, Steps 5.1–5.3): the contribution —
# how the SHUNTING gain g0 governs synchrony, the gamma (breathing) Hopf, PI gain,
# and bump drift. Like moving.jl / pathint.jl / conductance.jl this is a READOUT
# layer on top of the unchanged models — no new physics, NO Plots. Parameters +
# presentation live in the run scripts (run_regime_map / run_shunting_sweep / run_drift5).
#
# What the simulations established (and what this layer measures):
#   • The accessible SUSTAINED oscillatory ("gamma") regime is reached along the
#     conductance-gain axis g0 (the shunt strength), NOT via low Δ / high κ — those
#     are slow under-damped TRANSIENTS (the CLAUDE.md "fishbone"), confirmed decaying
#     under long settling. So the synchrony-oscillation Hopf rides the shunting axis.
#   • At the operating point (Δ=0.1, κ=2) a supercritical Hopf at g0* ≈ 0.32–0.33 turns
#     the static bump into a LOCALIZED breathing bump (oscillon). Below it the bump is
#     static; the headline shunting sweep lives in 0 ≤ g0 ≤ 0.3 (static band).
#
# Contents:
#   breathing_amplitude — tail oscillation amplitude of the peak rate + bump contrast
#                         (the oscillon detector for the regime map / Hopf location)
#   classify_regime     — :static / :oscillon / :collapsed from those numbers
#   drift_rate_stats    — per-realization darkness drift rate + RMS over seeds (∝N^−1/2),
#                         the robust finite-size heading-error measure (Step 5.2)
# ============================================================================

include("conductance.jl")   # → pathint → moving → field+ring; rate, field_step_cond, …

# --- Oscillon detector ------------------------------------------------------
# Evolve a SETTLED field from `zseed` under the supplied one-step map `stepfun`
# (z -> next z), and over the last `tail` fraction of the run record, each step, the
# peak firing rate maxₓ rate(z) and the bump contrast (maxₓ − minₓ) rate. Returns a
# NamedTuple:
#   osc  = (max−min)/mean of the peak-rate tail   — breathing amplitude (≈0 ⇒ static)
#   peak = mean peak rate over the tail
#   cmin = min bump contrast over the tail         — >0 ⇒ a bump survives the whole cycle
#   cmax = max bump contrast over the tail         — ≈0 ⇒ no bump at all (collapsed)
# REQUIRES `zseed` already settled (T_free ≳ 10/Δ); a transient masquerades as osc>0.
function breathing_amplitude(zseed, stepfun; T=1000.0, dt=0.02, tail=0.3)
    nt = round(Int, T/dt)
    i0 = floor(Int, (1 - tail) * nt)
    z  = copy(zseed)
    peaks     = Float64[]
    contrasts = Float64[]
    for i in 1:nt
        z = stepfun(z)
        if i >= i0
            r = rate.(z)
            push!(peaks, maximum(r))
            push!(contrasts, maximum(r) - minimum(r))
        end
    end
    pm = sum(peaks) / length(peaks)
    return (osc = (maximum(peaks) - minimum(peaks)) / pm, peak = pm,
            cmin = minimum(contrasts), cmax = maximum(contrasts))
end

# Classify a breathing_amplitude result. `osc_tol` is the sustained-oscillation
# threshold (above numerical floor); `contrast_tol` the bump-existence floor.
#   :collapsed — no localized bump (cmax below floor)
#   :static    — a bump that does not breathe
#   :oscillon  — a sustained breathing (localized) bump = the synchrony-oscillation regime
function classify_regime(b; osc_tol=0.02, contrast_tol=0.02)
    b.cmax < contrast_tol && return :collapsed
    b.osc  < osc_tol      && return :static
    return :oscillon
end

# --- Finite-size drift (Step 5.2) -------------------------------------------
# In darkness (Ω=0, B=0) the EXACT field bump is translation-marginal — it does not
# drift (a control). The finite-N spiking bump wanders because the FROZEN-η realization
# breaks the ring's translation symmetry into a fixed quenched potential, whose gradient
# pushes the bump at a realization-specific rate.
#
# SCALING (corrected — see step5_gate.md and the adjoint analysis in src/drift_adjoint.jl):
# this is NOT the light-tailed 1/√N CLT envelope. η is Cauchy/Lorentzian (the distribution
# that makes OA exact), heavy-tailed with NO finite variance. The phase-reduction drift law
# (Nakao 2014 adjoint) makes the per-realization drift rate a LINEAR functional of the
# quenched disorder, drift ≈ Σ_b S_η(x_b)·δη_b; a linear combination of Cauchy variables is
# itself Cauchy (a stable law), so the drift is Cauchy with an N-INDEPENDENT scale ∝ Δ·∫|S_η|
# — heavy-tailed and outlier-dominated, not concentrating as N grows. We therefore report the
# drift-rate statistics directly (scatter + median) rather than impose a spurious 1/√N fit.
#
# Given S unwrapped, zero-started heading series φ_seeds[s](t) on grid ts, return each
# realization's net drift rate r_s = (φ_s[end]−φ_s[1])/T and their RMS over seeds.
function drift_rate_stats(φ_seeds, ts)
    T = ts[end] - ts[1]
    rates = [(φ[end] - φ[1]) / T for φ in φ_seeds]
    rms   = sqrt(sum(abs2, rates) / length(rates))
    return (rates = rates, rms = rms)
end
