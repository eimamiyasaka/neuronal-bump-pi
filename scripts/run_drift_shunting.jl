# Phase-5, Step 5.3 (drift half) — how the per-realization darkness DRIFT depends on
# shunting g0 / proximity to the gamma Hopf. Completes the missing half of the headline:
# run_shunting_sweep.jl measured how PI GAIN falls toward the Hopf (g0*≈0.33); this script
# measures how the heading-code DRIFT behaves over the same static band, using the exact
# Nakao adjoint drift law instead of a noisy finite-N spiking sweep.
#
# WHY THE ADJOINT SCALE IS THE RIGHT METRIC. The finite-N darkness drift is Cauchy/heavy-
# tailed with N-INDEPENDENT scale Δ·Σ|S_η| (derived + validated in src/drift_adjoint.jl /
# run_drift_adjoint.jl, slope 0.997 r 0.999 at g0=0.1), where S_η(x) is the bump's
# phase-sensitivity (adjoint) to a local-drive perturbation. So the per-realization drift
# scale at each g0 is exactly Δ·Σ|S_η|(g0) — an exact (N→∞) functional of the field bump,
# requiring no stochastic sampling. We re-extract S_η along the g0 axis and report scale(g0).
#
# SELF-VALIDATING (the project's no-test-suite convention). At each g0 (B=0, matched
# reversals) we check three correctness gates and only trust the scale where they pass:
#   • RHS residual maxₓ|field_rhs_cond(z0)|  small  ⇒ z0 is a genuine STATIC fixed point
#     (above the Hopf there is none; such a point is flagged, not trusted);
#   • Goldstone residual ‖L·∂ₓz₀‖/‖∂ₓz₀‖ → 0       ⇒ ∂ₓz₀ is the null (translation) mode;
#   • right-null alignment |⟨v_right,∂ₓz₀⟩| → 1     ⇒ the smallest singular triple of L is
#     the translation mode, not the breathing mode creeping down toward the Hopf.
# A controlled linear-response no-fit check at the operating point g0=0.10 anchors the
# absolute magnitude (reproducing figures/drift_adjoint.csv); the per-g0 gates certify the
# mode is correct at every other g0, so the g0-DEPENDENCE is not an artifact.
#
# SCOPE (honest). Band g0∈[0.05,0.32], below the Hopf, where a static bump + its Goldstone
# mode exist. g0=0 is the singular matched_reversals limit (=current-based) and is the
# weak-shunt endpoint of the trend, not special-cased. The oscillon regime (g0≳0.33) has no
# static bump, so this adjoint construction does not apply there — that regime is handled by
# run_oscillon_pi.jl. Operating point Δ=0.1, η̄=−0.4, κ=2, n=2, Nx=256, B=0.
#
# Physics/measurement in src/drift_adjoint.jl + src/conductance.jl; presentation here.

include("../src/drift_adjoint.jl")  # field_jacobian, goldstone_mode/residual, phase_sensitivity,
                                    # eta_drift_sensitivity, measured/predict_drift_tilt (+ conductance/core)
include("plotting.jl")
using Plots
using Plots.PlotMeasures
using LinearAlgebra                 # dot, norm (also brought in by drift_adjoint.jl)
using Statistics                    # cor (linear-response anchor)

function main()
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    dt, Nx = 0.02, 256
    x = field_positions(Nx)
    g0s = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.32]
    g0_star = 0.33                  # gamma Hopf (run_shunting_sweep / AUTO g0*=0.33026)
    g0_anchor = 0.10                # operating point — magnitude anchor
    resid_tol, gres_tol, align_tol = 1e-3, 1e-3, 0.99   # per-g0 trust gates

    peak = Float64[]; zmax = Float64[]; resid_v = Float64[]; gres_v = Float64[]
    align_v = Float64[]; σ_v = Float64[]; Smax = Float64[]; scale = Float64[]; trust = Bool[]
    Sη_store = Dict{Float64,Vector{Float64}}(); z0_store = Dict{Float64,Vector{ComplexF64}}()
    syn_store = Dict{Float64,FieldSyn}()

    println("== Step 5.3 (drift half): adjoint drift scale Δ·Σ|S_η| vs shunting g0 ==")
    println("operating point Δ=$Δ κ=$κ η̄=$η̄ Nx=$Nx, B=0, matched reversals; gamma Hopf g0*≈$g0_star")
    println(rpad("g0",6), rpad("peakrate",10), rpad("|z|max",9), rpad("RHSres",11),
            rpad("gres",11), rpad("align",9), rpad("σ_min",11), rpad("Δ·Σ|Sη|",10), "trust")

    for g0 in g0s
        E_E, E_I = matched_reversals(κ, g0)
        syn = dale_field(x, E_E, E_I; g0=g0)                            # B=0, flat-K_I operating config
        z0  = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=600.0, dt=dt) # long settle near the Hopf
        resid = maximum(abs.(field_rhs_cond(z0, η̄, Δ, κ, syn)))         # gate: static fixed point?
        L  = field_jacobian(z0, η̄, Δ, κ, syn)
        vg = goldstone_mode(z0)
        gres = goldstone_residual(L, vg)                                # gate: ∂ₓz₀ is null?
        w, σmin, vr = phase_sensitivity(L, vg)
        align = abs(dot(vr, vg)) / (norm(vr) * norm(vg))                # gate: null triple = translation?
        Sη = eta_drift_sensitivity(z0, w)                              # sign-free for the scale (abs below)
        sc = Δ * sum(abs.(Sη))                                          # Cauchy drift scale = median|drift|
        ok = (resid < resid_tol) && (gres < gres_tol) && (align > align_tol)

        push!(peak, maximum(rate.(z0))); push!(zmax, maximum(abs.(z0)))
        push!(resid_v, resid); push!(gres_v, gres); push!(align_v, align)
        push!(σ_v, σmin); push!(Smax, maximum(abs.(Sη))); push!(scale, sc); push!(trust, ok)
        Sη_store[g0] = Sη; z0_store[g0] = z0; syn_store[g0] = syn
        println(rpad(g0,6), rpad(round(maximum(rate.(z0)),digits=4),10),
                rpad(round(maximum(abs.(z0)),digits=4),9), rpad(round(resid,sigdigits=2),11),
                rpad(round(gres,sigdigits=2),11), rpad(round(align,digits=5),9),
                rpad(round(σmin,sigdigits=2),11), rpad(round(sc,digits=4),10), ok ? "✓" : "✗ FLAG")
    end

    # ---- magnitude anchor: controlled linear-response no-fit check at g0=0.10 -----------
    # Impose η̄(x)=η̄+ε cos(x−x_p); the adjoint must predict the late-time bump drift with no
    # fit (only the overall ± sign is a convention). Reuses the validated z0/syn/Sη from the loop.
    z0a = z0_store[g0_anchor]; syna = syn_store[g0_anchor]; Sηa = copy(Sη_store[g0_anchor])
    ε, Tlr = 0.003, 30.0
    xps  = collect(range(0, 2π, length=13)[1:12])
    meas = [measured_drift_tilt(z0a, η̄, Δ, κ, syna, x, xp, ε; T=Tlr, dt=dt) for xp in xps]
    pred = [predict_drift_tilt(Sηa, x, xp, ε) for xp in xps]
    sgn  = sign(sum(meas .* pred)); pred .*= sgn                        # fix sign convention once
    slope_lr = sum(meas .* pred) / sum(abs2, pred)                      # want ≈ 1 (magnitude correct)
    r_lr = cor(pred, meas)
    println("\n-- magnitude anchor (g0=$g0_anchor, controlled linear response, 12 tilt phases): ",
            "slope=", round(slope_lr, digits=4), " (→1, NO fit)  r=", round(r_lr, digits=5))
    println("   drift scale at g0=$g0_anchor = ", round(scale[findfirst(==(g0_anchor), g0s)], digits=4),
            "  (ref figures/drift_adjoint.csv ≈ 0.187)")

    # ---- the trend (the result) ---------------------------------------------------------
    tr = [scale[i] for i in eachindex(g0s) if trust[i]]
    println("\n-- drift scale across the band: ", [round(s, digits=3) for s in scale])
    if length(tr) >= 2
        println("   scale(g0=", g0s[findfirst(trust)], ")=", round(tr[1], digits=3),
                " → scale(g0=", g0s[findlast(trust)], ")=", round(tr[end], digits=3),
                tr[end] > tr[1] ? "  ⇒ drift GROWS toward the gamma Hopf (heading code less stable as gain falls)"
                                : "  ⇒ drift does not grow toward the Hopf")
    end

    # ============================================================== figures
    msk = trust                                                        # plot trusted points solid
    p1 = plot(g0s[msk], scale[msk], m=:circle, lc=:crimson, mc=:crimson, lw=2, ms=6,
              xlabel="conductance gain g0  (shunt strength)",
              ylabel="drift scale Δ·Σ|S_η|  [rad/t.u.]", title="(a) drift grows toward the gamma Hopf",
              legend=:topleft, label="Cauchy drift scale", xlims=(0, g0_star + 0.02),
              left_margin=9mm, bottom_margin=5mm)
    any(.!msk) && scatter!(p1, g0s[.!msk], scale[.!msk], mc=:gray, ms=5, label="gate-flagged")
    vline!(p1, [g0_star], lc=:crimson, ls=:dash, lw=1.2, label="Hopf g0*≈$g0_star")
    plot!(twinx(p1), g0s, zmax, m=:diamond, lc=:purple, mc=:purple, lw=2, ls=:dash,
          ylabel="peak synchrony |z|max", legend=:topright, label="peak |z| (falls)",
          xlims=(0, g0_star + 0.02))

    # normalized S_η(x) shape at a few g0 (broadens as synchrony falls toward the Hopf)
    p2 = plot(xlabel="ring position x", ylabel="S_η(x) / max|S_η|", title="(b) phase-sensitivity broadens",
              xticks=pi_ticks(2π), legend=:topright, left_margin=7mm, bottom_margin=5mm)
    for (g0, c) in zip([0.10, 0.25, 0.32], (:teal, :orange, :crimson))
        Sη = Sη_store[g0]
        # align sign by the central-peak convention so the shapes overlay comparably
        s = Sη ./ maximum(abs.(Sη)); s .*= sign(s[argmax(abs.(s))])
        plot!(p2, x, s, lc=c, lw=2, label="g0=$g0")
    end
    hline!(p2, [0], lc=:black, ls=:dot, label="")

    # correctness gates across the band (certify the mode stays the translation mode)
    p3 = plot(g0s, max.(gres_v, 1e-16), m=:square, lc=:black, mc=:black, lw=2, yscale=:log10,
              xlabel="conductance gain g0", ylabel="Goldstone residual / σ_min  (log)",
              title="(c) per-g0 correctness gates", legend=:left, label="‖L·∂ₓz₀‖/‖∂ₓz₀‖ (→0)",
              xlims=(0, g0_star + 0.02), left_margin=9mm, bottom_margin=5mm)
    plot!(p3, g0s, max.(σ_v, 1e-16), m=:utriangle, lc=:gray, mc=:gray, lw=2, ls=:dash, label="σ_min (→0)")
    plot!(twinx(p3), g0s, align_v, m=:circle, lc=:green, mc=:green, lw=2,
          ylabel="right-null alignment (→1)", legend=:right, label="|⟨v_right,∂ₓz₀⟩|",
          ylims=(0.9, 1.001), xlims=(0, g0_star + 0.02))

    fig = plot(p1, p2, p3, layout=(1, 3), size=(1620, 450), bottom_margin=7mm)
    savefig(fig, joinpath("figures", "phase5_drift_shunting.png"))
    println("\n  saved figures/phase5_drift_shunting.png")

    open(joinpath("figures", "phase5_drift_shunting.csv"), "w") do io
        println(io, "# Step 5.3 drift half — adjoint Cauchy drift scale vs shunting g0 (Δ=$Δ κ=$κ, B=0)")
        println(io, "# magnitude anchor g0=$g0_anchor: linresp slope=$slope_lr r=$r_lr (ref drift_adjoint.csv scale≈0.187)")
        println(io, "g0,peak_rate,zmax,RHS_resid,gres,align,sigma_min,Smax,drift_scale,trust")
        for i in eachindex(g0s)
            println(io, g0s[i], ",", peak[i], ",", zmax[i], ",", resid_v[i], ",", gres_v[i], ",",
                    align_v[i], ",", σ_v[i], ",", Smax[i], ",", scale[i], ",", trust[i])
        end
    end
    println("  saved figures/phase5_drift_shunting.csv")
    return (g0s=g0s, scale=scale, trust=trust, slope_lr=slope_lr, r_lr=r_lr)
end

main()
