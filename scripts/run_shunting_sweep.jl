# Phase-5, Step 5.3 (headline) — how SHUNTING (conductance gain g0) governs synchrony,
# PI gain, and the usable velocity band, and how the static bump approaches the gamma
# (synchrony-oscillation) Hopf. Exact field model (N→∞), so these trends are not finite-N
# artifacts. Measurement primitives in src/phase5.jl; physics in src/conductance.jl.
#
# Mechanistic question (project.md §15, 5.3): "as the network approaches the gamma regime,
# PI gain/drift behaves as …, and increasing shunting does …". This script answers the
# shunting half on the field; run_drift5.jl adds the finite-N drift half.
#
# For each g0 over the STATIC band 0 ≤ g0 ≤ 0.3 (operating point g0=0.1), at matched
# reversals E=±κ/g0 (drive held ⇒ g0 is pure shunt strength):
#   • peak synchrony |z|_max  (bump coherence)
#   • velocity→speed slope ds/dB|₀  ⇒  β(g0)=1/slope  (asymmetry needed per unit Ω)
#   • unity-gain band edge Ω*  (where |k−1| first exceeds 0.1)
# plus a self-contained location of the Hopf g0* (onset of sustained breathing) above the band.
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2, Nx=256.

include("../src/phase5.jl")        # breathing_amplitude, classify_regime (+ conductance, core)
include("plotting.jl")
include("../src/calibration.jl")   # angvel_degpers for the band-edge in deg/s
using Plots
using Plots.PlotMeasures

# ds/dB|₀: least-squares slope through the origin over the small-B points (s ≈ m·B).
small_slope(B, s; Bmax=0.03) =
    sum(B[i]*s[i] for i in eachindex(B) if 0 < B[i] <= Bmax) /
    sum(B[i]^2     for i in eachindex(B) if 0 < B[i] <= Bmax)

# Ω* = first commanded velocity (dimensionless) where forward gain leaves the ±tol band.
function band_edge(k, Ω; tol=0.10)
    idx = findfirst(i -> !isnan(k[i]) && abs(k[i]-1) > tol, eachindex(k))
    idx === nothing ? Ω[end] : Ω[idx]
end

function main()
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    dt, Nx = 0.02, 256
    tol = 0.10
    τ_m = 10.0                      # for the deg/s band-edge readout (see calibration.jl)
    x = field_positions(Nx)

    Bs = collect(0.0:0.005:0.2)     # s(B) sweep range (slope from small B; band edge from full)
    g0s = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # 0.0 = current-based limit

    zmax = Float64[]; slope = Float64[]; βg = Float64[]; Ωstar = Float64[]; Ωstar_deg = Float64[]

    println("== Step 5.3: shunting sweep (exact field), Δ=$Δ κ=$κ ==")
    println(rpad("g0",6), rpad("|z|max",10), rpad("ds/dB|0",10), rpad("β=1/slope",12),
            rpad("Ω*",8), "Ω* [deg/s]")
    for g0 in g0s
        if g0 == 0.0
            Khat  = fft(field_kernel(x))
            zseed = seed_field_bump(x, η̄, Δ, κ, Khat; dt=dt)
            sF, _ = bsweep_field(Bs, x, η̄, Δ, κ, zseed; T=100.0, dt=dt)
        else
            E_E, E_I = matched_reversals(κ, g0)
            zseed = seed_field_bump_cond(x, η̄, Δ, κ, dale_field(x, E_E, E_I; g0=g0); dt=dt)
            sF, _ = bsweep_field_cond(Bs, x, η̄, Δ, κ, E_E, E_I, g0, zseed; T=100.0, dt=dt)
        end
        m  = small_slope(Bs, sF)
        β  = 1 / m
        Ω  = Bs .* m                                   # Ω = B/β = B·slope
        k  = [Ω[i] == 0 ? NaN : sF[i]/Ω[i] for i in eachindex(sF)]
        Ωs = band_edge(k, Ω; tol=tol)
        push!(zmax, maximum(abs.(zseed))); push!(slope, m); push!(βg, β)
        push!(Ωstar, Ωs); push!(Ωstar_deg, angvel_degpers(Ωs; τ_m=τ_m))
        println(rpad(g0,6), rpad(round(maximum(abs.(zseed)),digits=4),10),
                rpad(round(m,digits=3),10), rpad(round(β,digits=4),12),
                rpad(round(Ωs,digits=3),8), round(angvel_degpers(Ωs; τ_m=τ_m), digits=0))
    end

    # --- self-contained Hopf location: onset of sustained breathing above the band ---
    println("\n== locate gamma Hopf g0* (onset of sustained breathing) ==")
    g0_probe = [0.30, 0.32, 0.33, 0.34, 0.36]
    g0_star = NaN
    for g0 in g0_probe
        E_E, E_I = matched_reversals(κ, g0)
        syn = dale_field(x, E_E, E_I; g0=g0)
        zs  = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=400.0, dt=dt)   # settle (Δ=0.1 ⇒ fast)
        b   = breathing_amplitude(zs, z -> field_step_cond(z, η̄, Δ, κ, syn, dt); T=800.0, dt=dt)
        reg = classify_regime(b)
        if reg == :oscillon && isnan(g0_star); g0_star = g0; end
        println("  g0=", rpad(g0,5), "  osc=", rpad(round(b.osc,digits=3),7),
                "  ", reg)
    end
    println("  ⇒ gamma Hopf g0* ≈ ", g0_star)

    # ------------------------------------------------------------------ figure
    annot_hopf!(p) = (vspan!(p, [g0_star, 0.42], fc=:crimson, fa=0.06, lc=:transparent,
                             label="oscillon (gamma)"); vline!(p, [g0_star], lc=:crimson, ls=:dash, lw=1.2, label="Hopf g0*≈$(g0_star)"))

    p1 = plot(g0s, zmax, m=:circle, lc=:purple, mc=:purple, lw=2, label="peak |z|",
              xlabel="conductance gain g0  (shunt strength)", ylabel="peak synchrony |z|max",
              title="(a) shunting lowers synchrony", legend=:bottomleft, xlims=(0,0.42), left_margin=8mm)
    annot_hopf!(p1)

    p3 = plot(g0s, Ωstar_deg, m=:utriangle, lc=:teal, mc=:teal, lw=2, label="Ω* band edge",
              xlabel="conductance gain g0", ylabel="Ω* band edge  [deg/s]",
              title="(b) usable PI velocity band", legend=:bottomleft, xlims=(0,0.42), left_margin=10mm)
    annot_hopf!(p3)

    # slope + β (twinx) placed RIGHTMOST so the right-axis label has nothing beside it
    p2 = plot(g0s, slope, m=:square, lc=:black, mc=:black, lw=2, label="ds/dB|₀ (gain slope)",
              xlabel="conductance gain g0", ylabel="velocity→speed slope ds/dB|₀",
              title="(c) shunting lowers PI gain", legend=:bottomleft, xlims=(0,0.42),
              left_margin=8mm, right_margin=14mm)
    plot!(twinx(p2), g0s, βg, m=:diamond, lc=:orange, mc=:orange, lw=2, ls=:dash,
          ylabel="β = 1/slope", legend=:topright, label="β for unity gain", xlims=(0,0.42))
    annot_hopf!(p2)

    fig = plot(p1, p3, p2, layout=(1,3), size=(1560, 430), bottom_margin=6mm)
    savefig(fig, joinpath("figures", "phase5_shunting.png"))
    println("\n  saved figures/phase5_shunting.png")

    # CSV for the writeup
    open(joinpath("figures", "phase5_shunting.csv"), "w") do io
        println(io, "g0,zmax,slope,beta,Omega_star,Omega_star_degpers")
        for i in eachindex(g0s)
            println(io, g0s[i], ",", zmax[i], ",", slope[i], ",", βg[i], ",", Ωstar[i], ",", Ωstar_deg[i])
        end
    end
    println("  saved figures/phase5_shunting.csv")
    return (g0s=g0s, zmax=zmax, slope=slope, g0_star=g0_star)
end

main()
