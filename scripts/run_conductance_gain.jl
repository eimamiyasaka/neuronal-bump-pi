# Phase-4, Step 4.2: PI gain under conductance synapses — re-measure with biophysics.
# Repeats the Phase-3 gain measurement (run_gain.jl) and a dynamic PI demo (run_pathint.jl)
# at the conductance operating point (Step 4.1: g0=0.1, matched reversals E=±20). Produces
# the conductance gain curve k(Ω) vs. the current-based baseline, recalibrates β, and shows
# the bump still path-integrates ∫Ω dt with conductance/Dale synapses. Physics in
# src/conductance.jl; this script orchestrates + plots.
#
#   1. CONTINUITY debug — conductance s(B) at g0→0 must equal current-based s(B).
#   2. s(B) sweeps (g0=0.1), micro + macro + current-based baseline; re-measure ds/dB|₀ ⇒ β_cond.
#   3. GAIN curves k(Ω): recalibrated (unity small-signal) + at fixed Phase-3 β0 (raw shunt shift).
#   4. DYNAMIC PI demo — drive Ω(t), heading φ tracks ∫Ω dt, micro vs macro.
#   5. tv-vs-static consistency guard.
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2. Conductance g0=0.1; β0=1/3.91 (Phase-3).

include("../src/conductance.jl")   # core + pathint tools (omega_*, heading readouts) + 4.2 plumbing
include("plotting.jl")
using Plots
using Plots.PlotMeasures

gain(s, Ω) = [Ω[i] == 0 ? NaN : s[i]/Ω[i] for i in eachindex(s)]

# Ω* = first commanded velocity where the forward gain leaves the |k−1|<tol band.
function band_edge(k, Ω; tol=0.10)
    idx = findfirst(i -> !isnan(k[i]) && abs(k[i] - 1) > tol, eachindex(k))
    idx === nothing ? (Ω[end], false) : (Ω[idx], true)
end

# ds/dB|₀: least-squares slope through the origin over the small-B points (s ≈ m·B).
function small_slope(B, s; Bmax=0.03)
    idx = findall(b -> 0 < b <= Bmax, B)
    return sum(B[i]*s[i] for i in idx) / sum(B[i]^2 for i in idx)
end

function main()
    # --- operating point + calibration ---
    Δ   = 0.1
    η̄   = -0.4
    κ   = 2.0
    n   = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt  = 0.02
    g0  = 0.1
    E_E, E_I = matched_reversals(κ, g0)         # ±20
    β0  = 1 / 3.91                              # Phase-3 calibration
    tol = 0.10

    # --- cost knobs (field fine, spiking coarse — the expensive side) ---
    ΔB_field = 0.005; ΔB_ring = 0.02; T_sweep = 100.0; frac = 0.5; rep = 10
    Bmax = 0.2
    Bs_field = collect(0.0:ΔB_field:Bmax)
    Bs_ring  = collect(0.0:ΔB_ring:Bmax)
    Nx = 256; N = 8192

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    # ========================================================================
    # 1. CONTINUITY — conductance s(B) at g0→0 reduces to current-based s(B)
    # ========================================================================
    println("== 1. continuity: conductance s(B) at g0→0 vs current-based ==")
    Khat0    = fft(field_kernel(xf))
    zseed_cb = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    Bchk     = collect(0.0:0.02:0.1)
    s_cb, _  = bsweep_field(Bchk, xf, η̄, Δ, κ, zseed_cb; T=T_sweep, dt=dt, frac=frac)
    g0t      = 0.01
    eEt, eIt = matched_reversals(κ, g0t)
    zt       = seed_field_bump_cond(xf, η̄, Δ, κ, dale_field(xf, eEt, eIt; g0=g0t); dt=dt)
    s_tiny,_ = bsweep_field_cond(Bchk, xf, η̄, Δ, κ, eEt, eIt, g0t, zt; T=T_sweep, dt=dt, frac=frac)
    dmax = maximum(abs.(s_cb .- s_tiny))
    println("  s current-based: ", round.(s_cb, digits=4))
    println("  s cond g0=0.01 : ", round.(s_tiny, digits=4))
    println("  max|Δs| = ", round(dmax, digits=5), dmax < 5e-3 ? "   ✓ reduces to current-based" : "   ✗ check sweep wrappers")

    # ========================================================================
    # 2. s(B) sweeps at g0=0.1 (micro + macro) + baseline; re-measure ds/dB|₀ ⇒ β_cond
    # ========================================================================
    println("\n== 2. conductance s(B) sweeps (g0=$g0), field + spiking, + current-based baseline ==")
    sB_cb, zEnd_cb = bsweep_field(Bs_field, xf, η̄, Δ, κ, zseed_cb; T=T_sweep, dt=dt, frac=frac, tag="cb↑", every=rep)

    zco = seed_field_bump_cond(xf, η̄, Δ, κ, dale_field(xf, E_E, E_I; g0=g0); dt=dt)
    sF_fwd, zEnd = bsweep_field_cond(Bs_field,          xf, η̄, Δ, κ, E_E, E_I, g0, zco;  T=T_sweep, dt=dt, frac=frac, tag="cond↑", every=rep)
    sF_bwd, _    = bsweep_field_cond(reverse(Bs_field), xf, η̄, Δ, κ, E_E, E_I, g0, zEnd; T=T_sweep, dt=dt, frac=frac, tag="cond↓", every=rep)
    sF_bwd = reverse(sF_bwd)

    println("  spiking sweep (N=$N) …"); flush(stdout)
    θco = seed_ring_bump_cond(η, dale_ring(xr, E_E, E_I; g0=g0), a_n, n, κ, xr; dt=dt)
    sR_fwd, θEnd = bsweep_ring_cond(Bs_ring,          xr, η, a_n, n, κ, E_E, E_I, g0, θco;  T=T_sweep, dt=dt, frac=frac, tag="ring↑", every=rep)
    sR_bwd, _    = bsweep_ring_cond(reverse(Bs_ring), xr, η, a_n, n, κ, E_E, E_I, g0, θEnd; T=T_sweep, dt=dt, frac=frac, tag="ring↓", every=rep)
    sR_bwd = reverse(sR_bwd)

    slope_cb = small_slope(Bs_field, sB_cb)
    slope_co = small_slope(Bs_field, sF_fwd)
    β_cond   = 1 / slope_co
    println("  ds/dB|₀:  current-based=", round(slope_cb, digits=3), " (Phase-3 ref 3.91)",
            "   conductance=", round(slope_co, digits=3))
    println("  β_cond = 1/", round(slope_co, digits=3), " = ", round(β_cond, digits=4),
            "   (Phase-3 β0=", round(β0, digits=4), ")")

    # ========================================================================
    # 3. GAIN curves k(Ω): recalibrated (β_cond) + fixed-β0 + current-based baseline
    # ========================================================================
    Ωf_cond = Bs_field ./ β_cond;  Ωr_cond = Bs_ring ./ β_cond     # recalibrated axis
    Ωf_fix  = Bs_field ./ β0                                       # fixed-β0 axis
    kF_cond = gain(sF_fwd, Ωf_cond);  kR_cond = gain(sR_fwd, Ωr_cond)
    kF_fix  = gain(sF_fwd, Ωf_fix)
    kF_cb   = gain(sB_cb,  Ωf_fix)                                 # baseline at β0
    Ωstar, left = band_edge(kF_cond, Ωf_cond; tol=tol)
    println("\n  small-Ω gain (recalibrated, ≈1):  field k=", round(kF_cond[2], digits=4),
            "   spiking k=", round(kR_cond[2], digits=4))
    println("  near-unity band |k−1|<", tol, ":  Ω*=", round(Ωstar, digits=4), left ? "" : " (never left band)")

    open(joinpath("figures", "conductance_gain_curve.csv"), "w") do io
        println(io, "B,Omega_cond,kF_cond,kF_fixedbeta,sF_fwd,sF_bwd")
        for i in eachindex(Bs_field)
            println(io, Bs_field[i], ",", Ωf_cond[i], ",", kF_cond[i], ",", kF_fix[i], ",", sF_fwd[i], ",", sF_bwd[i])
        end
    end

    pg = plot(xlabel="Ω  (commanded angular velocity)", ylabel="k = s/Ω  (PI gain)",
              title="Phase-4 PI gain under conductance (g0=$g0)", legend=:bottomleft,
              xlims=(0, maximum(Ωf_cond)), ylims=(0, 1.4), left_margin=8mm, bottom_margin=6mm)
    hline!(pg, [1.0], lc=:gray, ls=:dot, lw=1, label="unity gain")
    hspan!(pg, [1-tol, 1+tol], fc=:green, fa=0.07, lc=:transparent, label="±$tol band")
    plot!(pg, Ωf_cond, kF_cb,   lc=:gray,   lw=2, ls=:dot,  label="current-based baseline (β0)")
    plot!(pg, Ωf_cond, kF_cond, lc=:black,  lw=2,           label="conductance field (β_cond)")
    scatter!(pg, Ωr_cond, kR_cond, m=:xcross, mc=:darkred, ms=4, label="conductance spiking (β_cond)")
    plot!(pg, Ωf_fix,  kF_fix,  lc=:orange, lw=2, ls=:dash, label="conductance field (fixed β0)")
    vline!(pg, [Ωstar], lc=:green, ls=:dash, lw=1, label="Ω* band edge")
    savefig(pg, joinpath("figures", "conductance_gain_curve.png"))
    println("  saved conductance_gain_curve.png + .csv")

    # ========================================================================
    # 4. DYNAMIC PI demo — heading φ(t) tracks ∫Ω dt, micro vs macro
    # ========================================================================
    println("\n== 4. dynamic PI demo under conductance (β_cond) ==")
    synf0 = dale_field(xf, E_E, E_I; g0=g0)          # B=0 config for the tv steppers
    synr0 = dale_ring(xr, E_E, E_I; g0=g0)
    commands = [
        ("constant",  omega_const(0.10),                                         300.0),
        ("realistic", omega_realistic(300.0, dt; τ=20.0, σ=0.05, Ωbar=0.0,
                                      rng=Random.MersenneTwister(2)),             300.0),
    ]
    head_panels = Plots.Plot[]
    for (name, Ωfun, T) in commands
        Bfun = Bfun_from_omega(Ωfun, β_cond)
        tsf, xcf, _ = drive_field_cond(zco, η̄, Δ, κ, xf, synf0, Bfun; T=T, dt=dt)
        println("  [$name] spiking drive (N=$N) …"); flush(stdout)
        tsr, xcr, _ = drive_ring_cond(θco, η, xr, synr0, a_n, n, κ, Bfun; T=T, dt=dt)
        φf = heading_estimate(xcf); φr = heading_estimate(xcr)
        Φ  = commanded_heading(tsf, Ωfun)
        # decoded-vs-commanded gain = slope of φ vs Φ over the run (robust least-squares)
        kφf = sum((Φ .- sum(Φ)/length(Φ)) .* (φf .- φf[1])) / sum((Φ .- sum(Φ)/length(Φ)).^2)
        kφr = sum((Φ .- sum(Φ)/length(Φ)) .* (φr .- φr[1])) / sum((Φ .- sum(Φ)/length(Φ)).^2)
        println("  [$name] decoded/commanded gain:  field=", round(kφf, digits=3), "  spiking=", round(kφr, digits=3))
        p = plot(tsf, φf .- φf[1], lc=:black, lw=2, label="field φ",
                 xlabel="time", ylabel="heading (rad)", title="$name: φ tracks ∫Ω dt", left_margin=8mm)
        plot!(p, tsr, φr .- φr[1], lc=:darkred, lw=2, ls=:dash, label="spiking φ")
        plot!(p, tsf, Φ .- Φ[1],   lc=:green,   lw=1.5, ls=:dot, label="commanded ∫Ω dt")
        push!(head_panels, p)
    end
    fig = plot(head_panels..., layout=(1, length(head_panels)), size=(500*length(head_panels), 380))
    savefig(fig, joinpath("figures", "conductance_heading_tracking.png"))
    println("  saved conductance_heading_tracking.png")

    # ========================================================================
    # 5. tv-vs-static consistency — const Ω drive speed == static-B sweep speed
    # ========================================================================
    println("\n== 5. tv-vs-static consistency (const Ω=0.1) ==")
    B_demo = β_cond * 0.10
    tsf, xcf, _ = drive_field_cond(zco, η̄, Δ, κ, xf, synf0, Bfun_from_omega(omega_const(0.10), β_cond); T=200.0, dt=dt)
    s_tv = lateral_speed(xcf, dt; frac=frac)
    xcS, _ = track_field_centroid_cond(zco, η̄, Δ, κ, dale_field(xf, E_E, E_I; g0=g0, B=B_demo), cos.(xf), sin.(xf); T=200.0, dt=dt)
    s_static = lateral_speed(xcS, dt; frac=frac)
    println("  B_demo=", round(B_demo, digits=4), "  s_tv=", round(s_tv, digits=4),
            "  s_static=", round(s_static, digits=4), "  |Δ|=", round(abs(s_tv-s_static), digits=5))

    return β_cond
end

main()
