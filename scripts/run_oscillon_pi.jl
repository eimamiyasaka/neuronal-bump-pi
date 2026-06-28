# Phase-5, Step 5.3 (oscillon half) — does the breathing bump still PATH-INTEGRATE in/near
# the gamma (synchrony-oscillation) regime? run_shunting_sweep.jl stopped at the static band
# (g0≤0.3); this script extends the g0 axis ACROSS the Hopf (g0*≈0.33) into the oscillon
# regime and asks the contribution's most distinctive question — one classical rate attractors
# cannot pose — whether a localized breathing bump can dead-reckon, at what gain, until it
# collapses.
#
# Part A — gain & breathing across static→oscillon→collapse. For each g0 (matched reversals,
#   warm-started seed): classify the regime (breathing_amplitude/classify_regime, the
#   run_shunting_sweep idiom), and where a bump exists measure the velocity→speed slope
#   ds/dB|₀ (the PI gain). Shows the bump keeps translating with a finite, measured gain as
#   g0 crosses g0* into the oscillon regime, while the breathing amplitude rises from 0 at g0*,
#   until the surround floods and the bump collapses (~0.42). Slope is reported ONLY where the
#   bump exists (collapsed ⇒ centroid is meaningless ⇒ NaN), not assumed.
#
# Part B — the moving oscillon (the money figure). At an auto-selected robust oscillon g0,
#   drive a constant self-motion command Ω and show: (i) the decoded heading φ tracks ∫Ω dt
#   (it path-integrates — unity gain is still RECALIBRATABLE above the Hopf), and (ii) the peak
#   firing rate keeps oscillating WHILE the bump translates — i.e. a *moving oscillon* — via a
#   rate(x,t) heatmap + a peak-rate(t) trace. The breathing also imprints a within-cycle
#   centroid wobble, reported as the position uncertainty the oscillon adds.
#
# DRIFT NOTE (honest scope). The per-realization heavy-tail (Cauchy) drift law of
# run_drift_shunting.jl assumes a STATIC bump + its Goldstone mode, so it applies to the static
# band, not here. In the oscillon regime the relevant within-cycle position spread is the
# breathing centroid wobble measured below; a full per-realization drift law for the moving
# oscillon (no static Goldstone) is left as future work.
#
# Operating point Δ=0.1, η̄=−0.4, κ=2, n=2, Nx=256. Field-only (exact, N→∞).
# Measurement primitives in src/phase5.jl; physics in src/conductance.jl; presentation here.

include("../src/phase5.jl")        # breathing_amplitude, classify_regime (+ conductance, core)
include("plotting.jl")
include("../src/calibration.jl")   # angvel_degpers for the deg/s readout
using Plots
using Plots.PlotMeasures

# ds/dB|₀: least-squares slope through the origin over the small-B points (s ≈ m·B).
# (verbatim from run_shunting_sweep.jl — the established 3-line script-local helper.)
small_slope(B, s; Bmax=0.03) =
    sum(B[i]*s[i] for i in eachindex(B) if 0 < B[i] <= Bmax) /
    sum(B[i]^2     for i in eachindex(B) if 0 < B[i] <= Bmax)

# tail "osc" amplitude (max−min)/mean of a series — same breathing measure as
# breathing_amplitude, applied here to a MOVING bump's peak-rate trace over its settled tail.
function tail_osc(v; frac=0.5)
    i0 = max(1, floor(Int, (1 - frac) * length(v)))
    w  = @view v[i0:end]
    return (maximum(w) - minimum(w)) / (sum(w) / length(w))
end

function main()
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    dt, Nx = 0.02, 256
    τ_m = 10.0
    x = field_positions(Nx)
    g0_star = 0.33                                # gamma Hopf (run_shunting_sweep / AUTO)
    Bs = collect(0.0:0.005:0.05)                  # small-B sweep for the gain slope

    # ===================================================================== Part A
    g0s = [0.25, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
    slope = Float64[]; osc = Float64[]; zmax = Float64[]; peakr = Float64[]
    βv = Float64[]; regimes = Symbol[]; cmin_v = Float64[]

    println("== Step 5.3 (oscillon half): PI gain + breathing across the gamma Hopf ==")
    println("Δ=$Δ κ=$κ η̄=$η̄ Nx=$Nx; gamma Hopf g0*≈$g0_star")
    println(rpad("g0",6), rpad("regime",11), rpad("osc",9), rpad("ds/dB|0",10),
            rpad("β=1/slope",11), rpad("|z|max",9), "cmin")
    for g0 in g0s
        E_E, E_I = matched_reversals(κ, g0)
        syn   = dale_field(x, E_E, E_I; g0=g0)
        zseed = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=400.0, dt=dt)   # settle (onto LC if oscillon)
        b   = breathing_amplitude(zseed, z -> field_step_cond(z, η̄, Δ, κ, syn, dt); T=800.0, dt=dt)
        reg = classify_regime(b)
        if reg == :collapsed
            m = NaN; β = NaN                                                 # no bump ⇒ no gain
        else
            sF, _ = bsweep_field_cond(Bs, x, η̄, Δ, κ, E_E, E_I, g0, zseed; T=100.0, dt=dt)
            m = small_slope(Bs, sF); β = 1 / m
        end
        push!(slope, m); push!(osc, b.osc); push!(zmax, maximum(abs.(zseed)))
        push!(peakr, b.peak); push!(βv, β); push!(regimes, reg); push!(cmin_v, b.cmin)
        println(rpad(g0,6), rpad(string(reg),11), rpad(round(b.osc,digits=3),9),
                rpad(isnan(m) ? "—" : string(round(m,digits=3)),10),
                rpad(isnan(β) ? "—" : string(round(β,digits=4)),11),
                rpad(round(maximum(abs.(zseed)),digits=4),9), round(b.cmin,digits=4))
    end

    # ===================================================================== Part B
    # pick the most robust oscillon among candidates (largest cmin = bump survives the cycle)
    println("\n== Part B: moving-oscillon demo — pick a robust oscillon g0 ==")
    cand = [0.34, 0.35, 0.36, 0.37]
    g0o = NaN; zo = ComplexF64[]; syno = nothing; Eo = (0.0, 0.0); bo = nothing; best_cmin = -Inf
    for g0 in cand
        E_E, E_I = matched_reversals(κ, g0)
        syn = dale_field(x, E_E, E_I; g0=g0)
        zs  = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=400.0, dt=dt)
        b   = breathing_amplitude(zs, z -> field_step_cond(z, η̄, Δ, κ, syn, dt); T=800.0, dt=dt)
        reg = classify_regime(b)
        println("  g0=", rpad(g0,5), " ", rpad(string(reg),10), " osc=", round(b.osc,digits=3),
                " cmin=", round(b.cmin,digits=4))
        if reg == :oscillon && b.cmin > best_cmin
            best_cmin = b.cmin; g0o = g0; zo = zs; syno = syn; Eo = (E_E, E_I); bo = b
        end
    end
    if isnan(g0o)                                       # honest fallback: no clean oscillon found
        println("  ⚠ no robustly-classified oscillon among $cand — skipping Part B demo (Part A still saved).")
    end

    # drive a constant self-motion command and record centroid + peak rate + rate snapshots
    demo = !isnan(g0o)
    local kφ, s_mean, osc_move, wobble, Ω0, R, tf, ts, φd, Φd, pk
    if demo
        E_E, E_I = Eo
        m_o = small_slope(Bs, bsweep_field_cond(Bs, x, η̄, Δ, κ, E_E, E_I, g0o, zo; T=100.0, dt=dt)[1])
        β = 1 / m_o
        Ω0 = 0.05                                       # gentle turn (small-B linear band)
        Bfun = Bfun_from_omega(omega_const(Ω0), β)
        syn0 = dale_field(x, E_E, E_I; g0=g0o)          # B=0 config for the tv stepper
        Shat = fft(sin.(x)); cosx, sinx = cos.(x), sin.(x)
        T = 150.0; nt = round(Int, T/dt); stride = 15
        nf = fld(nt, stride)
        z  = copy(zo)
        ts = Vector{Float64}(undef, nt); xc = similar(ts); pk = similar(ts)
        R  = Matrix{Float64}(undef, Nx, nf); tf = Vector{Float64}(undef, nf); f = 0
        # measurement loop — mirrors drive_field_cond but additionally records peak rate +
        # subsampled rate columns; calls the VALIDATED field_step_cond_tv (no physics re-implemented).
        for i in 1:nt
            t = (i - 1) * dt
            z = field_step_cond_tv(z, t, η̄, Δ, κ, syn0, Shat, Bfun, dt)
            r = rate.(z)
            ts[i] = i * dt; xc[i] = bump_centroid(r, cosx, sinx); pk[i] = maximum(r)
            if i % stride == 0
                f += 1; R[:, f] = r; tf[f] = i * dt
            end
        end

        # readouts on the settled tail
        i0 = fld(nt, 2)
        Φd = commanded_heading(ts, omega_const(Ω0)); φd = heading_estimate(xc)
        Φt = Φd[i0:end] .- Φd[i0]; φt = φd[i0:end] .- φd[i0]
        kφ = sum(Φt .* φt) / sum(Φt .^ 2)               # decoded/commanded gain (≈1 ⇒ path-integrates)
        s_mean = lateral_speed(xc, dt; frac=0.5)        # mean translation speed (≈Ω0 by β calibration)
        osc_move = tail_osc(pk; frac=0.5)               # breathing amplitude WHILE moving
        # within-cycle centroid wobble = detrended unwrapped centroid (position uncertainty)
        u = unwrap(xc); tt = ts[i0:end]; ut = u[i0:end]
        np = length(tt); st = sum(tt); su = sum(ut); stt = sum(tt .^ 2); stu = sum(tt .* ut)
        a_ = (np*stu - st*su) / (np*stt - st^2); c_ = (su - a_*st) / np
        wobble = maximum(abs.(ut .- (a_ .* tt .+ c_)))
        println("\n  moving oscillon g0=$g0o: Ω0=$Ω0  β=", round(β, digits=4),
                "  decoded gain kφ=", round(kφ, digits=3), "  mean speed s=", round(s_mean, digits=4),
                " (≈Ω0 ⇒ unity)")
        println("  breathing osc: stationary=", round(bo.osc, digits=3), "  while moving=", round(osc_move, digits=3),
                "  ⇒ MOVING OSCILLON")
        println("  within-cycle centroid wobble = ", round(wobble, digits=4), " rad (",
                round(wobble * 180/π, digits=2), "°) position uncertainty added by breathing")
    end

    # ============================================================== figures
    annot_hopf!(p; yloc=:bottomleft) = (vspan!(p, [g0_star, 0.42], fc=:crimson, fa=0.06, lc=:transparent,
                                               label="oscillon (gamma)");
                                        vline!(p, [g0_star], lc=:crimson, ls=:dash, lw=1.2, label="Hopf g0*≈$g0_star"))

    # Part A row
    finite_slope = [isnan(slope[i]) ? NaN : slope[i] for i in eachindex(slope)]
    pA1 = plot(g0s, finite_slope, m=:square, lc=:black, mc=:black, lw=2, ms=6,
               xlabel="conductance gain g0", ylabel="ds/dB|₀  (PI gain slope)",
               title="(a) bump still path-integrates above g0*", legend=:topright,
               label="ds/dB|₀ (NaN ⇒ collapsed)", xlims=(0.24, 0.41), left_margin=9mm, bottom_margin=5mm)
    annot_hopf!(pA1)
    pA2 = plot(g0s, osc, m=:circle, lc=:purple, mc=:purple, lw=2, ms=6,
               xlabel="conductance gain g0", ylabel="breathing osc  (max−min)/mean",
               title="(b) breathing rises from 0 at the Hopf", legend=:topleft, label="oscillon amplitude",
               xlims=(0.24, 0.41), left_margin=9mm, bottom_margin=5mm)
    annot_hopf!(pA2)
    pA3 = plot(g0s, zmax, m=:diamond, lc=:teal, mc=:teal, lw=2, ms=6,
               xlabel="conductance gain g0", ylabel="peak synchrony |z|max",
               title="(c) synchrony across the Hopf", legend=:bottomleft, label="peak |z|",
               xlims=(0.24, 0.41), left_margin=9mm, bottom_margin=5mm)
    annot_hopf!(pA3)

    # Part B row
    if demo
        pB1 = heatmap(tf, x, R, color=:viridis, xlabel="time  [t.u.]", ylabel="ring position x",
                      yticks=pi_ticks(2π), title="(d) moving oscillon: rate(x,t)  (g0=$g0o, Ω=$Ω0)",
                      colorbar_title="rate", left_margin=9mm, bottom_margin=5mm)
        pB2 = plot(ts, φd .- φd[1], lc=:black, lw=2, label="decoded φ (bump)",
                   xlabel="time  [t.u.]", ylabel="heading  [rad]",
                   title="(e) φ tracks ∫Ω dt  (gain kφ=$(round(kφ,digits=3)))", legend=:topleft,
                   left_margin=9mm, bottom_margin=5mm)
        plot!(pB2, ts, Φd .- Φd[1], lc=:green, ls=:dot, lw=2, label="commanded ∫Ω dt")
        pkmean = sum(pk) / length(pk)
        pB3 = plot(ts, pk, lc=:crimson, lw=1.4, label="peak rate (breathing while moving)",
                   xlabel="time  [t.u.]", ylabel="peak firing rate",
                   title="(f) breathing persists during translation  (osc=$(round(osc_move,digits=3)))",
                   legend=:topright, left_margin=9mm, bottom_margin=5mm)
        hline!(pB3, [pkmean], lc=:black, ls=:dot, label="")
        fig = plot(pA1, pA2, pA3, pB1, pB2, pB3, layout=(2, 3), size=(1680, 900), bottom_margin=7mm)
    else
        fig = plot(pA1, pA2, pA3, layout=(1, 3), size=(1680, 460), bottom_margin=7mm)
    end
    savefig(fig, joinpath("figures", "phase5_oscillon_pi.png"))
    println("\n  saved figures/phase5_oscillon_pi.png")

    open(joinpath("figures", "phase5_oscillon_pi.csv"), "w") do io
        println(io, "# Step 5.3 oscillon half — PI gain + breathing across the gamma Hopf (Δ=$Δ κ=$κ); g0*≈$g0_star")
        if demo
            println(io, "# Part B moving oscillon: g0=$g0o Omega=$Ω0 decoded_gain=$kφ mean_speed=$s_mean ",
                    "osc_stationary=$(bo.osc) osc_moving=$osc_move wobble_rad=$wobble")
        else
            println(io, "# Part B skipped: no robust oscillon among $cand")
        end
        println(io, "g0,regime,osc,ds_dB0,beta,zmax,peak_rate,cmin")
        for i in eachindex(g0s)
            println(io, g0s[i], ",", regimes[i], ",", osc[i], ",", slope[i], ",", βv[i], ",",
                    zmax[i], ",", peakr[i], ",", cmin_v[i])
        end
    end
    println("  saved figures/phase5_oscillon_pi.csv")
    return (g0s=g0s, regimes=regimes, slope=slope, g0o=g0o)
end

main()
