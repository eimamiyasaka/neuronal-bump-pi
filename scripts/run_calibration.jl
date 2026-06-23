# Dimensional calibration — physical-unit (Hz, deg, deg/s, s) counterparts of the
# existing dimensionless figures, for empirical comparison with HD-cell tuning and
# behavioral angular-velocity data. The model is UNCHANGED; this only fixes the one
# free physical scale (membrane time constant τ_m) and re-expresses existing readouts.
# Conversions live in src/calibration.jl; this script orchestrates + plots.
#
# Produces SEPARATE figures (prefixed calib_*) so the dimensionless originals are
# untouched:
#   figures/calib_static_bump_hz.png       — bump firing rate [Hz] vs heading [deg]
#   figures/calib_gain_curve_degpers.png   — PI gain k vs commanded Ω [deg/s]
#   figures/calib_heading_tracking.png     — decoded heading [deg] vs time [s]
# plus a printed calibration table at τ_m = 10 and 20 ms.
#
# Operating point matches the conductance figures: Δ=0.1, κ=2, η̄=−0.4, n=2, g0=0.1,
# matched reversals E=±20 (Step 4.1/4.2). β_cond is re-measured from the field sweep.

include("../src/conductance.jl")    # core + conductance + pathint readouts
include("../src/calibration.jl")    # rate_hz, heading_deg, angvel_degpers/min, fwhm_deg, time_s
include("plotting.jl")
using Plots
using Plots.PlotMeasures

# ds/dB|₀: least-squares slope through the origin over the small-B points (s ≈ m·B).
small_slope(B, s; Bmax=0.03) =
    sum(B[i]*s[i] for i in eachindex(B) if 0 < B[i] <= Bmax) /
    sum(B[i]^2     for i in eachindex(B) if 0 < B[i] <= Bmax)

# Ω* = first commanded velocity (dimensionless) where forward gain leaves the band.
function band_edge(k, Ω; tol=0.10)
    idx = findfirst(i -> !isnan(k[i]) && abs(k[i]-1) > tol, eachindex(k))
    idx === nothing ? Ω[end] : Ω[idx]
end

function main()
    # --- operating point (matches the conductance figures) ---
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt  = 0.02
    g0  = 0.1
    E_E, E_I = matched_reversals(κ, g0)          # ±20
    Nx, N = 256, 8192
    tol = 0.10

    # The ONE physical scale. τ_m = membrane time constant (ms). 10 ms is the headline
    # value; the table below also reports 20 ms (slow end). Everything else is derived.
    τ_m = 10.0

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    synf0 = dale_field(xf, E_E, E_I; g0=g0)
    synr0 = dale_ring(xr, E_E, E_I; g0=g0)

    # ========================================================================
    # 1. Calibrated STATIC bump — firing rate [Hz] vs heading [deg], micro vs macro
    # ========================================================================
    println("== 1. static bump (Hz vs deg) ==")
    zco = seed_field_bump_cond(xf, η̄, Δ, κ, synf0; dt=dt)
    rfield = rate.(zco)                                            # spikes per t.u.

    println("  spiking seed + mean frequencies (N=$N) …"); flush(stdout)
    θco = seed_ring_bump_cond(η, synr0, a_n, n, κ, xr; dt=dt)
    fk  = mean_frequencies_cond(θco, η, synr0, a_n, n, κ; T=100.0, dt=dt)   # spikes per t.u.
    fk_s = ring_smooth(fk, 256)

    peak_hz_field = maximum(rate_hz(rfield; τ_m=τ_m))
    peak_hz_ring  = maximum(rate_hz(fk_s;  τ_m=τ_m))
    fwhm          = fwhm_deg(rfield)
    println("  peak rate: field=", round(peak_hz_field, digits=1), " Hz  spiking=",
            round(peak_hz_ring, digits=1), " Hz   |   bump FWHM=", round(fwhm, digits=0), "°")

    # y-axis clipped to ~1.3× the bump peak: the raw f_k scatter has a few fast outliers
    # (the Lorentzian-η tail spins fast), as in the dimensionless figure's fixed axis.
    p1 = plot(xlabel="heading  [deg]", ylabel="firing rate  [Hz]",
              title="Calibrated static bump (τ_m=$(τ_m) ms),  g0=$g0",
              legend=:topright, xlims=(0, 360), xticks=0:90:360,
              ylims=(0, 1.3*peak_hz_ring), left_margin=8mm, bottom_margin=5mm)
    scatter!(p1, heading_deg(xr), rate_hz(fk; τ_m=τ_m), ms=1, mc=:gray, ma=0.25, label="spiking f_k")
    plot!(p1, heading_deg(xr), rate_hz(fk_s; τ_m=τ_m), lc=:orange, lw=2, ls=:dash, label="spiking (smoothed)")
    plot!(p1, heading_deg(xf), rate_hz(rfield; τ_m=τ_m), lc=:black, lw=2, label="field rate")
    savefig(p1, joinpath("figures", "calib_static_bump_hz.png"))
    println("  saved calib_static_bump_hz.png")

    # ========================================================================
    # 2. Calibrated PI GAIN curve — k vs commanded Ω [deg/s]  (field-only sweep)
    # ========================================================================
    println("\n== 2. PI gain vs Ω [deg/s] (field sweep) ==")
    Bmax = 0.2
    Bs   = collect(0.0:0.005:Bmax)
    sF, _ = bsweep_field_cond(Bs, xf, η̄, Δ, κ, E_E, E_I, g0, zco; T=100.0, dt=dt, frac=0.5)
    β_cond = 1 / small_slope(Bs, sF)
    Ω_cond = Bs ./ β_cond                                         # dimensionless rad/t.u.
    k      = [Ω_cond[i] == 0 ? NaN : sF[i]/Ω_cond[i] for i in eachindex(sF)]
    Ωstar  = band_edge(k, Ω_cond; tol=tol)
    Ω_deg  = angvel_degpers(Ω_cond; τ_m=τ_m)
    Ωstar_deg = angvel_degpers(Ωstar; τ_m=τ_m)
    println("  β_cond = ", round(β_cond, digits=4), "   Ω* band edge = ",
            round(Ωstar, digits=3), " rad/t.u. = ", round(Ωstar_deg, digits=0), " deg/s")

    p2 = plot(xlabel="commanded angular velocity Ω  [deg/s]", ylabel="PI gain  k = s/Ω",
              title="Calibrated PI gain (τ_m=$(τ_m) ms),  g0=$g0",
              legend=:bottomleft, xlims=(0, maximum(Ω_deg)), ylims=(0, 1.4),
              left_margin=8mm, bottom_margin=6mm)
    # typical rodent exploratory head angular velocity sits well below the band edge
    vspan!(p2, [0, 300], fc=:steelblue, fa=0.08, lc=:transparent, label="typical behavioral range (≲300 deg/s)")
    hline!(p2, [1.0], lc=:gray, ls=:dot, lw=1, label="unity gain")
    hspan!(p2, [1-tol, 1+tol], fc=:green, fa=0.07, lc=:transparent, label="±$tol band")
    plot!(p2, Ω_deg, k, lc=:black, lw=2, label="conductance field (β_cond)")
    vline!(p2, [Ωstar_deg], lc=:green, ls=:dash, lw=1, label="Ω* = $(round(Int,Ωstar_deg)) deg/s")
    savefig(p2, joinpath("figures", "calib_gain_curve_degpers.png"))
    println("  saved calib_gain_curve_degpers.png")

    # ========================================================================
    # 3. Calibrated HEADING TRACKING — decoded heading [deg] vs time [s], micro/macro
    # ========================================================================
    println("\n== 3. heading tracking (deg vs s) ==")
    T = 200.0
    commands = [
        ("constant turn",  omega_const(0.05),                                          # 0.05 rad/t.u.
            angvel_degpers(0.05; τ_m=τ_m)),
        ("realistic Ω(t)", omega_realistic(T, dt; τ=20.0, σ=0.05, Ωbar=0.0,
                                           rng=Random.MersenneTwister(2)), NaN),
    ]
    panels = Plots.Plot[]
    for (name, Ωfun, rate_deg) in commands
        Bfun = Bfun_from_omega(Ωfun, β_cond)
        tsf, xcf, _ = drive_field_cond(zco, η̄, Δ, κ, xf, synf0, Bfun; T=T, dt=dt)
        println("  [$name] spiking drive (N=$N) …"); flush(stdout)
        tsr, xcr, _ = drive_ring_cond(θco, η, xr, synr0, a_n, n, κ, Bfun; T=T, dt=dt)
        φf = heading_estimate(xcf); φr = heading_estimate(xcr)
        Φ  = commanded_heading(tsf, Ωfun)
        ts_s = time_s(tsf; τ_m=τ_m)
        ttl  = isnan(rate_deg) ? name : "$name  (≈$(round(Int,rate_deg)) deg/s)"
        p = plot(time_s(tsf; τ_m=τ_m), heading_deg(φf .- φf[1]), lc=:black, lw=2, label="field φ",
                 xlabel="time  [s]", ylabel="heading  [deg]", title=ttl, left_margin=9mm, bottom_margin=5mm)
        plot!(p, time_s(tsr; τ_m=τ_m), heading_deg(φr .- φr[1]), lc=:darkred, lw=2, ls=:dash, label="spiking φ")
        plot!(p, ts_s, heading_deg(Φ .- Φ[1]), lc=:green, lw=1.5, ls=:dot, label="commanded ∫Ω dt")
        push!(panels, p)
    end
    fig = plot(panels..., layout=(1, length(panels)), size=(560*length(panels), 400))
    savefig(fig, joinpath("figures", "calib_heading_tracking.png"))
    println("  saved calib_heading_tracking.png")

    # ========================================================================
    # 4. Calibration table — the same readouts at τ_m = 10 and 20 ms, with refs
    # ========================================================================
    println("\n== 4. calibration table (model → physical units) ==")
    drift_rad_tu = 3e-5    # representative N=8192 darkness drift rate (figures/drift.png)
    println("  ┌ quantity ─────────────── model ─── τ_m=10 ms ──── τ_m=20 ms ──── biological ref")
    rowf(label, model, f) = println(rpad("  │ "*label, 33),
        rpad(string(model), 11), rpad(string(round(f(10.0), digits=1)), 14),
        rpad(string(round(f(20.0), digits=1)), 14))
    rowf("peak firing rate [Hz]", round(maximum(rfield),digits=3),
         t -> maximum(rate_hz(rfield; τ_m=t)))
    println(rpad("  │ bump FWHM [deg]", 33), rpad(string(round(fwhm,digits=0)), 11),
            rpad("(τ_m-independent)", 28), " HD tuning ~90°")
    rowf("Ω* unity-gain edge [deg/s]", round(Ωstar,digits=3),
         t -> angvel_degpers(Ωstar; τ_m=t))
    rowf("darkness drift [deg/min]", drift_rad_tu,
         t -> angvel_degpermin(drift_rad_tu; τ_m=t))
    println("  └ refs: HD peak rate ~tens of Hz; HD tuning width ~90°; behavioral head")
    println("         angular velocity ≲ a few hundred deg/s (peak fast turns ~700 deg/s).")

    return (β_cond=β_cond, peak_hz=peak_hz_field, fwhm=fwhm, Ωstar_deg=Ωstar_deg)
end

main()
