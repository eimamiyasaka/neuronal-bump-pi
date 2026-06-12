# Phase-3, Step 3.2: DRIFT in "darkness" (Ω=0) — the heading-memory error.
#
# This model has NO landmark/positional-correction term (modeling note §5), so it is
# always dead-reckoning: the "darkness test" is automatic, and the meaningful
# observable is how much the stored heading WANDERS with no command. We seed a bump,
# hold Ω=0, and measure the centroid displacement over a fixed window.
#
# The field (N→∞, homogeneous kernel, no quenched disorder) is the noise-free
# BASELINE — its bump should sit essentially still. The spiking net carries FROZEN
# Cauchy heterogeneity η, so its bump drifts (toward favourable sites + finite-size
# wander); the drift should SHRINK with N. This is a finite-size phenomenon the field
# structurally cannot show — the spiking model is the measurement, not a check (see
# the micro/macro rationale). The rigorous diffusion-coefficient / adjoint analysis
# is Phase 5; here we just quantify the heading error vs N.
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2.

include("../src/pathint.jl")
include("plotting.jl")
using Plots
using Plots.PlotMeasures

function main()
    Δ  = 0.1
    η̄  = -0.4
    κ  = 2.0
    n  = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt = 0.02

    T_drift = 400.0                 # hold window at Ω=0 (≫ τ_set≈50, so it's settled wander)
    Nx = 256
    Ns = [2048, 4096, 8192]         # spiking sizes to test the 1/√N trend
    Bzero = Bfun_from_omega(omega_const(0.0), 1.0)   # Ω=0 ⇒ B(t)=0 (symmetric kernel)

    xf = field_positions(Nx)

    # --- field baseline (homogeneous; expect ~no drift) ---
    println("== field drift baseline (Ω=0) =="); flush(stdout)
    Khat0 = fft(field_kernel(xf))
    zf0 = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    tsF, xcF, _ = drive_field(zf0, η̄, Δ, κ, xf, Bzero; T=T_drift, dt=dt)
    φF = heading_estimate(xcF); φF .-= φF[1]
    driftF = (φF[end] - φF[1]) / T_drift
    println("  field: net Δφ=", round(φF[end], digits=5), " over T=", T_drift,
            "  → drift rate=", round(driftF, sigdigits=3), " rad/t.u.")

    # --- spiking drift vs N ---
    drift_traces = Vector{Vector{Float64}}()
    drift_ts     = Vector{Vector{Float64}}()
    drift_rates  = Float64[]
    for N in Ns
        println("== spiking drift (N=$N, Ω=0) =="); flush(stdout)
        xr = make_positions(N)
        η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))   # frozen disorder
        K0 = make_kernel(xr)
        θ0 = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)
        tsR, xcR, _ = drive_ring(θ0, η, xr, a_n, n, κ, Bzero; T=T_drift, dt=dt)
        φR = heading_estimate(xcR); φR .-= φR[1]
        rate = (φR[end] - φR[1]) / T_drift
        push!(drift_traces, φR); push!(drift_ts, tsR); push!(drift_rates, rate)
        println("  N=$N: net Δφ=", round(φR[end], digits=5),
                "  → drift rate=", round(rate, sigdigits=3), " rad/t.u.  (|Δφ|/√N·... see trend)")
    end

    # --- trend check: |drift| should fall with N ---
    println("\ndrift |rate| vs N:  ", [(Ns[i], round(abs(drift_rates[i]), sigdigits=3)) for i in eachindex(Ns)])
    println("field baseline |rate|=", round(abs(driftF), sigdigits=3),
            "  (≈0 expected — homogeneous N→∞ limit)")

    open(joinpath("figures", "drift.csv"), "w") do io
        println(io, "N,drift_rate")
        println(io, "field_baseline,", driftF)
        for i in eachindex(Ns); println(io, Ns[i], ",", drift_rates[i]); end
    end

    # --- figure: heading wander traces (left) + |drift rate| vs N (right) ---
    pL = plot(xlabel="t", ylabel="Δ heading φ(t)  (Ω=0)", title="heading wander in darkness",
              legend=:topleft, left_margin=8mm, bottom_margin=6mm)
    plot!(pL, tsF, φF, lc=:black, lw=2, label="field (N→∞)")
    cols = [:orange, :red, :purple]
    for (i, N) in enumerate(Ns)
        plot!(pL, drift_ts[i], drift_traces[i], lc=cols[i], lw=1.5, label="spiking N=$N")
    end

    pR = plot(Ns, abs.(drift_rates), m=:circle, ms=6, lc=:red, mc=:red,
              xlabel="N", ylabel="|drift rate|  (rad/t.u.)", title="finite-size drift vs N",
              xscale=:log10, yscale=:log10, legend=false, left_margin=10mm, bottom_margin=6mm)

    fig = plot(pL, pR, layout=(1, 2), size=(1200, 450))
    display(fig)
    savefig(fig, joinpath("figures", "drift.png"))
    println("saved drift.png + drift.csv")
    return nothing
end

main()
