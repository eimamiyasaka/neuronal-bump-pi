# Phase-5, Step 5.2 — drift quantification, and a finding about its scaling.
#
# In darkness (Ω=0, B=0) the EXACT field bump is translation-marginal and does NOT drift
# (deterministic control). The finite-N spiking bump wanders because the frozen-η realization
# breaks the ring symmetry. The TWIST (and the actual result): η is drawn from the Cauchy /
# Lorentzian distribution — the very distribution that makes the Ott–Antonsen reduction exact
# — which is HEAVY-TAILED with no finite variance. Its extreme values GROW with N
# (η_extreme ≈ η̄ + Δ·2N/π), so the symmetry-breaking force is dominated by rare
# extreme-excitability neurons rather than a clean CLT sum. Consequently the darkness drift is
# large, strongly realization-dependent, and does NOT follow the light-tailed-disorder 1/√N
# finite-size law: the per-seed drift rate does not concentrate, and even its median does not
# fall cleanly with N over accessible windows. We REPORT this honestly (scatter + median, no
# spurious fit) rather than impose a scaling the data does not support.
#
# This is a genuine tension worth stating: the Lorentzian is required for exact closure, yet
# its heavy tails make finite-size drift outlier-dominated. The rigorous, per-realization drift
# rate is exactly what the Nakao (2014) adjoint / phase-sensitivity method computes (from the
# bump's Goldstone mode against the realized quenched potential) — the recommended Step-5.2
# tool, DEFERRED here (step5_gate.md) precisely because ensemble scaling is the wrong frame.
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2, conductance g0=0.1 (matched reversals E=±20).
# Cost: spiking, multiple N × seeds — run in background.

include("../src/phase5.jl")        # drift_rate_stats (+ conductance, core)
using Plots
using Plots.PlotMeasures

# extreme excitability in the Cauchy quantile set (j=N): η̄ + Δ·tan(π·((2N−1)/(2N) − 1/2)).
extreme_eta(N, η̄, Δ) = η̄ + Δ * tan(π * ((2N - 1) / (2N) - 0.5))
median_(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2)

function main()
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt, g0 = 0.02, 0.1
    E_E, E_I = matched_reversals(κ, g0)
    Ns    = [2048, 4096, 8192]
    nseed = 6
    T_track = 300.0

    # --- deterministic control: exact field B=0 bump does not drift ---
    Nx = 256; xf = field_positions(Nx)
    synf = dale_field(xf, E_E, E_I; g0=g0)
    zf   = seed_field_bump_cond(xf, η̄, Δ, κ, synf; dt=dt)
    xcf, _ = track_field_centroid_cond(zf, η̄, Δ, κ, synf, cos.(xf), sin.(xf); T=T_track, dt=dt)
    φf = unwrap(xcf) .- xcf[1]
    println("== Step 5.2 drift ==")
    println("  field (N→∞) control: max|Δφ| over T=$T_track = ", round(maximum(abs.(φf)), sigdigits=3),
            "  (≈0 ⇒ no deterministic field drift, as expected)")

    nt = round(Int, T_track/dt); ts = collect(1:nt) .* dt
    all_rates = Dict{Int,Vector{Float64}}()
    traces    = Dict{Int,Vector{Vector{Float64}}}()
    for N in Ns
        xr = make_positions(N)
        synr = dale_ring(xr, E_E, E_I; g0=g0)
        φ_seeds = Vector{Vector{Float64}}()
        for s in 1:nseed
            η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(100 + s))
            θ0 = seed_ring_bump_cond(η, synr, a_n, n, κ, xr; T_free=150.0, dt=dt)
            xc, _ = track_ring_centroid_cond(θ0, η, synr, a_n, n, κ; T=T_track, dt=dt)
            push!(φ_seeds, unwrap(xc) .- xc[1])
            print("  N=$N seed $s/$nseed done\r"); flush(stdout)
        end
        st = drift_rate_stats(φ_seeds, ts)
        all_rates[N] = st.rates
        traces[N]    = φ_seeds[1:min(3, nseed)]
        println("\n  N=", rpad(N,6), " |η|_extreme≈", round(abs(extreme_eta(N, η̄, Δ)), digits=0),
                "   median|rate|=", round(median_(abs.(st.rates)), sigdigits=3),
                "   spread[min,max]|rate|=[", round(minimum(abs.(st.rates)), sigdigits=2), ", ",
                round(maximum(abs.(st.rates)), sigdigits=2), "]")
    end
    medians = [median_(abs.(all_rates[N])) for N in Ns]
    println("  ⇒ median|drift| does NOT fall ~1/√N (", [round(m, sigdigits=2) for m in medians],
            "); drift is heavy-tail (Cauchy) outlier-dominated, realization-specific.")

    # ------------------------------------------------------------------ figure
    pa = plot(xlabel="time  [t.u.]", ylabel="Δ heading φ(t)  [rad]",
              title="(a) darkness wander (3 seeds per N)", legend=:topleft,
              left_margin=9mm, bottom_margin=5mm)
    plot!(pa, ts, φf, lc=:black, lw=2.5, label="field N→∞ (no drift)")
    for (N, c) in zip(Ns, (:orange, :red, :purple))
        for (j, tr) in enumerate(traces[N])
            plot!(pa, ts, tr, lc=c, lw=1, label=(j == 1 ? "spiking N=$N" : ""))
        end
    end

    # per-seed |drift rate| scatter vs N (NO 1/√N fit — the data does not support one)
    pb = plot(xscale=:log10, yscale=:log10, xlabel="N (log)", ylabel="|drift rate|  [rad/t.u.]  (log)",
              title="(b) drift is outlier-dominated, not ~1/√N", legend=:topleft,
              xlims=(1500, 11000), left_margin=10mm, bottom_margin=5mm)
    for (k, N) in enumerate(Ns)
        r = abs.(all_rates[N])
        scatter!(pb, fill(N, length(r)), r, mc=:gray, ms=5, ma=0.7, label=(k == 1 ? "per-seed" : ""))
    end
    plot!(pb, Ns, medians, m=:diamond, lc=:purple, mc=:purple, lw=2, ms=7, label="median")
    fig = plot(pa, pb, layout=(1,2), size=(1180, 460))
    savefig(fig, joinpath("figures", "phase5_drift.png"))
    println("  saved figures/phase5_drift.png")

    open(joinpath("figures", "phase5_drift.csv"), "w") do io
        println(io, "N,eta_extreme,median_abs_rate,seed_rates...")
        for N in Ns
            println(io, N, ",", extreme_eta(N, η̄, Δ), ",", median_(abs.(all_rates[N])), ",",
                    join(all_rates[N], ","))
        end
    end
    println("  saved figures/phase5_drift.csv")
    return (Ns=Ns, medians=medians)
end

main()
