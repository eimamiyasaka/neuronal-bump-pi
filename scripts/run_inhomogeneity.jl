# Phase-3, Step 3.2: PI gain vs POSITION — the Secer et al. (2024) spatial-
# inhomogeneity test. Secer show that in classical bump attractors the PI gain is
# NOT a single global constant but varies around the ring. We test the same here.
#
# Method: drive a constant Ω in the near-unity band so the bump laps the ring many
# times; bin the instantaneous bump speed by the bump's CURRENT position, average per
# bin, divide by Ω → k_local(x). The field (homogeneous kernel, no quenched disorder)
# is the flat BASELINE — any structure there would be a measurement artifact. The
# spiking net carries FROZEN Cauchy η, so different ring sectors have different local
# excitability and the bump speeds up / slows down as it passes — a finite-size gain
# inhomogeneity the field cannot show (the spiking model is the measurement here).
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2.

include("../src/pathint.jl")
include("plotting.jl")
using Plots
using Plots.PlotMeasures

# Bin a per-sample quantity v by the per-sample ring position pos∈[0,2π) into nbins,
# returning the bin centres and the per-bin mean (NaN for empty bins). Samples before
# `tskip` are dropped (settling transient).
function bin_by_position(ts, pos, v, nbins; tskip=60.0)
    centres = [(b - 0.5) * 2π / nbins for b in 1:nbins]
    acc = zeros(nbins); cnt = zeros(Int, nbins)
    for i in eachindex(ts)
        ts[i] < tskip && continue
        b = clamp(floor(Int, mod(pos[i], 2π) / (2π / nbins)) + 1, 1, nbins)
        acc[b] += v[i]; cnt[b] += 1
    end
    means = [cnt[b] > 0 ? acc[b] / cnt[b] : NaN for b in 1:nbins]
    return centres, means
end

function main()
    Δ  = 0.1
    η̄  = -0.4
    κ  = 2.0
    n  = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt = 0.02
    β  = 1 / 3.91

    Ω0 = 0.10                       # constant command in the near-unity band (k≈1)
    T  = 800.0                      # ~12 laps at s≈0.1 (lap time 2π/0.1≈63) for bin stats
    nbins = 24
    Nx = 256
    N  = 8192
    Bfun = Bfun_from_omega(omega_const(Ω0), β)

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    # --- field: homogeneous baseline ---
    println("== field k(x) baseline (Ω=$Ω0, T=$T) =="); flush(stdout)
    Khat0 = fft(field_kernel(xf))
    zf0 = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    tsF, xcF, _ = drive_field(zf0, η̄, Δ, κ, xf, Bfun; T=T, dt=dt)
    vF = instantaneous_speed(tsF, xcF; window=2.0)
    cxF, kF = bin_by_position(tsF, xcF, vF ./ Ω0, nbins)

    # --- spiking: frozen-disorder inhomogeneity ---
    println("== spiking k(x) (N=$N, Ω=$Ω0, T=$T) =="); flush(stdout)
    K0 = make_kernel(xr)
    θ0 = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)
    tsR, xcR, _ = drive_ring(θ0, η, xr, a_n, n, κ, Bfun; T=T, dt=dt)
    vR = instantaneous_speed(tsR, xcR; window=2.0)
    cxR, kR = bin_by_position(tsR, xcR, vR ./ Ω0, nbins)

    # --- summary: spread of k around the ring ---
    spreadF = maximum(filter(isfinite, kF)) - minimum(filter(isfinite, kF))
    spreadR = maximum(filter(isfinite, kR)) - minimum(filter(isfinite, kR))
    println("\nk(x) spread (max−min over ring):  field=", round(spreadF, digits=4),
            " (≈0 expected)   spiking=", round(spreadR, digits=4))
    println("k(x) mean:  field=", round(sum(filter(isfinite, kF))/count(isfinite, kF), digits=4),
            "   spiking=", round(sum(filter(isfinite, kR))/count(isfinite, kR), digits=4))

    open(joinpath("figures", "gain_position.csv"), "w") do io
        println(io, "x,k_field,k_spiking")
        for b in 1:nbins; println(io, cxR[b], ",", kF[b], ",", kR[b]); end
    end

    xt = pi_ticks(2π)
    p = plot(xlabel="bump position x", ylabel="k_local = s(x)/Ω",
             title="PI gain vs position  (Secer test, Ω=$Ω0)",
             legend=:topright, xticks=xt, ylims=(0.7, 1.3), left_margin=8mm, bottom_margin=6mm)
    hline!(p, [1.0], lc=:gray, ls=:dot, label="unity")
    plot!(p, cxF, kF, lc=:black, lw=2, m=:circle, ms=3, label="field (homogeneous)")
    plot!(p, cxR, kR, lc=:red, lw=2, m=:square, ms=3, label="spiking N=$N (frozen η)")
    savefig(p, joinpath("figures", "gain_position.png"))
    println("saved gain_position.png + gain_position.csv")
    return nothing
end

main()
