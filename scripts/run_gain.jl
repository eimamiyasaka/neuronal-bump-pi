# Phase-3, Step 3.2: PI GAIN curve k(Ω) = (bump speed)/(commanded velocity).
#
# The quasi-static gain (modeling note §3a): for a command varying slowly vs the
# settling time τ_set≈50, the bump speed equals the steady travelling-wave value
# s(B) at B=β·Ω, so k(Ω)=s(βΩ)/Ω. This is exactly the Phase-2 s(B) curve recast
# against commanded velocity — so we REUSE the validated warm-started sweeps
# (bsweep_field/bsweep_ring) and the analytic continuation (tw_continuation), then
# divide by Ω. The calibration β=1/3.91 is chosen so k→1 as Ω→0 (note §2).
#
# What this answers (Hinge B / the Step-3.3 gate input): the velocity band over which
# k stays near 1, and where the non-monotonic / multistable s(B) structure ends it —
# read off as Ω* (where the forward gain leaves |k−1|<tol) and the forward/backward
# hysteresis in gain. Both field and spiking (the recurring closure gate).
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2. B∈[0,0.2] ⇒ Ω=B/β∈[0,0.78].

include("../src/pathint.jl")        # (pulls moving.jl: bsweep_field/ring, seed_*_bump)
include("../src/continuation.jl")   # tw_continuation — analytic s(B) backbone
include("plotting.jl")
using Plots
using Plots.PlotMeasures

function main()
    # --- operating point + calibration ---
    Δ  = 0.1
    η̄  = -0.4
    κ  = 2.0
    n  = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt = 0.02
    β  = 1 / 3.91
    tol = 0.10                       # gain tolerance for the "near-unity" band |k−1|<tol

    # --- cost knobs (LIGHT, as run_sweep.jl; warm-start makes T_sweep short) ---
    ΔB_field = 0.005                 # field B-grid step
    ΔB_ring  = 0.01                  # spiking B-grid step (coarser; expensive side)
    T_sweep  = 100.0                 # integrate per B (warm-started)
    frac     = 0.5
    Nx = 256
    N  = 8192
    rep = 5

    Bmax = 0.2
    Bs_field = collect(0.0:ΔB_field:Bmax)
    Bs_ring  = collect(0.0:ΔB_ring:Bmax)
    Ω_field  = Bs_field ./ β         # commanded velocity for each B
    Ω_ring   = Bs_ring  ./ β

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    # =============== quasi-static speed sweeps (reuse Phase-2 machinery) ===============
    println("== field gain sweep (", length(Bs_field), " Ω-values × 2 dir) =="); flush(stdout)
    Khat0 = fft(field_kernel(xf))
    zseed = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    sF_fwd, zEnd = bsweep_field(Bs_field,          xf, η̄, Δ, κ, zseed; T=T_sweep, dt=dt, frac=frac, tag="field↑", every=rep)
    sF_bwd, _    = bsweep_field(reverse(Bs_field), xf, η̄, Δ, κ, zEnd;  T=T_sweep, dt=dt, frac=frac, tag="field↓", every=rep)
    sF_bwd = reverse(sF_bwd)

    println("== spiking gain sweep (N=", N, ", ", length(Bs_ring), " Ω-values × 2 dir) =="); flush(stdout)
    K0    = make_kernel(xr)
    θseed = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)
    sR_fwd, θEnd = bsweep_ring(Bs_ring,          xr, η, a_n, n, κ, θseed; T=T_sweep, dt=dt, frac=frac, tag="ring↑", every=rep)
    sR_bwd, _    = bsweep_ring(reverse(Bs_ring), xr, η, a_n, n, κ, θEnd;  T=T_sweep, dt=dt, frac=frac, tag="ring↓", every=rep)
    sR_bwd = reverse(sR_bwd)

    # --- gain k = s/Ω (Ω=0 point undefined → NaN; uses the B=0 entry's index 1) ---
    gain(s, Ω) = [Ω[i] == 0 ? NaN : s[i]/Ω[i] for i in eachindex(s)]
    kF_fwd = gain(sF_fwd, Ω_field); kF_bwd = gain(sF_bwd, Ω_field)
    kR_fwd = gain(sR_fwd, Ω_ring);  kR_bwd = gain(sR_bwd, Ω_ring)

    # --- analytic backbone: continuation s(B) → k(Ω) (multivalued through the folds) ---
    println("\n== field gain continuation =="); flush(stdout)
    Bc, sc = tw_continuation(Nx, η̄, Δ, κ)
    Ωc = Bc ./ β
    kc = [isnan(Bc[i]) || Bc[i] == 0 ? NaN : sc[i]/(Bc[i]/β) for i in eachindex(Bc)]

    # =============== readouts for the Step-3.3 gate ===============
    # Ω* = first commanded velocity where the forward gain leaves the |k−1|<tol band.
    band_edge(k, Ω) = begin
        idx = findfirst(i -> !isnan(k[i]) && abs(k[i] - 1) > tol, eachindex(k))
        idx === nothing ? (Ω[end], false) : (Ω[idx], true)
    end
    ΩstarF, leftF = band_edge(kF_fwd, Ω_field)
    ΩstarR, leftR = band_edge(kR_fwd, Ω_ring)
    @assert all(isfinite, sF_fwd) && all(isfinite, sR_fwd) "non-finite speeds — check settling"
    println("\nsmall-Ω gain (unity-gain check, should be ≈1):")
    println("  field   Ω=", round(Ω_field[2], digits=4), "  k=", round(kF_fwd[2], digits=4))
    println("  spiking Ω=", round(Ω_ring[2],  digits=4), "  k=", round(kR_fwd[2], digits=4))
    println("near-unity band |k−1|<", tol, ":  field Ω*=", round(ΩstarF, digits=4),
            leftF ? "" : " (never left band in range)",
            "   spiking Ω*=", round(ΩstarR, digits=4), leftR ? "" : " (never left band)")
    gapF = maximum(filter(isfinite, abs.(kF_fwd .- kF_bwd)))
    println("gain hysteresis (max|k_fwd−k_bwd|):  field=", round(gapF, digits=4))

    # --- persist gain data ---
    open(joinpath("figures", "gain_curve.csv"), "w") do io
        println(io, "Omega,B,kF_fwd,kF_bwd")
        for i in eachindex(Ω_field)
            println(io, Ω_field[i], ",", Bs_field[i], ",", kF_fwd[i], ",", kF_bwd[i])
        end
    end

    # =============== figure: k(Ω) ===============
    p = plot(xlabel="Ω  (commanded angular velocity)", ylabel="k = s/Ω  (PI gain)",
             title="Phase-3 PI gain  k(Ω)   (β=1/3.91, Δ=0.1)",
             legend=:bottomleft, xlims=(0, Ω_field[end]), ylims=(0, 1.4),
             left_margin=8mm, bottom_margin=6mm)
    hline!(p, [1.0], lc=:gray, ls=:dot, lw=1, label="unity gain")
    hspan!(p, [1-tol, 1+tol], fc=:green, fa=0.07, lc=:transparent, label="±$tol band")
    plot!(p, Ωc, kc, lc=:black, lw=2, label="field continuation")
    scatter!(p, Ω_field, kF_fwd, m=:xcross, mc=:red,  ms=3, label="field, forward")
    scatter!(p, Ω_field, kF_bwd, m=:circle, mc=:blue, ms=3, label="field, backward")
    scatter!(p, Ω_ring,  kR_fwd, m=:xcross, mc=:darkred, ms=4, label="spiking, forward")
    scatter!(p, Ω_ring,  kR_bwd, m=:circle, mc=:navy,    ms=4, label="spiking, backward")
    vline!(p, [ΩstarF], lc=:green, ls=:dash, lw=1, label="Ω* (band edge)")
    savefig(p, joinpath("figures", "gain_curve.png"))
    println("saved gain_curve.png + gain_curve.csv")
    return nothing
end

main()
