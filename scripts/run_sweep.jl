# Phase-2 gate: lateral bump speed s versus asymmetry B (Laing & Omel'chenko 2020,
# Fig. 3 — the target figure, notes/image3.png). A motionless bump (B = 0) is
# continued forward to B = 0.2 and back, on BOTH the macroscopic field and the
# microscopic spiking net; the speed of the settled travelling wave is measured at
# each B. The diagnostic signature to reproduce is the NON-MONOTONIC, HYSTERETIC
# s(B): the forward and backward sweeps trace different branches because moving
# bumps are multistable (the bifurcation scenario Laing analyses by continuation).
#
# On top of the direct-simulation MARKERS, the field travelling-wave branch is traced
# analytically by Newton continuation (src/continuation.jl) — the solid curve of
# Fig. 3. Three figures are produced: field only, spiking only, and the two overlaid.
#
# Operating point: Δ = 0.1, κ = 2, η̄ = −0.4, n = 2  (Laing Fig. 3).
#
# COST. The cost knobs below default LIGHT so the script returns in minutes; raise
# them toward the paper-faithful values (noted inline) for a publication-quality
# curve. The spiking sweep is the expensive side (N neurons × steps × |Bs|), so it
# uses a coarser B-grid than the cheap 256-point field.

include("../src/moving.jl")        # bsweep_field, bsweep_ring, seed_field_bump, seed_ring_bump
include("../src/continuation.jl")  # tw_continuation — the field travelling-wave s(B) curve
include("plotting.jl")             # (shared presentation helpers; kept consistent across scripts)
using Plots
using Plots.PlotMeasures

function main()
    # --- fixed moving-bump operating point ---
    Δ  = 0.1
    η̄  = -0.4
    κ  = 2.0
    n  = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt = 0.02                      # Laing's integrator step

    # --- cost knobs (LIGHT defaults; paper-faithful values in comments) ---
    # Each sweep WARM-STARTS from the previous B's settled state (as the paper does
    # too — the forward/backward hysteresis only exists because the bump is carried
    # over between B values), so a single ΔB step is a tiny perturbation that
    # re-settles within ~5/Δ ≈ 50 t.u. at Δ=0.1. Hence T_sweep can be short: the
    # only Cut here vs the paper is the per-step integration time (100 vs its 2000),
    # not the warm-start. The paper's 2000 is conservative — robustness near folds
    # and where s≈0 — not a structural need. The measurement window (last frac·T_sweep)
    # still spans many bump traversals (s·50 ≈ several full laps), so s is well sampled.
    ΔB_field = 0.005               # field B-grid step          (paper: 0.001)
    ΔB_ring  = 0.01                # spiking B-grid step (coarser; expensive side)
    T_sweep  = 100.0               # integrate per B (warm-started) (paper: 2000)
    frac     = 0.5                 # measure speed over last frac of T_sweep
    Nx = 256                       # field grid points
    N  = 8192                      # spiking neurons
    rep = 5                        # print progress every `rep` B-values (flushed)

    Bmax = 0.2
    Bs_field = collect(0.0:ΔB_field:Bmax)
    Bs_ring  = collect(0.0:ΔB_ring:Bmax)

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    # =============== FIELD (macroscopic) sweep ===============
    println("== field sweep (", length(Bs_field), " B-values × 2 directions) =="); flush(stdout)
    Khat0 = fft(field_kernel(xf))                              # B = 0
    zseed = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)         # motionless B=0 bump
    # forward 0 → Bmax, then backward Bmax → 0 continuing from the forward endpoint
    sF_fwd, zEnd = bsweep_field(Bs_field,          xf, η̄, Δ, κ, zseed; T=T_sweep, dt=dt, frac=frac, tag="field↑", every=rep)
    sF_bwd, _    = bsweep_field(reverse(Bs_field), xf, η̄, Δ, κ, zEnd;  T=T_sweep, dt=dt, frac=frac, tag="field↓", every=rep)
    sF_bwd = reverse(sF_bwd)                                   # realign to ascending Bs_field

    # =============== RING (microscopic) sweep ===============
    println("== spiking sweep (N=", N, ", ", length(Bs_ring), " B-values × 2 directions) =="); flush(stdout)
    K0   = make_kernel(xr)                                     # B = 0
    θseed = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)
    sR_fwd, θEnd = bsweep_ring(Bs_ring,          xr, η, a_n, n, κ, θseed; T=T_sweep, dt=dt, frac=frac, tag="ring↑", every=rep)
    sR_bwd, _    = bsweep_ring(reverse(Bs_ring), xr, η, a_n, n, κ, θEnd;  T=T_sweep, dt=dt, frac=frac, tag="ring↓", every=rep)
    sR_bwd = reverse(sR_bwd)

    # --- self-validation prints ---
    # (i) speeds finite; (ii) small-B slope near the classical guide s ≈ B/0.3 (the
    # B = 0 traveling-front speed for kernel (4), Laing's remark after Eq. 11); (iii)
    # a measurable forward/backward hysteresis gap somewhere in [0, Bmax].
    @assert all(isfinite, sF_fwd) && all(isfinite, sR_fwd) "non-finite speeds — check settling"
    i_lo = 2                                                   # first nonzero-B sample
    println("\nsmall-B slope check  (classical guide s ≈ B/0.3):")
    println("  field   B=", Bs_field[i_lo], "  s=", round(sF_fwd[i_lo], digits=4),
            "   guide=", round(Bs_field[i_lo] / 0.3, digits=4))
    println("  spiking B=", Bs_ring[i_lo],  "  s=", round(sR_fwd[i_lo], digits=4),
            "   guide=", round(Bs_ring[i_lo] / 0.3, digits=4))
    gapF = maximum(abs.(sF_fwd .- sF_bwd))
    gapR = maximum(abs.(sR_fwd .- sR_bwd))
    println("hysteresis gap (max |forward − backward|):  field=", round(gapF, digits=4),
            "   spiking=", round(gapR, digits=4),
            gapF > 0.02 ? "   → hysteresis present" : "   → no hysteresis detected (raise T_sweep / refine ΔB)")
    # (iv) micro/macro closure: the spiking B-grid (ΔB_ring) is a subset of the field
    # grid (ΔB_field) when ΔB_ring is an integer multiple, so compare on the common B.
    if isinteger(round(ΔB_ring / ΔB_field; digits=6))
        step = round(Int, ΔB_ring / ΔB_field)
        dmicromacro = maximum(abs.(sR_fwd .- sF_fwd[1:step:end]))
        println("micro/macro speed agreement (max |spiking − field|, forward sweep): ",
                round(dmicromacro, digits=4), "  → closure", dmicromacro < 0.05 ? " OK" : " LOOSE (check settling)")
    end

    # =============== FIELD travelling-wave CONTINUATION (the solid curve) ===========
    # Newton + PSEUDO-ARCLENGTH continuation of the exact travelling-wave equation — the
    # analytic branch the markers only sample. From a SINGLE seed it follows one connected
    # branch through both folds (the B-fold and the knot at B≈0.1, s≈0.85), so the curve
    # is CONTINUOUS — the solid curve of Laing Fig. 3, no NaN gap (see src/continuation.jl).
    println("\n== field travelling-wave continuation =="); flush(stdout)
    Bc, sc = tw_continuation(Nx, η̄, Δ, κ)   # defaults: Δarc=0.02, wprof=0.01
    # persist the continuation curve as data (NaN row separates the two branches);
    # scripts/digitize_fig3.py compares this against the digitized Laing Fig. 3.
    open(joinpath("figures", "continuation_bs.csv"), "w") do io
        println(io, "B,s")
        for i in eachindex(Bc); println(io, Bc[i], ",", sc[i]); end
    end

    # ---- FIGURES: s vs B (recreates Laing Fig. 3 / notes/image3.png) ----
    # Forward sweep = crosses, backward = circles (the paper's marker convention);
    # solid black = field continuation. Three figures: field only, spiking only, and
    # the two overlaid. `add_continuation!` and the marker helpers keep them identical.
    addcont!(p) = plot!(p, Bc, sc, lc=:black, lw=2, label="field continuation")
    base(title) = plot(xlabel="B  (asymmetry)", ylabel="s  (lateral speed)", title=title,
                       legend=:topleft, xlims=(0, Bmax), ylims=(0, 1.35),
                       left_margin=8mm, bottom_margin=6mm)

    p_field = base("moving-bump speed s(B) — field")
    scatter!(p_field, Bs_field, sF_fwd, m=:xcross, mc=:red,  ms=3, label="field, forward")
    scatter!(p_field, Bs_field, sF_bwd, m=:circle, mc=:blue, ms=3, label="field, backward")
    addcont!(p_field)
    savefig(p_field, joinpath("figures", "sweep_speed_field.png"))

    p_ring = base("moving-bump speed s(B) — spiking")
    scatter!(p_ring, Bs_ring, sR_fwd, m=:xcross, mc=:darkred, ms=4, label="spiking, forward")
    scatter!(p_ring, Bs_ring, sR_bwd, m=:circle, mc=:navy,    ms=4, label="spiking, backward")
    addcont!(p_ring)
    savefig(p_ring, joinpath("figures", "sweep_speed_spiking.png"))

    p_all = base("moving-bump speed s(B) — field vs spiking")
    scatter!(p_all, Bs_field, sF_fwd, m=:xcross, mc=:red,  ms=3, label="field, forward")
    scatter!(p_all, Bs_field, sF_bwd, m=:circle, mc=:blue, ms=3, label="field, backward")
    scatter!(p_all, Bs_ring,  sR_fwd, m=:xcross, mc=:darkred, ms=4, label="spiking, forward")
    scatter!(p_all, Bs_ring,  sR_bwd, m=:circle, mc=:navy,    ms=4, label="spiking, backward")
    addcont!(p_all)
    savefig(p_all, joinpath("figures", "sweep_speed.png"))

    println("saved sweep_speed_field.png, sweep_speed_spiking.png, sweep_speed.png")
    return nothing
end

main()
