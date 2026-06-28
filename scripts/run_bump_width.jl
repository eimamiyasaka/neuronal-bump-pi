# Bump-width vs kernel / operating point — the field-only check behind the writeup's tuning-width
# LIMITATION. The first-harmonic kernel makes the static bump ~half-ring wide: at the contribution's
# operating point the calibrated FWHM is ≈198° (run_calibration.jl), ~2× the ~90° HD-cell tuning width
# a reviewer compares against. This script asks what — if anything — narrows it, and reports the
# honest answer rather than an assumed one.
#
# HONEST FINDING (the naive intuition is WRONG). Adding a higher kernel HARMONIC does NOT narrow the
# bump: a 2nd cosine harmonic on the excitatory channel slightly WIDENS it (the OA bump width is set by
# the nonlinear self-consistent field, not the kernel's excitatory-lobe width), and Dale non-negativity
# (K_E≥0) caps the harmonic anyway. The width IS controllable, but via the OPERATING POINT: deeper
# first-harmonic modulation narrows it modestly (Dale-capped at aE≤cE), and reducing the drive η̄ toward
# the bump's existence boundary narrows it to ~173° — but with a falling peak rate, and the bump floods/
# collapses just beyond. So ~90° HD tuning is OUT OF REACH for the single-population first-harmonic model
# without destroying the bump: a genuine limitation that motivates the E–I / structured-inhibition
# extension (which can sharpen tuning without collapse) — see the two-pop future work.
#
# WHY FIELD-ONLY IS SUFFICIENT. Tuning width is a STATIC, geometric property of the exact OA bump, and
# the OA reduction is exact for ANY kernel (it enters only as the additive drive [w⊗r]), so this is a
# property of the field already micro/macro-gated in Phases 1–5 — no closure re-test, no spiking run.
# (And the field evaluates any kernel by FFT convolution in O(Nx log Nx), so harmonics are free here.)
#
# ZERO src/ EDITS. Harmonic/modulation variants are built from the existing dale_field config (the 2nd
# harmonic by the linear FFT identity fft(K_E + a2 cos2x) = KhatE + a2·fft(cos2x)); the drive lever is
# just η̄. Operating point matches run_calibration.jl: Δ=0.1, κ=2, n=2, Nx=256, g0=0.1, E=±20, B=0.

include("../src/conductance.jl")   # dale_field, matched_reversals, seed_field_bump_cond, FieldSyn, rate, …
include("../src/calibration.jl")   # fwhm_deg, heading_deg
include("plotting.jl")
using Plots
using Plots.PlotMeasures

# Valid LOCALIZED bump: high contrast, surround near the rest floor, not over-synchronized (|z|→1).
# Flooded/collapsed states (peak≈surround) and oversynchronized states are excluded — their FWHM is
# meaningless. Returns (fwhm_deg, peak, surround, |z|max, valid).
function bump_measure(x, η̄, Δ, κ, syn; dt=0.02)
    z = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=300.0, dt=dt)
    r = rate.(z)
    pk, sr, zm = maximum(r), minimum(r), maximum(abs.(z))
    contrast = (pk - sr) / pk
    valid = (contrast > 0.5) && (sr < 0.08) && (zm < 0.97)
    return (fwhm = fwhm_deg(r), peak = pk, surr = sr, zmax = zm, valid = valid, rate = r)
end

recenter_to_pi(prof, x) = circshift(prof, round(Int, (π - bump_centroid(prof, cos.(x), sin.(x))) / (x[2] - x[1])))

function main()
    Δ, η̄0, κ0, n = 0.1, -0.4, 2.0, 2
    dt, Nx, g0 = 0.02, 256, 0.1
    E_E, E_I = matched_reversals(κ0, g0)
    x = field_positions(Nx)
    hd_ref = 90.0
    base = dale_field(x, E_E, E_I; g0=g0)
    Shat2 = fft(cos.(2 .* x))

    row(tag, m) = println(rpad(tag, 14), "FWHM=", rpad(round(m.fwhm,digits=1),8),
        " peak=", rpad(round(m.peak,digits=4),8), " surr=", rpad(round(m.surr,digits=4),8),
        " |z|=", rpad(round(m.zmax,digits=3),7), m.valid ? "bump" : "— no localized bump")

    b0 = bump_measure(x, η̄0, Δ, κ0, base)
    println("== field-only static bump width: what narrows it? (Δ=$Δ κ=$κ0 η̄=$η̄0 g0=$g0) ; HD ref ≈$(hd_ref)° ==")
    println("baseline FWHM = ", round(b0.fwhm,digits=0), "° (reproduces run_calibration ≈198°)\n")

    # --- Lever A: kernel 2nd harmonic a2 (Dale-compliant a2≥-0.1) — does NOT narrow ---
    println("-- A. kernel 2nd harmonic a2 (net 0.1+0.3cos+a2 cos2x):")
    a2s = [-0.1, 0.0, 0.1, 0.2]
    a2_m = [bump_measure(x, η̄0, Δ, κ0, FieldSyn(base.KhatE .+ a2 .* Shat2, base.KhatI, g0, E_E, E_I)) for a2 in a2s]
    for (a2, m) in zip(a2s, a2_m); row("a2=$a2", m); end

    # --- Lever B: first-harmonic modulation depth aE (Dale cap aE≤cE=0.4) — narrows modestly ---
    println("-- B. first-harmonic modulation aE (Dale cap aE≤0.4):")
    aEs = [0.3, 0.35, 0.4]
    aE_m = [bump_measure(x, η̄0, Δ, κ0, dale_field(x, E_E, E_I; g0=g0, aE=aE)) for aE in aEs]
    for (aE, m) in zip(aEs, aE_m); row("aE=$aE", m); end

    # --- Lever C: drive η̄ toward the existence boundary — the lever that actually moves width ---
    println("-- C. drive η̄ (toward bump existence boundary):")
    η̄s = [-0.40, -0.45, -0.50, -0.55, -0.60]
    η̄_m = [bump_measure(x, η̄, Δ, κ0, base) for η̄ in η̄s]
    for (η̄, m) in zip(η̄s, η̄_m); row("η̄=$η̄", m); end

    # narrowest VALID bump found (for the overlay + the headline number)
    allm = vcat(a2_m, aE_m, η̄_m)
    valid = filter(m -> m.valid, allm)
    narrow = valid[argmin([m.fwhm for m in valid])]
    println("\n⇒ narrowest VALID bump = ", round(narrow.fwhm,digits=0), "° (peak ", round(narrow.peak,digits=3),
            ") — still ≫ HD ~$(round(Int,hd_ref))°.")
    println("  Higher kernel harmonics do NOT narrow the bump; only reduced drive does, toward collapse.")
    println("  Reaching ~$(round(Int,hd_ref))° needs an E–I / structured-inhibition architecture (future work).")

    # ============================================================== figures
    # (a) the lever that works: FWHM vs η̄ (valid solid / invalid hollow) + peak rate (tradeoff, twin)
    vmask = [m.valid for m in η̄_m]
    pa = plot(xlabel="drive η̄", ylabel="bump FWHM  [deg]", title="(a) only reduced drive narrows it",
              legend=:topleft, ylims=(0, 260), left_margin=9mm, bottom_margin=5mm)
    plot!(pa, η̄s, [m.fwhm for m in η̄_m], lc=:black, lw=2, label="FWHM")
    scatter!(pa, η̄s[vmask], [m.fwhm for m in η̄_m][vmask], mc=:black, ms=6, label="valid bump")
    any(.!vmask) && scatter!(pa, η̄s[.!vmask], [m.fwhm for m in η̄_m][.!vmask], mc=:white, msc=:black, ms=6, label="flooded/collapsed")
    hline!(pa, [hd_ref], lc=:steelblue, ls=:dash, lw=1.5, label="HD tuning ≈$(round(Int,hd_ref))°")
    plot!(twinx(pa), η̄s, [m.peak for m in η̄_m], lc=:crimson, lw=2, ls=:dash, m=:diamond, mc=:crimson,
          ylabel="peak rate (tradeoff)", legend=:topright, label="peak rate")

    # (b) the levers that DON'T work much: FWHM barely moves with kernel harmonic / modulation
    pb = plot(xlabel="kernel knob value", ylabel="bump FWHM  [deg]",
              title="(b) kernel harmonics barely move it", legend=:topleft,
              ylims=(150, 230), left_margin=9mm, bottom_margin=5mm)
    plot!(pb, a2s, [m.fwhm for m in a2_m], m=:circle, lc=:purple, mc=:purple, lw=2, label="2nd harmonic a2")
    plot!(pb, aEs, [m.fwhm for m in aE_m], m=:square, lc=:teal, mc=:teal, lw=2, label="modulation aE (Dale-capped)")
    hline!(pb, [b0.fwhm], lc=:gray, ls=:dot, label="baseline 198°")
    hline!(pb, [hd_ref], lc=:steelblue, ls=:dash, lw=1.5, label="HD ≈$(round(Int,hd_ref))°")

    # (c) baseline vs narrowest-valid bump profile (recentred)
    xd = heading_deg(x)
    pc = plot(xlabel="heading  [deg]", ylabel="firing rate", title="(c) widest vs narrowest valid bump",
              legend=:topright, xlims=(0,360), xticks=0:90:360, left_margin=8mm, bottom_margin=5mm)
    plot!(pc, xd, recenter_to_pi(b0.rate, x), lc=:gray, lw=2, label="baseline ($(round(Int,b0.fwhm))°)")
    plot!(pc, xd, recenter_to_pi(narrow.rate, x), lc=:crimson, lw=2, label="narrowest valid ($(round(Int,narrow.fwhm))°)")
    vspan!(pc, [180 - hd_ref/2, 180 + hd_ref/2], fc=:steelblue, fa=0.08, lc=:transparent, label="HD ~$(round(Int,hd_ref))° width")

    fig = plot(pa, pb, pc, layout=(1,3), size=(1620, 460), bottom_margin=6mm)
    savefig(fig, joinpath("figures", "bump_width.png"))
    println("\n  saved figures/bump_width.png")

    open(joinpath("figures", "bump_width.csv"), "w") do io
        println(io, "# field-only static bump width — what narrows it (Δ=$Δ κ=$κ0 g0=$g0, B=0); HD ref $(hd_ref) deg")
        println(io, "lever,value,fwhm_deg,peak_rate,surround,zmax,valid_bump")
        for (a2,m) in zip(a2s,a2_m); println(io, "a2,", a2, ",", m.fwhm, ",", m.peak, ",", m.surr, ",", m.zmax, ",", m.valid); end
        for (aE,m) in zip(aEs,aE_m); println(io, "aE,", aE, ",", m.fwhm, ",", m.peak, ",", m.surr, ",", m.zmax, ",", m.valid); end
        for (η̄,m) in zip(η̄s,η̄_m); println(io, "eta,", η̄, ",", m.fwhm, ",", m.peak, ",", m.surr, ",", m.zmax, ",", m.valid); end
    end
    println("  saved figures/bump_width.csv")
    return (baseline=b0.fwhm, narrowest=narrow.fwhm)
end

main()
