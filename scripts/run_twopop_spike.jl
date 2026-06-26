# ============================================================================
# EXPLORATORY two-population (E–I) field spike — FIELD ONLY (not a gated result).
#
# Question: does a genuinely DIVISIVE shunting-inhibition regime with an INTACT bump
# become reachable in a two-population E–I architecture, over a WIDER range than the
# single-population Dale-split-kernel model (which is bounded to a marginal effect,
# aI ≲ 0.1 — surround a few % lower, peak ~20–28% better sustained, Hopf/collapse
# unmoved; see notes/writeupAssist/step5_divisive_gate.md)?
#
# Field-only spike. The microscopic spiking model and the micro/macro agreement gate
# are OUT OF SCOPE and are named as required future work in the write-up.
#
# Design note (learned from a coarse operating-point search, scripts/twopop_search.jl):
# the conductance drive uses the synaptic pulse meanpulse(z), which is O(1) even at
# rest. So a kernel with large DC tonically drives its target via E_exc·g — an
# UNBALANCED config makes the I population fire on E's resting pulse and blanket E into
# silence. A bump exists only with PER-POPULATION DC BALANCE (tonic E/I nearly cancel,
# exactly why the single-pop net kernel K has DC 0.1). We therefore hold the net DC
# drive at 0.1 onto BOTH pops and use the cos-structure of the I→E kernel, aEI, as the
# single DIVISIVE knob (aEI=0 ⇒ flat/subtractive inhibition; aEI>0 ⇒ localized inhibition
# that tracks the bump). A sparse high-threshold inhibitory follower (η̄_I ≪ η̄_E) was
# found to silence the whole network, so I is co-active with E (η̄_I = η̄_E).
#
# Operating point: ΔE=ΔI=0.1, η̄E=η̄I=−0.4, κ=2, n=2, Nx=256, dt=0.02, conductance
# gain g0=0.1, matched reversals ±κ/g0=±20 (same scale as the single-pop op point).
# Run:  julia --project=. scripts/run_twopop_spike.jl
# ============================================================================

include("../src/twopop.jl")        # two-pop field model + breathing_amplitude_twopop
                                   #   (pulls phase5 → conductance → … → field core)
include("../src/calibration.jl")   # fwhm_deg
include("plotting.jl")
using Plots
using Plots.PlotMeasures

# Bump detector relative to the rest-rate floor (same logic as run_conductance.jl /
# run_structured_inhibition.jl): a localized, silent-surround, sub-ring-wide bump.
function bump_metrics(x, z, rrest)
    r = rate.(z)
    rmax, imax = findmax(r); rmin = minimum(r)
    width = count(>(0.5*(rmax + rmin)), r)
    is_bump = (rmax > rrest + 0.05) && (rmin < rrest + 0.05) && (width < 0.7 * length(x))
    return rmax, rmin, width, x[imax], is_bump
end
shift_to(activity, cosx, sinx, c_target) =
    round(Int, mod(c_target - bump_centroid(activity, cosx, sinx), 2π) / (2π) * length(activity))

# --- Science E–I config (DC-balanced; aEI = the single divisive knob) ---------------
# Excitation W_EE = W_IE = 0.5 + 0.5cos (recurrent excitation onto both pops). Inhibition
# W_EI = 0.4 + aEI·cos (DC 0.4 ⇒ net DC drive onto E = 0.1, balanced; aEI structures it),
# W_II = 0.4 + 0.1cos (DC-balanced I self-inhibition). aEI=0 ⇒ flat (subtractive) inhibition;
# aEI>0 ⇒ localized inhibition tracking the bump (divisive). Dale non-negativity: aEI ≤ 0.4.
# Unlike the single-population model, aEI is NOT tied to the excitatory kernel by a net-kernel
# constraint — but the bump's robustness still bounds it (Part 2 is exactly that test).
function sci_config(x, aEI; g0, E_exc, E_inh)
    twopop_field(x; g0=g0, E_exc=E_exc, E_inh=E_inh,
                 cEE=0.5, aEE=0.5, cIE=0.5, aIE=0.5,   # recurrent excitation onto E and I
                 cEI=0.4, aEI=aEI,                     # I→E: DC-balanced, structured by aEI
                 cII=0.4, aII=0.1)                     # I→I: DC-balanced, mild structure
end

function main()
    ΔE = ΔI = 0.1
    η̄E = η̄I = -0.4
    κ, n = 2.0, 2
    Nx, dt = 256, 0.02
    g0 = 0.1
    E_exc, E_inh = matched_reversals(κ, g0)            # ±20
    x = field_positions(Nx); cosx, sinx = cos.(x), sin.(x)
    rrest = rate(rest_state(η̄E, ΔE))
    println("== EXPLORATORY two-population (E–I) field spike — FIELD ONLY ==")
    println("op point: Δ=$ΔE, η̄=$η̄E, κ=$κ, g0=$g0, reversals=(",
            round(E_exc,digits=1), ",", round(E_inh,digits=1), "), Nx=$Nx, dt=$dt")
    println("rest-rate floor rate(z_rest) = ", round(rrest, digits=5))

    # ========================================================================
    # PART 0 — VALIDATION ANCHOR: reduce to the validated single-pop conductance bump
    # ========================================================================
    println("\n== Part 0. VALIDATION ANCHOR — two-pop ⇒ single-pop conductance bump ==")
    syn1 = dale_field(x, E_exc, E_inh; g0=g0)
    z_single = seed_field_bump_cond(x, η̄E, ΔE, κ, syn1; dt=dt)
    # Two-pop clone config: I is an identical copy of E (W_EE=W_IE=K_E, W_EI=W_II=K_I),
    # same η̄/Δ/g0/reversals, kick BOTH identically ⇒ z_I ≡ z_E for all time.
    syn_clone = twopop_field(x; g0=g0, E_exc=E_exc, E_inh=E_inh,
                             cEE=0.4, aEE=0.3, cEI=0.3, aEI=0.0,   # K_E / K_I (conductance.jl split)
                             cIE=0.4, aIE=0.3, cII=0.3, aII=0.0)
    zE2, zI2 = seed_twopop_bump(x, η̄E, η̄I, ΔE, ΔI, κ, syn_clone; AE=0.6, AI=0.6, dt=dt)
    d_rate = maximum(abs.(rate.(z_single) .- rate.(zE2)))
    d_sym  = maximum(abs.(zE2 .- zI2))
    println("  max|z_E − z_I|              = ", d_sym,
            d_sym < 1e-12 ? "  ✓ clone symmetry z_I≡z_E held" : "  ✗ symmetry broke")
    println("  max|rate_single − rate_2pop| = ", d_rate,
            d_rate < 1e-13 ? "  ✓ PASS (reduces to validated single-pop bump to ~1e-13)" :
                             "  ✗ FAIL — do NOT proceed to science on an unvalidated model")
    if d_rate >= 1e-13
        println("  STOP: validation anchor failed."); return
    end

    # ========================================================================
    # PART 1 — bump existence under the DC-balanced E–I config (kick E, I follows)
    # ========================================================================
    println("\n== Part 1. bump existence (DC-balanced E–I, kick E only, I follows) ==")
    aEI_demo = 0.15
    syn = sci_config(x, aEI_demo; g0=g0, E_exc=E_exc, E_inh=E_inh)
    zE, zI = seed_twopop_bump(x, η̄E, η̄I, ΔE, ΔI, κ, syn; AE=0.9, AI=0.0, T_free=300.0, dt=dt)
    @assert all(isfinite, zE) && all(isfinite, zI) "non-finite field"
    g = twopop_conductances(zE, zI, syn)
    gmin = minimum(min.(g.gE_onE, g.gI_onE, g.gE_onI, g.gI_onI))
    println("  Dale check: min conductance over all four channels = ", round(gmin, sigdigits=3),
            gmin >= -1e-12 ? "  ✓ all g ≥ 0" : "  ✗ negative conductance!")
    rmaxE, rminE, wE, xcE, isb = bump_metrics(x, zE, rrest)
    println("  E bump: peak=", round(rmaxE,digits=3), "  surround=", round(rminE,digits=4),
            "  FWHM=", round(fwhm_deg(rate.(zE)),digits=0), "°  |z_E|max=", round(maximum(abs.(zE)),digits=3))
    println("  I pop : peak=", round(maximum(rate.(zI)),digits=3), "  |z_I|max=", round(maximum(abs.(zI)),digits=3),
            "  (co-active follower)")
    gmod = 100*(maximum(g.gI_onE)-minimum(g.gI_onE))/maximum(g.gI_onE)
    println("  g_I→E(x) ∈ [", round(minimum(g.gI_onE),digits=4), ",", round(maximum(g.gI_onE),digits=4),
            "]  ⇒ ", round(gmod,digits=0), "% spatial modulation (localized inhibition = divisive)")
    println("  is_bump=", isb, isb ? "  ✓ localized silent-surround bump exists" : "  ⚠ NOT a clean bump")

    # Figure A: validation overlay + the science bump profiles + localized conductance
    xt = pi_ticks(2π)
    shS = shift_to(rate.(z_single), cosx, sinx, π); sh2 = shift_to(rate.(zE2), cosx, sinx, π)
    pA1 = plot(x, circshift(rate.(z_single), shS), lc=:black, lw=3, label="single-pop conductance",
               xlabel="x", ylabel="firing rate", title="Part 0: two-pop clone == single-pop bump (Δ=$d_rate)",
               xticks=xt, left_margin=8mm)
    plot!(pA1, x, circshift(rate.(zE2), sh2), lc=:orange, lw=2, ls=:dash, label="two-pop z_E (clone)")
    shE = shift_to(rate.(zE), cosx, sinx, π)
    pA2 = plot(x, circshift(rate.(zE), shE), lc=:crimson, lw=2, label="E rate",
               xlabel="x", ylabel="firing rate", title="Part 1: E–I bump (aEI=$aEI_demo)", xticks=xt, left_margin=8mm)
    plot!(pA2, x, circshift(rate.(zI), shE), lc=:navy, lw=2, label="I rate")
    hline!(pA2, [rrest], lc=:gray, ls=:dot, label="rest floor")
    pA3 = plot(x, circshift(g.gE_onE, shE), lc=:crimson, lw=2, label="g_E→E (excitatory)",
               xlabel="x", ylabel="conductance", title="conductances onto E (g_I→E localized)", xticks=xt, left_margin=8mm)
    plot!(pA3, x, circshift(g.gI_onE, shE), lc=:navy, lw=2, label="g_I→E (inhibitory, structured)")
    savefig(plot(pA1, pA2, pA3, layout=(3,1), size=(760, 980)),
            joinpath("figures", "twopop_spike_bump.png"))
    println("  saved figures/twopop_spike_bump.png")

    # ========================================================================
    # PART 2 — sweep the DIVISIVE structure aEI (flat/subtractive → localized/divisive)
    # ========================================================================
    println("\n== Part 2. divisive-structure sweep aEI (0=subtractive → localized/divisive) ==")
    aEIs = collect(0.0:0.05:0.40)
    peaks=Float64[]; surr=Float64[]; widths=Float64[]; zE_abs=Float64[]; zI_abs=Float64[]; regimes=Symbol[]; isbump=Bool[]
    for aEI in aEIs
        syn = sci_config(x, aEI; g0=g0, E_exc=E_exc, E_inh=E_inh)
        zEa, zIa = seed_twopop_bump(x, η̄E, η̄I, ΔE, ΔI, κ, syn; AE=0.9, AI=0.0, T_free=250.0, dt=dt)
        stepfun2 = (a,b) -> twopop_step(a, b, η̄E, η̄I, ΔE, ΔI, κ, syn, dt)
        bo = breathing_amplitude_twopop(zEa, zIa, stepfun2; T=400.0, dt=dt)
        rmaxv, rminv, wv, _, isbv = bump_metrics(x, zEa, rrest)
        push!(peaks, rmaxv); push!(surr, rminv); push!(isbump, isbv)
        push!(widths, isbv ? fwhm_deg(rate.(zEa)) : NaN)
        push!(zE_abs, maximum(abs.(zEa))); push!(zI_abs, maximum(abs.(zIa))); push!(regimes, classify_regime(bo))
    end
    # Report the CONTIGUOUS bump band from aEI=0 (do NOT bridge a collapse gap).
    firstgap = findfirst(.!isbump)
    last_contig = firstgap === nothing ? length(aEIs) : firstgap - 1
    p0 = peaks[1]
    println("  full aEI→is_bump: ", join(["$(aEIs[i])=>$(isbump[i] ? "B" : "·")" for i in eachindex(aEIs)], " "))
    println("  full aEI→regime : ", join(["$(aEIs[i])=>$(regimes[i])" for i in eachindex(aEIs)], " "))
    println("  CONTIGUOUS silent-surround bump band: aEI ∈ [0.0, ", aEIs[last_contig], "]")
    println("  peak across it: aEI=0 (subtractive) = ", round(p0,digits=3),
            "  →  aEI=", aEIs[last_contig], " = ", round(peaks[last_contig],digits=3),
            "   ⇒ peak normalized by ", round(100*(1 - peaks[last_contig]/p0),digits=0), "% with the bump INTACT")
    println("    (single-pop divisive bound, step5_divisive_gate.md: ~24% peak change over its static band)")
    println("  surround across the band: ", round(surr[1],digits=4), " → ", round(surr[last_contig],digits=4),
            "  (rest floor ", round(rrest,digits=4), " — stays silent)")
    println("  BEYOND the band the localized inhibition COLLAPSES the bump (peak→",
            round(minimum(peaks[last_contig+1:end]),digits=3),
            "); a marginal re-entrant low-amplitude state appears at large aEI (honest, not a clean bump).")
    println("  finiteness: all peaks finite = ", all(isfinite, peaks))

    p1 = plot(aEIs, peaks, m=:square, lc=:crimson, mc=:crimson, lw=2, label="peak rate r_max",
              xlabel="I→E inhibitory structure aEI", ylabel="firing rate",
              title="(a) divisive peak normalization", legend=:topright, left_margin=8mm, bottom_margin=5mm)
    plot!(p1, aEIs, surr, m=:circle, lc=:navy, mc=:navy, lw=2, label="surround r_min")
    scatter!(p1, aEIs[isbump], peaks[isbump], mc=:green, ms=7, msc=:black, label="bump exists")
    hline!(p1, [rrest], lc=:black, ls=:dot, label="rest floor")
    p2 = plot(aEIs, widths, m=:square, lc=:purple, mc=:purple, lw=2, label="bump FWHM",
              xlabel="I→E inhibitory structure aEI", ylabel="FWHM [deg]",
              title="(b) bump width (NaN = no bump)", legend=:topleft, left_margin=10mm, bottom_margin=5mm)
    p3 = plot(aEIs, zE_abs, m=:square, lc=:crimson, mc=:crimson, lw=2, label="|z_E|",
              xlabel="I→E inhibitory structure aEI", ylabel="peak |z|",
              title="(c) synchrony |z_E|, |z_I|", legend=:bottomright, left_margin=8mm, bottom_margin=5mm, ylims=(0,1))
    plot!(p3, aEIs, zI_abs, m=:diamond, lc=:navy, mc=:navy, lw=2, label="|z_I|")
    savefig(plot(p1, p2, p3, layout=(1,3), size=(1560, 440)),
            joinpath("figures", "twopop_spike_sweep.png"))
    println("  saved figures/twopop_spike_sweep.png")

    # ========================================================================
    # PART 3 — divisive signature: peak-rate vs DRIVE, structured vs flat inhibition
    # ========================================================================
    println("\n== Part 3. drive-response: in-band divisive (aEI=0.15) vs subtractive (aEI=0) ==")
    drives = collect(-0.6:0.1:0.4)
    resp = Dict{Symbol,Vector{Float64}}()
    for (name, aEI) in ((:divisive, 0.15), (:subtractive, 0.0))
        pk = Float64[]
        syn = sci_config(x, aEI; g0=g0, E_exc=E_exc, E_inh=E_inh)
        for ηd in drives
            zEd, _ = seed_twopop_bump(x, ηd, η̄I, ΔE, ΔI, κ, syn; AE=0.9, AI=0.0, T_free=250.0, dt=dt)
            push!(pk, maximum(rate.(zEd)))
        end
        resp[name] = pk
        imax = argmax(pk)
        println("  ", rpad(String(name),12), " peak rate over η̄_E∈[-0.6,0.4]: max=", round(pk[imax],digits=3),
                " at η̄_E=", drives[imax], "  (endpoints ", round(first(pk),digits=3), "/", round(last(pk),digits=3), ")")
    end
    # Honest caveat: the bump exists only in a narrow drive window; outside it the E field
    # OVER-SYNCHRONIZES (z→1 ⇒ rate→0), so the textbook monotone "sublinear response" curve
    # is confounded. The divisive read is the LOWER in-window peak vs subtractive (gain control).
    iw = findall(d -> -0.55 <= d <= -0.35, drives)
    println("  in the bump window η̄_E∈[-0.5,-0.4]: divisive peak ≤ ", round(maximum(resp[:divisive][iw]),digits=3),
            " vs subtractive ", round(maximum(resp[:subtractive][iw]),digits=3),
            "  ⇒ divisive holds a lower (gain-controlled) peak; outside the window both over-synchronize.")
    pD = plot(drives, resp[:subtractive], m=:circle, lc=:gray, mc=:gray, lw=2, label="subtractive (flat I, aEI=0)",
              xlabel="excitatory drive η̄_E", ylabel="peak firing rate",
              title="Part 3: drive-response (bump window narrow; outside it E over-synchronizes)", legend=:topleft,
              size=(720, 480), left_margin=8mm, bottom_margin=5mm)
    plot!(pD, drives, resp[:divisive], m=:square, lc=:crimson, mc=:crimson, lw=2, label="divisive (structured I, aEI=0.15)")
    hline!(pD, [rrest], lc=:black, ls=:dot, label="rest floor")
    savefig(pD, joinpath("figures", "twopop_spike_driveresponse.png"))
    println("  saved figures/twopop_spike_driveresponse.png")

    # ------------------------------------------------------------------- CSV
    open(joinpath("figures", "twopop_spike.csv"), "w") do io
        println(io, "# two-pop E-I exploratory spike. validation max|rate_single-rate_2pop|=", d_rate)
        println(io, "# Part 2 divisive-structure sweep (aEI)")
        println(io, "aEI,peak,surr,fwhm,zE,zI,regime,is_bump")
        for i in eachindex(aEIs)
            println(io, aEIs[i], ",", peaks[i], ",", surr[i], ",", widths[i], ",", zE_abs[i], ",", zI_abs[i], ",", regimes[i], ",", isbump[i])
        end
        println(io, "# Part 3 drive-response (peak rate vs eta_E)")
        println(io, "etaE,div_peak,sub_peak")
        for i in eachindex(drives)
            println(io, drives[i], ",", resp[:divisive][i], ",", resp[:subtractive][i])
        end
    end
    println("  saved figures/twopop_spike.csv")
    println("\n== done (exploratory; micro/macro spiking gate is required future work) ==")
    return (validation=d_rate, aEIs=aEIs, peaks=peaks, surr=surr, isbump=isbump)
end

main()
