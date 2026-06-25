# Phase-5 extension — structured (localized) inhibition and DIVISIVE shunting.
#
# Closes the deferred item in step4_2_gate.md / step5_gate.md: "the divisive-shunting regime
# with an intact bump needs a structured-K_I / two-population E–I extension." Phase-4 flat
# inhibition K_I=0.3 makes g_I=g0∫K_I·P spatially UNIFORM, so the shunt is subtractive (holds
# the silent surround but cannot reshape the bump). Here K_I gets a cosine structure (aI>0, see
# src/conductance.jl) so g_I(x) TRACKS the bump and the shunt acts DIVISIVELY at the bump. To
# keep the Phase-1–4 bump unchanged we preserve the NET kernel K_E−K_I = 0.1+0.3cos (cE−cI=0.1,
# aE−aI=0.3 ⇒ aE=0.3+aI). Dale non-negativity caps the structure at aI≲0.1 (K_E=cE+aE cos ≥ 0);
# we use aI=0.1 (maximal single-kernel structure, K_E just touches 0).
#
# Establishes:
#   0. CLOSURE is exact at the algebra level — with the shunt OFF and matched reversals the
#      structured config reduces to the current-based bump (net kernel preserved) to ~1e-15.
#   1. CLOSURE holds with the shunt ON — micro/macro static overlay, structured vs a FLAT
#      baseline with the IDENTICAL pipeline (controlled: any residual is finite-N, not
#      structure-specific). The recurring project gate.
#   2. DIVISIVE signature over the g0 shunt band: gain control (peak rate), silent-surround
#      quenching (the divisive win), bump width, and the phase5-consistent regime (static/
#      oscillon/collapsed). Honest reporting — measured, not assumed; the effect is bounded by
#      Dale, which is itself the finding (full divisive range wants a two-population E–I).
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2, Nx=256, N=8192, dt=0.02, conductance g0 (matched
# reversals ±κ/g0). Field-only for the g0 sweep; spiking overlays for the gate.

include("../src/phase5.jl")        # breathing_amplitude, classify_regime (+ conductance, core)
include("../src/calibration.jl")   # fwhm_deg
include("plotting.jl")
using Plots
using Plots.PlotMeasures

# Bump detector relative to the rest-rate floor (same logic as run_conductance.jl).
function bump_metrics(x, z, rrest)
    r = rate.(z)
    rmax, imax = findmax(r); rmin = minimum(r)
    width = count(>(0.5*(rmax + rmin)), r)
    is_bump = (rmax > rrest + 0.05) && (rmin < rrest + 0.01) && (width < 0.7 * length(x))
    return rmax, rmin, width, x[imax], is_bump
end
shift_to(activity, cosx, sinx, c_target) =
    round(Int, mod(c_target - bump_centroid(activity, cosx, sinx), 2π) / (2π) * length(activity))

# Micro/macro static-bump agreement under a given Dale config, identical pipeline for flat and
# structured (so the comparison is controlled). Returns the field/spiking profiles + RMSE/FWHM.
function measure_closure(xf, xr, η, a_n, n, κ, η̄, Δ, dt, synf, synr; Nx=256, N=8192)
    cf, sf = cos.(xf), sin.(xf)
    zf = seed_field_bump_cond(xf, η̄, Δ, κ, synf; dt=dt)
    θset = seed_ring_bump_cond(η, synr, a_n, n, κ, xr; dt=dt)
    half = N ÷ 32
    f_smooth = ring_smooth(mean_frequencies_cond(θset, η, synr, a_n, n, κ; T=200.0, dt=dt), half)
    zr = ring_smooth(mean_order_parameter_cond(θset, η, synr, a_n, n, κ; T=200.0, dt=dt), half)
    rprof_f = circshift(rate.(zf), shift_to(rate.(zf), cf, sf, π))
    zabs_f  = circshift(abs.(zf),  shift_to(rate.(zf), cf, sf, π))
    cr, sr  = cos.(xr), sin.(xr)
    shr = shift_to(f_smooth, cr, sr, π)
    f_smooth = circshift(f_smooth, shr); zabs_r = circshift(abs.(zr), shr)
    idx = [round(Int, (j-1)/Nx*N)+1 for j in 1:Nx]
    rmse = sqrt(sum(abs2, f_smooth[idx] .- rprof_f) / Nx)
    return (rprof_f=rprof_f, zabs_f=zabs_f, f_smooth=f_smooth, zabs_r=zabs_r, zf=zf,
            rmse=rmse, peakf=maximum(rprof_f), peakr=maximum(f_smooth),
            fwhm_f=fwhm_deg(rprof_f), fwhm_r=fwhm_deg(f_smooth))
end

function main()
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt, Nx, N = 0.02, 256, 8192
    aI = 0.1; aE = 0.3 + aI                    # structured inhibition (net kernel preserved)
    g0_op = 0.1
    xf = field_positions(Nx); xr = make_positions(N)
    cf, sf = cos.(xf), sin.(xf)
    rrest = rate(rest_state(η̄, Δ))
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))
    println("== structured inhibition / divisive shunting ==")
    println("structured K_I: aI=$aI, aE=$aE (net K = 0.1+0.3cos preserved); rest floor=", round(rrest, digits=4))

    # ========================================================================
    # 0. EXACT closure check — shunt OFF + matched reversals ⇒ current-based bump
    # ========================================================================
    println("\n== 0. exact closure: structured K_I, shunt OFF, matched reversals == current-based ==")
    z_cb = seed_field_bump(xf, η̄, Δ, κ, fft(field_kernel(xf)); dt=dt)
    eE1, eI1 = matched_reversals(κ, 1.0)
    syn_off = dale_field(xf, eE1, eI1; g0=1.0, aE=aE, aI=aI)
    z_off = seed_field_bump_cond(xf, η̄, Δ, κ, syn_off; dt=dt, shunt=false)
    maxdiff = maximum(abs.(rate.(z_cb) .- rate.(z_off)))
    println("  max|rate_cb − rate_structured(shunt off)| = ", maxdiff,
            maxdiff < 1e-6 ? "  ✓ net kernel preserved, drive assembly exact (closure intact)" :
                             "  ✗ assembly error")

    # ========================================================================
    # 1. CLOSURE gate (shunt ON) — structured vs FLAT baseline, identical pipeline
    # ========================================================================
    println("\n== 1. closure gate (g0=$g0_op, B=0): structured vs flat baseline, N=$N ==")
    eE, eI = matched_reversals(κ, g0_op)
    synf_fl = dale_field(xf, eE, eI; g0=g0_op)                       # flat K_I (aI=0)
    synr_fl = dale_ring(xr, eE, eI; g0=g0_op)
    synf_st = dale_field(xf, eE, eI; g0=g0_op, aE=aE, aI=aI)         # structured K_I
    synr_st = dale_ring(xr, eE, eI; g0=g0_op, aE=aE, aI=aI)
    gI = synf_st.g0 .* coupling_field(seed_field_bump_cond(xf, η̄, Δ, κ, synf_st; dt=dt), synf_st.KhatI)
    println("  structured g_I(x)∈[", round(minimum(gI),digits=3), ",", round(maximum(gI),digits=3),
            "]  (varies in x; flat g_I is uniform)")
    println("  measuring spiking bumps (N=$N) — flat then structured …"); flush(stdout)
    cl_fl = measure_closure(xf, xr, η, a_n, n, κ, η̄, Δ, dt, synf_fl, synr_fl; Nx=Nx, N=N)
    cl_st = measure_closure(xf, xr, η, a_n, n, κ, η̄, Δ, dt, synf_st, synr_st; Nx=Nx, N=N)
    for (tag, c) in (("flat", cl_fl), ("structured", cl_st))
        println("  ", rpad(tag,11), " rate RMSE=", round(c.rmse,digits=4),
                " (", round(100*c.rmse/c.peakf,digits=1), "% of peak)  peak f/m=",
                round(c.peakf,digits=3), "/", round(c.peakr,digits=3),
                "  FWHM f/m=", round(c.fwhm_f,digits=0), "°/", round(c.fwhm_r,digits=0), "°")
    end
    verdict = abs(cl_st.rmse - cl_fl.rmse) < 0.25 * cl_fl.rmse
    println(verdict ? "  ✓ structured closure ≈ flat baseline ⇒ structure does NOT break closure (residual is finite-N)" :
                      "  ⚠ structured RMSE differs from flat by >25% — inspect")

    xt = pi_ticks(2π)
    p_r = plot(xf, cl_st.rprof_f, lc=:black, lw=2, label="field rate(z)", xlabel="position x", ylabel="firing rate",
               title="structured-K_I bump (micro vs macro), g0=$g0_op, aI=$aI", xticks=xt, left_margin=8mm)
    plot!(p_r, xr, cl_st.f_smooth, lc=:orange, lw=2, ls=:dash, label="spiking f_k")
    p_z = plot(xf, cl_st.zabs_f, lc=:black, lw=2, label="field |z|", xlabel="position x", ylabel="|z|",
               title="synchrony |z|", ylims=(0,1), xticks=xt, left_margin=8mm)
    plot!(p_z, xr, cl_st.zabs_r, lc=:orange, lw=2, ls=:dash, label="spiking |z|")
    gIs = circshift(gI, shift_to(rate.(cl_st.zf), cf, sf, π))
    p_g = plot(xf, gIs, lc=:blue, lw=2, label="g_I(x) structured (localized)", xlabel="position x",
               ylabel="inhibitory conductance", title="shunt g_I(x): structured tracks the bump", xticks=xt, left_margin=8mm)
    hline!(p_g, [g0_op * 0.3 * sum(meanpulse.(cl_st.zf)) * (2π/Nx)], lc=:gray, ls=:dot, label="flat-K_I g_I (uniform)")
    savefig(plot(p_r, p_z, p_g, layout=(3,1), size=(760, 980)),
            joinpath("figures", "structured_inhibition_microvsmacro.png"))
    println("  saved figures/structured_inhibition_microvsmacro.png")

    # ========================================================================
    # 2. DIVISIVE sweep — flat vs structured up the g0 shunt band (field, phase5-consistent)
    # ========================================================================
    println("\n== 2. divisive shunting: flat (aI=0) vs structured (aI=$aI) over g0 ==")
    g0s = collect(0.0:0.05:0.55)
    res = Dict{Symbol,NamedTuple}()
    for (name, aIc, aEc) in ((:flat, 0.0, 0.3), (:structured, aI, aE))
        peaks = Float64[]; surr = Float64[]; widths = Float64[]; regimes = Symbol[]
        for g0 in g0s
            if g0 == 0.0
                z = seed_field_bump(xf, η̄, Δ, κ, fft(field_kernel(xf)); dt=dt)
                step = zz -> field_step(zz, η̄, Δ, κ, fft(field_kernel(xf)), dt)
            else
                e1, e2 = matched_reversals(κ, g0)
                syn = dale_field(xf, e1, e2; g0=g0, aE=aEc, aI=aIc)
                z = seed_field_bump_cond(xf, η̄, Δ, κ, syn; T_free=300.0, dt=dt)
                step = zz -> field_step_cond(zz, η̄, Δ, κ, syn, dt)
            end
            rmaxv, rminv, _, _, _ = bump_metrics(xf, z, rrest)
            b = breathing_amplitude(z, step; T=600.0, dt=dt)
            reg = classify_regime(b)
            push!(peaks, rmaxv); push!(surr, rminv)
            push!(widths, reg == :collapsed ? NaN : fwhm_deg(rate.(z))); push!(regimes, reg)
        end
        res[name] = (peaks=peaks, surr=surr, widths=widths, regimes=regimes)
        # silent-surround band edge: largest g0 with a NON-collapsed bump whose surround is still
        # near rest (a collapsed state trivially has a low surround, so it must be excluded).
        si = findlast(i -> regimes[i] != :collapsed && surr[i] < rrest + 0.01, eachindex(g0s))
        silent = si === nothing ? NaN : g0s[si]
        hopf = let i = findfirst(==(:oscillon), regimes); i === nothing ? NaN : g0s[i] end
        coll = let i = findfirst(==(:collapsed), regimes); i === nothing ? NaN : g0s[i] end
        println("  ", rpad(name,11), " silent-surround band g0≤", silent,
                "   oscillon onset g0*≈", isnan(hopf) ? "—" : hopf,
                "   collapse g0≈", isnan(coll) ? "—" : coll)
    end
    pf, ps = res[:flat], res[:structured]
    i03 = findfirst(==(0.3), g0s)
    println("  divisive read — at g0=0.3:  peak flat=", round(pf.peaks[i03],digits=3),
            " structured=", round(ps.peaks[i03],digits=3),
            " | surround flat=", round(pf.surr[i03],digits=3), " structured=", round(ps.surr[i03],digits=3),
            "  (lower surround = stronger divisive quenching)")

    p1 = plot(g0s, pf.peaks, m=:circle, lc=:gray, mc=:gray, lw=2, label="flat K_I (subtractive)",
              xlabel="conductance gain g0 (shunt)", ylabel="peak firing rate r_max",
              title="(a) gain control (both shunt the peak)", legend=:topright, left_margin=8mm, bottom_margin=5mm)
    plot!(p1, g0s, ps.peaks, m=:square, lc=:crimson, mc=:crimson, lw=2, label="structured K_I (divisive)")
    hline!(p1, [rrest], lc=:black, ls=:dot, label="rest floor")
    p2 = plot(g0s, pf.surr, m=:circle, lc=:gray, mc=:gray, lw=2, label="flat K_I",
              xlabel="conductance gain g0 (shunt)", ylabel="surround rate r_min",
              title="(b) silent-surround quenching (divisive win)", legend=:topleft, left_margin=10mm, bottom_margin=5mm)
    plot!(p2, g0s, ps.surr, m=:square, lc=:crimson, mc=:crimson, lw=2, label="structured K_I")
    hline!(p2, [rrest], lc=:black, ls=:dot, label="rest floor")
    p3 = plot(g0s, pf.widths, m=:circle, lc=:gray, mc=:gray, lw=2, label="flat K_I",
              xlabel="conductance gain g0 (shunt)", ylabel="bump FWHM [deg]",
              title="(c) bump width", legend=:bottomleft, left_margin=10mm, bottom_margin=5mm)
    plot!(p3, g0s, ps.widths, m=:square, lc=:crimson, mc=:crimson, lw=2, label="structured K_I")
    savefig(plot(p1, p2, p3, layout=(1,3), size=(1560, 440)),
            joinpath("figures", "structured_inhibition_divisive.png"))
    println("  saved figures/structured_inhibition_divisive.png")

    open(joinpath("figures", "structured_inhibition.csv"), "w") do io
        println(io, "# structured-inhibition closure: flat RMSE=", cl_fl.rmse, "  structured RMSE=", cl_st.rmse,
                "  exact_shunt_off_diff=", maxdiff)
        println(io, "g0,flat_peak,flat_surr,flat_fwhm,flat_regime,struct_peak,struct_surr,struct_fwhm,struct_regime")
        for i in eachindex(g0s)
            println(io, g0s[i], ",", pf.peaks[i], ",", pf.surr[i], ",", pf.widths[i], ",", pf.regimes[i],
                    ",", ps.peaks[i], ",", ps.surr[i], ",", ps.widths[i], ",", ps.regimes[i])
        end
    end
    println("  saved figures/structured_inhibition.csv")
    return (rmse_flat=cl_fl.rmse, rmse_struct=cl_st.rmse, exact_diff=maxdiff, g0s=g0s, flat=pf, structured=ps)
end

main()
