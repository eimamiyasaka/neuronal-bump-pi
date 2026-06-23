# Phase-4 closing spot-check: the SHUNTING regime — what the conductance reversals actually do.
#
# Steps 4.1/4.2 ran at MATCHED reversals E=±κ/g0 (=±20 at g0=0.1), where the drive J=κI equals
# the current-based drive and only the shunt G=g0∫(K_E+K_I)P is new. Two distinct "shunting"
# axes were conflated in the modeling note and are separated here:
#
#   • g0 = conductance/shunt GAIN at fixed (matched, subtractive) drive — the shunt term scales
#     ∝ g0 while J stays = κI. This is the real shunt-STRENGTH knob; the bump tolerates 0<g0≲0.3.
#   • E_I → v_rest (≈−0.63) = making inhibition DIVISIVE (shunting) rather than subtractive. The
#     note claimed this is the Phase-5 shunting axis. THIS SPOT-CHECK SHOWS IT DISSOLVES THE BUMP:
#     the silent surround is held by subtractive (hyperpolarizing) global inhibition, so as E_I→rest
#     the surround floods. Pure divisive shunting is gain-controlling, not pattern-forming.
#
# Analytic closure holds for any E_I/g0 (the term is linear in v), so these are operating-point
# results, not closure risks. Deliverables: (A) the E_I bump→flood transition; (B) micro/macro
# closure confirmed at the stronger shunt g0=0.3 (the valid shunt-strength demonstration).
#
# Frame: Δ=0.1, κ=2, n=2, Nx=256, N=8192, dt=0.02.

include("../src/conductance.jl")
include("plotting.jl")
using Plots
using Plots.PlotMeasures

function bump_metrics(x, z, rrest)
    r = rate.(z)
    rmax, imax = findmax(r)
    rmin = minimum(r)
    width = count(>(0.5*(rmax + rmin)), r)
    is_bump = (rmax > rrest + 0.05) && (rmin < rrest + 0.01) && (width < 0.7*length(x))
    return rmax, rmin, width, x[imax], is_bump
end

shift_to(activity, cosx, sinx, c_target) =
    round(Int, mod(c_target - bump_centroid(activity, cosx, sinx), 2π) / (2π) * length(activity))

# Static micro/macro overlay at a given conductance config (see run_conductance.jl). Saves a figure.
function micro_macro_static(tag, xf, xr, η, η̄, Δ, κ, a_n, n, dt, synf, synr, N, figname)
    rrest = rate(rest_state(η̄, Δ))
    zf = seed_field_bump_cond(xf, η̄, Δ, κ, synf; dt=dt)
    rmax, rmin, width, _, isb = bump_metrics(xf, zf, rrest)
    println("  [$tag] field:  r_max=", round(rmax,digits=3), "  r_min=", round(rmin,digits=4),
            "  width=", width, "/", length(xf), "  bump=", isb, "  (rest floor ", round(rrest,digits=3), ")")
    rprof_f = rate.(zf); zabs_f = abs.(zf); zarg_f = to_02pi.(arg_laing.(zf))

    println("  [$tag] settling + measuring spiking bump (N=$N) …"); flush(stdout)
    θset = seed_ring_bump_cond(η, synr, a_n, n, κ, xr; dt=dt)
    half = N ÷ 32
    f_raw    = mean_frequencies_cond(θset, η, synr, a_n, n, κ; T=200.0, dt=dt)
    f_smooth = ring_smooth(f_raw, half)
    zr       = ring_smooth(mean_order_parameter_cond(θset, η, synr, a_n, n, κ; T=200.0, dt=dt), half)
    zabs_r   = abs.(zr)
    zarg_r   = to_02pi.(arg_laing.(zr));  zarg_r[zabs_r .< 0.02] .= NaN

    cf, sf = cos.(xf), sin.(xf); cr, sr = cos.(xr), sin.(xr)
    shf = shift_to(rprof_f, cf, sf, π)
    rprof_f = circshift(rprof_f, shf); zabs_f = circshift(zabs_f, shf); zarg_f = circshift(zarg_f, shf)
    shr = shift_to(f_smooth, cr, sr, π)
    f_raw = circshift(f_raw, shr); f_smooth = circshift(f_smooth, shr)
    zabs_r = circshift(zabs_r, shr); zarg_r = circshift(zarg_r, shr)
    println("  [$tag] micro/macro:  field r_max=", round(maximum(rprof_f),digits=3),
            " spiking f_max=", round(maximum(f_smooth),digits=3),
            " | field |z|_max=", round(maximum(zabs_f),digits=3),
            " spiking |z|_max=", round(maximum(zabs_r),digits=3))

    xt = pi_ticks(2π)
    p_r = plot(xf, rprof_f, lc=:black, lw=2, label="field rate(z)", xlabel="position x", ylabel="firing rate",
               title="$tag (micro vs macro)", ylims=(0, max(0.5,1.1*maximum(rprof_f))), xticks=xt, left_margin=8mm)
    scatter!(p_r, xr, f_raw, ms=1, mc=:purple, alpha=0.15, label="spiking f_k")
    plot!(p_r, xr, f_smooth, lc=:orange, lw=2, ls=:dash, label="spiking smoothed")
    p_z = plot(xf, zabs_f, lc=:black, lw=2, label="field |z|", xlabel="position x", ylabel="|z|",
               title="synchrony |z|", ylims=(0,1), xticks=xt, left_margin=8mm)
    plot!(p_z, xr, zabs_r, lc=:orange, lw=2, ls=:dash, label="spiking |z|")
    p_a = plot(xf, break_wraps(zarg_f), lc=:black, lw=2, label="field arg z", xlabel="position x", ylabel="arg(z)",
               title="mean phase arg(z)", ylims=(0,2π), xticks=xt, yticks=pi_ticks(2π), left_margin=8mm)
    plot!(p_a, xr, zarg_r, lc=:orange, lw=2, ls=:dash, label="spiking arg z")
    savefig(plot(p_r, p_z, p_a, layout=(3,1), size=(750,950)), joinpath("figures", figname))
    println("  [$tag] saved ", figname)
end

function main()
    Δ=0.1; κ=2.0; n=2; dt=0.02
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    Nx=256; N=8192
    xf = field_positions(Nx); xr = make_positions(N)

    # ====================================================================
    # A. E_I transition: subtractive (matched) → divisive (rest). Field-only.
    # ====================================================================
    println("== A. inhibition subtractive→divisive: sweep E_I at E_E=20, g0=0.1, η̄=−0.4 ==")
    rr = rate(rest_state(-0.4, Δ))
    println("  rest floor=", round(rr,digits=3), "  v_rest=−√(−η̄)=", round(-sqrt(0.4),digits=3))
    println("  ", rpad("E_I",8), rpad("r_max",9), rpad("r_min",9), rpad("width",9), "bump?  (r_min≈r_max ⇒ flooded)")
    for E_I in (-20.0, -18.0, -15.0, -10.0, -6.0, -3.0, -1.0, -0.6)
        z = seed_field_bump_cond(xf, -0.4, Δ, κ, dale_field(xf, 20.0, E_I; g0=0.1); dt=dt, T_free=200.0)
        rmax, rmin, w, _, isb = bump_metrics(xf, z, rr)
        println("  ", rpad(E_I,8), rpad(round(rmax,digits=3),9), rpad(round(rmin,digits=4),9), rpad(string(w,"/256"),9), isb)
    end
    println("  ⇒ bump exists only near the matched (strongly subtractive) E_I≈−20; as E_I→rest the")
    println("    surround floods (r_min→r_max). Silent surround needs SUBTRACTIVE inhibition; pure")
    println("    divisive shunting is gain-controlling, not pattern-forming. (Closure holds throughout.)")

    # ====================================================================
    # B. Valid shunt-strength demonstration: micro/macro at the stronger shunt g0=0.3
    #    (matched reversals ⇒ drive J=κI fixed, shunt G ∝ g0 tripled vs the g0=0.1 op point).
    # ====================================================================
    println("\n== B. micro/macro closure at stronger shunt g0=0.3 (matched reversals) ==")
    g0=0.3; η̄=-0.4
    eE, eI = matched_reversals(κ, g0)
    println("  g0=", g0, "  reversals E=±", round(eE,digits=2), "  (shunt G ∝ g0 is 3× the g0=0.1 op point)")
    η = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))
    micro_macro_static("shunt g0=0.3", xf, xr, η, η̄, Δ, κ, a_n, n, dt,
                       dale_field(xf, eE, eI; g0=g0), dale_ring(xr, eE, eI; g0=g0), N,
                       "conductance_g03_microvsmacro.png")
    return nothing
end

main()
