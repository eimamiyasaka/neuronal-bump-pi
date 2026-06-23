# Phase-4, Step 4.1: conductance-based, Dale-compliant synapses — the closure gate.
# Swaps the current-based coupling η+κI for conductance synapses g·(E_syn−v); the only
# new physics is one term per model (− G·sinθ ring / +(G/2)(z²−1) field). All modeling
# decisions live in notes/writeupAssist/step4_modeling_note.md; the physics is in
# src/conductance.jl. This script is the validation gate (mirrors run_field + run_ring):
#
#   1. DEBUG — with g0=1, matched reversals (E_E=κ, E_I=−κ) and the shunt OFF,
#      E_E·K_E+E_I·K_I = κK and there is no shunting, so the conductance code must
#      reproduce the current-based bump exactly. Isolates the drive J from the shunt G.
#   2. SHUNT SWEEP — increase the conductance gain g0 from ~0 (current-based limit) at
#      matched reversals; locate the operating g0 where a static bump still exists.
#   3. STATIC gate  (B=0,   chosen g0) — micro/macro 3-panel overlay.
#   4. MOVING gate  (B=0.1, chosen g0) — micro/macro lateral-speed agreement.
#   5. RE-TUNE scan — conductance static-bump regime over (η̄, g0).
#
# Operating point (inherited): Δ=0.1, κ=2, η̄=−0.4, n=2. The conductance gain g0 is the
# new shunt knob; reversals start matched (±κ/g0) so g0→0 is the validated current-based
# bump. Field grid Nx=256, spiking N=8192, dt=0.02.

include("../src/conductance.jl")   # dale_*, matched_reversals, *_cond steppers/seeding/readouts
include("plotting.jl")             # pi_ticks, wrap_extend, to_02pi, break_wraps
using Plots
using Plots.PlotMeasures

# Classify a released field state as a bump (localized high rate, surround at rest).
# Like run_field.jl's helper but the silent-surround test is RELATIVE to the rest-rate
# floor `rrest` = rate(rest_state(η̄,Δ)): at Δ=0.1 that floor is ~0.025 (the Δ/π
# heterogeneity baseline), so an absolute 0.02 threshold would mislabel every bump
# (the surround can never go below rest). A bump = a clear peak above rest with the
# surround returned to ~rest.
function bump_metrics(x, z, rrest)
    r = rate.(z)
    rmax, imax = findmax(r)
    rmin  = minimum(r)
    width = count(>(0.5*(rmax + rmin)), r)
    # bump is naturally ~half-ring wide (first-harmonic kernel, CLAUDE.md); flood = nearly
    # the whole ring, so the bump/flood boundary is 0.7·N, not N/2.
    is_bump = (rmax > rrest + 0.05) && (rmin < rrest + 0.01) && (width < 0.7 * length(x))
    return rmax, rmin, width, x[imax], is_bump
end

# Circular-shift amount (grid points) that moves a profile's activity centroid to
# c_target — aligns the micro and macro bumps so the panels overlay 1:1.
function shift_to(activity, cosx, sinx, c_target)
    c = bump_centroid(activity, cosx, sinx)
    return round(Int, mod(c_target - c, 2π) / (2π) * length(activity))
end

function main()
    # --- operating point (locals, not globals) ---
    Δ   = 0.1
    η̄   = -0.4
    κ   = 2.0
    n   = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt  = 0.02

    Nx = 256
    N  = 8192

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))
    cf, sf = cos.(xf), sin.(xf)
    rrest = rate(rest_state(η̄, Δ))                  # rest-rate floor (~0.025 at Δ=0.1)
    println("rest-rate floor rrest=", round(rrest, digits=4), " (surround can't go below this)")

    # ========================================================================
    # 1. DEBUG — J-assembly (g0=1, matched reversals, shunt OFF) == current-based
    # ========================================================================
    println("== 1. debug: J-assembly (shunt OFF, matched reversals) must equal current-based ==")
    Khat_cb  = fft(field_kernel(xf))
    z_cb     = seed_field_bump(xf, η̄, Δ, κ, Khat_cb; dt=dt)
    E_E1, E_I1 = matched_reversals(κ, 1.0)                           # (κ, −κ); g0=1, J=κK
    syn_unit = dale_field(xf, E_E1, E_I1; g0=1.0)
    z_co     = seed_field_bump_cond(xf, η̄, Δ, κ, syn_unit; dt=dt, shunt=false)
    maxdiff  = maximum(abs.(rate.(z_cb) .- rate.(z_co)))
    println("  max |rate_cb − rate_cond| = ", maxdiff,
            maxdiff < 1e-6 ? "   ✓ assembly correct" : "   ✗ MISMATCH — check J/kernel split")

    # ========================================================================
    # 2. SHUNT SWEEP — turn on the conductance gain g0 at matched reversals
    # ========================================================================
    println("\n== 2. shunt sweep (matched reversals ±κ/g0; field-only) — locate operating g0 ==")
    g0s = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    g0_op = 0.0
    for g0 in g0s
        eE, eI = matched_reversals(κ, g0)
        s = dale_field(xf, eE, eI; g0=g0)
        z = seed_field_bump_cond(xf, η̄, Δ, κ, s; dt=dt)
        rmax, rmin, width, _, isb = bump_metrics(xf, z, rrest)
        println("  g0=", rpad(g0,5), " (E_E=", round(eE,digits=1), ")  r_max=", round(rmax,digits=3),
                "  r_min=", round(rmin,digits=4), "  width=", width, "/", Nx, "  bump=", isb)
        isb && (g0_op = g0)                                          # keep the largest g0 with a bump
    end
    g0_op == 0.0 && (g0_op = 0.1)                                    # fallback if none survived
    eEo, eIo = matched_reversals(κ, g0_op)
    println("  → operating conductance gain g0=", g0_op, "  (E_E=", round(eEo,digits=2), ", E_I=", round(eIo,digits=2), ")")

    # ========================================================================
    # 3. STATIC gate — conductance at the chosen g0, B=0, micro vs macro
    # ========================================================================
    println("\n== 3. static bump, conductance g0=$g0_op, B=0 ==")
    synf = dale_field(xf, eEo, eIo; g0=g0_op)
    synr = dale_ring(xr, eEo, eIo; g0=g0_op)

    zf = seed_field_bump_cond(xf, η̄, Δ, κ, synf; dt=dt)
    gE = synf.g0 .* coupling_field(zf, synf.KhatE); gI = synf.g0 .* coupling_field(zf, synf.KhatI)
    println("  conductance ranges:  g_E∈[", round(minimum(gE),digits=3), ",", round(maximum(gE),digits=3),
            "]  g_I∈[", round(minimum(gI),digits=3), ",", round(maximum(gI),digits=3),
            "]  G_max=", round(maximum(gE.+gI),digits=3), "  (g_E,g_I must be ≥ 0)")
    rmax, rmin, width, _, isb = bump_metrics(xf, zf, rrest)
    println("  field bump:  r_max=", round(rmax,digits=3), "  r_min=", round(rmin,digits=4),
            "  width=", width, "/", Nx, "  bump=", isb)

    rprof_f = rate.(zf); zabs_f = abs.(zf); zarg_f = to_02pi.(arg_laing.(zf))

    println("  settling + measuring spiking bump (N=$N) …"); flush(stdout)
    θset = seed_ring_bump_cond(η, synr, a_n, n, κ, xr; dt=dt)
    half = N ÷ 32
    f_raw    = mean_frequencies_cond(θset, η, synr, a_n, n, κ; T=200.0, dt=dt)
    f_smooth = ring_smooth(f_raw, half)
    zr       = ring_smooth(mean_order_parameter_cond(θset, η, synr, a_n, n, κ; T=200.0, dt=dt), half)
    zabs_r   = abs.(zr)
    zarg_r   = to_02pi.(arg_laing.(zr));  zarg_r[zabs_r .< 0.02] .= NaN

    # align both bumps to x=π so the profiles overlay 1:1
    shf = shift_to(rprof_f, cf, sf, π)
    rprof_f = circshift(rprof_f, shf); zabs_f = circshift(zabs_f, shf); zarg_f = circshift(zarg_f, shf)
    cr, sr = cos.(xr), sin.(xr)
    shr = shift_to(f_smooth, cr, sr, π)
    f_raw = circshift(f_raw, shr); f_smooth = circshift(f_smooth, shr)
    zabs_r = circshift(zabs_r, shr); zarg_r = circshift(zarg_r, shr)
    println("  field r_max=", round(maximum(rprof_f),digits=3), "  spiking f_max=", round(maximum(f_smooth),digits=3),
            "  |  field |z|_max=", round(maximum(zabs_f),digits=3), "  spiking |z|_max=", round(maximum(zabs_r),digits=3))

    xt = pi_ticks(2π)
    p_r = plot(xf, rprof_f, lc=:black, lw=2, label="field rate(z)",
               xlabel="position x", ylabel="firing rate", title="conductance bump (micro vs macro), g0=$g0_op",
               ylims=(0, max(0.5, 1.1*maximum(rprof_f))), xticks=xt, left_margin=8mm)
    scatter!(p_r, xr, f_raw, ms=1, mc=:purple, alpha=0.15, label="spiking f_k")
    plot!(p_r, xr, f_smooth, lc=:orange, lw=2, ls=:dash, label="spiking smoothed")
    p_z = plot(xf, zabs_f, lc=:black, lw=2, label="field |z|", xlabel="position x", ylabel="|z|",
               title="synchrony |z|", ylims=(0, 1), xticks=xt, left_margin=8mm)
    plot!(p_z, xr, zabs_r, lc=:orange, lw=2, ls=:dash, label="spiking |z|")
    p_a = plot(xf, break_wraps(zarg_f), lc=:black, lw=2, label="field arg z", xlabel="position x", ylabel="arg(z)",
               title="mean phase arg(z)", ylims=(0, 2π), xticks=xt, yticks=pi_ticks(2π), left_margin=8mm)
    plot!(p_a, xr, zarg_r, lc=:orange, lw=2, ls=:dash, label="spiking arg z")
    fig = plot(p_r, p_z, p_a, layout=(3,1), size=(750, 950))
    savefig(fig, joinpath("figures", "conductance_static_microvsmacro.png"))
    println("  saved conductance_static_microvsmacro.png")

    # ========================================================================
    # 4. MOVING gate — conductance g0, B=0.1, micro vs macro speed
    # ========================================================================
    println("\n== 4. moving bump, conductance g0=$g0_op, B=0.1 ==")
    Bmov = 0.1
    synf_m = dale_field(xf, eEo, eIo; g0=g0_op, B=Bmov)
    synr_m = dale_ring(xr, eEo, eIo; g0=g0_op, B=Bmov)
    xcf, _ = track_field_centroid_cond(zf,  η̄, Δ, κ, synf_m, cf, sf; T=200.0, dt=dt)
    sfield = lateral_speed(xcf, dt)
    println("  measuring spiking moving bump (N=$N) …"); flush(stdout)
    xcr, _ = track_ring_centroid_cond(θset, η, synr_m, a_n, n, κ; T=200.0, dt=dt)
    sring  = lateral_speed(xcr, dt)
    println("  s_field=", round(sfield,digits=4), "  s_ring=", round(sring,digits=4),
            "  |Δs|=", round(abs(sfield-sring),digits=4))

    pm = plot((1:length(xcf)).*dt, unwrap(xcf), lc=:black, lw=2, label="field centroid",
              xlabel="time", ylabel="unwrapped centroid", title="moving bump B=$Bmov, g0=$g0_op (micro vs macro)", left_margin=8mm)
    plot!(pm, (1:length(xcr)).*dt, unwrap(xcr), lc=:orange, lw=2, ls=:dash, label="spiking centroid")
    savefig(pm, joinpath("figures", "conductance_moving_microvsmacro.png"))
    println("  saved conductance_moving_microvsmacro.png")

    # ========================================================================
    # 5. RE-TUNE scan — conductance static-bump regime over (η̄, g0), field-only
    # ========================================================================
    println("\n== 5. conductance bump regime map over (η̄, g0); ✓ = localized bump ==")
    η̄s = (-0.6, -0.5, -0.4, -0.3, -0.2)
    print(rpad("η̄ \\ g0", 8)); for g0 in g0s; print(rpad(g0, 7)); end; println()
    for η̄v in η̄s
        print(rpad(η̄v, 8))
        rrest_v = rate(rest_state(η̄v, Δ))
        for g0 in g0s
            eE, eI = matched_reversals(κ, g0)
            s = dale_field(xf, eE, eI; g0=g0)
            z = seed_field_bump_cond(xf, η̄v, Δ, κ, s; dt=dt, T_free=150.0)
            _, rmin, _, _, isb = bump_metrics(xf, z, rrest_v)
            print(rpad(isb ? "✓" : (rmin >= rrest_v + 0.01 ? "flood" : "·"), 7))
        end
        println()
    end
    println("(· = decays to rest;  flood = whole ring active;  ✓ = localized bump)")

    return xf, η, synf, synr, g0_op
end

main()
