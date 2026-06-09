include("../src/ring.jl")
include("../src/field.jl")     # reuse rate(z), arg_laing, angular_distance
include("plotting.jl")         # pi_ticks, wrap_extend, to_02pi, break_wraps
using Plots
using Plots.PlotMeasures       # for mm margins

# Seed a bump in the spiking net the same way the field does (run_field.jl): relax
# from θ=0 to the rest (PSR) state, apply a localized transient drive Iext to carve
# a bump, then release and run free. Returns the SETTLED phase state after release
# (evolve_population stores no trajectory, so this stays memory-light at large N).
function kick_then_release(η, K, a_n, n, κ, x;
                           A=0.6, σ=0.6, x0=π, T_relax=30.0, T_kick=20.0,
                           T_free=400.0, dt=0.01)
    θ0   = zeros(length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    θr = evolve_population(θ0, η, K, a_n, n, κ; T=T_relax, dt=dt)            # → rest
    θk = evolve_population(θr, η, K, a_n, n, κ; T=T_kick,  dt=dt, Iext=kick) # carve bump
    θf = evolve_population(θk, η, K, a_n, n, κ; T=T_free,  dt=dt)            # release
    return θf
end

function main()
    # --- parameters (locals, not globals) ---
    N  = 8192        # network size; start small; scale to 8192 once correct

    η̄  = -0.4       # excitability distribution: Cauchy centre
    Δ  = 0.01       #                            half-width (heterogeneity)

    κ = 2.0         # coupling strength (global dial)

    n = 2           # pulse sharpness
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)   # derived: unit-area normalisation

    # --- build the fixed scaffolding ---
    rng = Random.MersenneTwister(1)            # seeded ⇒ reproducible η draw across runs
    x = make_positions(N)
    η = make_excitabilities(N, η̄, Δ; rng=rng)
    K = make_kernel(x)

    # --- seed a bump by kick-then-release, then run long enough to settle ---
    # (settling timescale ~1/Δ = 100, so T_free is several hundred, as on the field side)
    θ_set = kick_then_release(η, K, a_n, n, κ, x; T_free=1000.0)   # settled bump state

    half = N ÷ 32                          # smoothing half-window (~constant physical scale vs N)

    # ---- Row-2 rate: Laing & Omel'chenko Eq. (10) ----
    # Mean frequency measured DIRECTLY from the neurons (time-averaged dθ/dt over a
    # settled window), independent of any order parameter — the genuine microscopic
    # counterpart of the field's rate(z) (Eq. 9). f_raw are the per-neuron dots;
    # f_smooth is the profile line (ring_smooth) — exactly Laing's dots-and-curve.
    T_meas = 200.0                         # averaging window on the settled bump
    f_raw    = mean_frequencies(θ_set, η, K, a_n, n, κ; T=T_meas, dt=0.01)
    f_smooth = ring_smooth(f_raw, half)

    # ---- Rows 3–4: synchrony and mean phase from the order parameter ----
    # Time-average ⟨e^{-iθ}⟩ over the settled window (mean_order_parameter), THEN
    # spatial-smooth — a single snapshot under-samples and would crush |z|→0 in the
    # firing band, leaving arg z as pure noise. This matches how the rate above is
    # time-averaged (Eq. 10), so all three panels are measured consistently.
    z    = ring_smooth(mean_order_parameter(θ_set, η, K, a_n, n, κ; T=T_meas, dt=0.01), half)
    zabs = abs.(z)                               # |z| = <e^{-iθ}> magnitude (conjugate of Laing's)
    zarg = to_02pi.(arg_laing.(z))              # arg in [0,2π), sign matching the paper
    # Mask the phase where |z| is essentially zero: arg z is the angle of a ~zero
    # vector there (undefined in the limit), so plotting a line would assert a phase
    # the data don't support. Blank it instead (honest "no defined phase here").
    # Threshold 0.04 is below the bump's true central |z| (~0.08 in the field) yet
    # above the finite-N sampling floor, so it keeps the real centre plateau and
    # masks only the genuine z-origin crossings. State this threshold in any caption.
    zarg[zabs .< 0.02] .= NaN

    # ---- COMPARISON figure: the 3-panel bump profile (Laing Fig. 1 rows 2–4) ----
    # Spiking panels are wrap-extended π/2 past 2π (the ring is periodic) so the
    # bump — which drifts to an arbitrary centre on a finite network — always reads
    # whole across the 0/2π seam, and so they sit visibly alongside the field's 0..2π.
    pad = π/2
    xe, f_raw_e    = wrap_extend(x, f_raw, pad)
    _,  f_smooth_e = wrap_extend(x, f_smooth, pad)
    _,  zabs_e     = wrap_extend(x, zabs, pad)
    _,  zarg_e     = wrap_extend(x, zarg, pad)             # already in [0,2π) and masked
    zarg_e = break_wraps(zarg_e)                           # break the 0↔2π seam (now clean, as field)
    xt = pi_ticks(2.5π)                                    # π-scaled x-axis to 5π/2
    yt = pi_ticks(2π)                                      # π-scaled y-axis for arg (0..2π)

    p_r = scatter(xe, f_raw_e, ms=1, mc=:purple, alpha=0.3, label="f_k", # Laing eq.10
                  xlabel="position x", ylabel="firing rate f_k", title="bump profile",
                  ylims=(0, 0.5), xlims=(0, 2.5π), xticks=xt, left_margin=8mm, right_margin=12mm)
    plot!(p_r, xe, f_smooth_e, lc=:black, lw=2, label="smoothed")
    p_z = plot(xe, zabs_e, xlabel="position x", ylabel="|z|", title="synchrony |z|",
               legend=false, ylims=(0, 1), xlims=(0, 2.5π), xticks=xt, left_margin=8mm, right_margin=12mm)
    p_a = plot(xe, zarg_e, xlabel="position x", ylabel="arg(z)", title="mean phase arg(z)",
               legend=false, ylims=(0, 2π), xlims=(0, 2.5π), xticks=xt, yticks=yt, left_margin=8mm, right_margin=12mm)
    display(p_r); display(p_z); display(p_a)              # individual panels (REPL)
    h = plot(p_r, p_z, p_a, layout=(3, 1), size=(700, 900))
    display(h)
    savefig(h, joinpath("figures", "ring_profile.png"))
    println("saved ring_profile.png")

    # ---- DIAGNOSTICS figure: neuron snapshot (Laing Fig. 1 row 1) ----
    # θ_k vs x_k at the settled time: the bump shows directly as a band of neurons
    # spread around the phase circle (firing) over a quiescent surround clustered
    # near the rest phase. Replaces the old raw-activity heatmap, which could not
    # distinguish a bump from a flooded ring. Same π/2 wrap-extension as the profiles.
    xe_s, θ_set_e = wrap_extend(x, θ_set, pad)
    snap = scatter(xe_s, θ_set_e, ms=1, mc=:purple, alpha=0.4, legend=false,
                   xlabel="position x_k", ylabel="θ_k", title="neuron snapshot",
                   ylims=(0, 2π), xlims=(0, 2.5π), xticks=xt, yticks=yt, left_margin=8mm, right_margin=12mm)
    display(snap)
    savefig(snap, joinpath("figures", "ring_diagnostics.png"))
    println("saved ring_diagnostics.png")

    println("done: N=", N,
            "   bump centre x=", round(x[argmax(f_smooth)], digits=3),   # where it pinned (drift check)
            "   f_k range ", round(minimum(f_smooth), digits=3), " ... ", round(maximum(f_smooth), digits=3),
            "   |z| range ", round(minimum(zabs), digits=3), " ... ", round(maximum(zabs), digits=3))

    return x, η, K   # hand back for later steps
end

x, η, K = main()