# ============================================================================
# Step 1.3 [GATE] — micro/macro static-bump match (project.md §15).
#
# The comparison QUANTITIES exist on both sides (field rate(z)/|z|/arg z vs the
# spiking mean_frequencies/mean_order_parameter), but no script had ever overlaid
# the two profiles or reported a quantitative agreement metric. This is that script.
#
# Both models are seeded into the SAME static bump at the Laing Fig. 1c operating
# point (η̄=-0.4, κ=2, Δ=0.01, n=2) by the identical kick-then-release protocol the
# two run scripts use. The macro bump is pinned at x0=π by construction; the finite
# spiking bump drifts to an arbitrary centre, so we recentre BOTH (shift computed
# from the firing-rate centroid) onto a common grid before overlaying and scoring.
#
# Success criterion: the firing-rate and |z| profiles agree in shape/width/amplitude
# (Laing Fig. 1). Reported: peak/surround/FWHM on each side + RMS profile error.
# Artifact: figures/compare_profile.png + figures/compare_metrics.csv.
# ============================================================================
include("../src/ring.jl")      # spiking model + mean_frequencies/mean_order_parameter/ring_smooth
include("../src/field.jl")     # macro field + rate/arg_laing/meanpulse/rest_state
include("plotting.jl")
using Plots, Plots.PlotMeasures, Random, Printf, DelimitedFiles

# --- seeding (copied from run_field.jl / run_ring.jl so we don't run their main()) ---
function seed_field(x, η̄, Δ, κ, Khat; A=0.6, σ=0.6, x0=π, T_kick=20.0, T_free=1000.0, dt=0.01)
    z0   = fill(rest_state(η̄, Δ), length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    Zk = simulate_field(z0, η̄, Δ, κ, Khat; T=T_kick, dt=dt, Iext=kick)
    Zf = simulate_field(Zk[:, end], η̄, Δ, κ, Khat; T=T_free, dt=dt)
    return Zf[:, end]
end

function seed_ring(η, K, a_n, n, κ, x; A=0.6, σ=0.6, x0=π, T_relax=30.0, T_kick=20.0, T_free=1000.0, dt=0.01)
    θ0   = zeros(length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    θr = evolve_population(θ0, η, K, a_n, n, κ; T=T_relax, dt=dt)
    θk = evolve_population(θr, η, K, a_n, n, κ; T=T_kick,  dt=dt, Iext=kick)
    return evolve_population(θk, η, K, a_n, n, κ; T=T_free, dt=dt)
end

# Firing-rate centroid (phase of the 1st spatial Fourier mode) on grid x.
centroid(r, x) = mod(angle(sum(r .* exp.(im .* x))), 2π)

# Periodic linear interpolation of a (real or complex) uniform-grid profile v at
# query angle q ∈ ℝ (wrapped); grid is [0,2π) with N points.
function pinterp(v, q)
    N = length(v); t = mod(q, 2π) / (2π / N)
    i0 = floor(Int, t); frac = t - i0
    (1 - frac) * v[mod(i0, N) + 1] + frac * v[mod(i0 + 1, N) + 1]
end

# Recentre a profile so its bump (centroid `c`) sits at π, resampled onto `θq`.
recenter(v, θq, c) = [pinterp(v, θ + (c - π)) for θ in θq]

# Full-width at half-max in radians (rate profile on uniform grid of M points).
function fwhm(r, M)
    h = (0.5*(maximum(r) + minimum(r)))
    count(>(h), r) * (2π / M)
end

function main()
    η̄, κ, Δ, n = -0.4, 2.0, 0.01, 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)      # a_2 = 2/3
    Nx, N = 256, 8192
    Mc = 256                                          # common comparison grid
    θq = [2π*(j-1)/Mc for j in 1:Mc]

    println("== Step 1.3 gate: seeding the SAME static bump on both models (Δ=$Δ) ==")
    # ---- macro field ----
    xf   = field_positions(Nx)
    Khat = fft(field_kernel(xf))
    print("  macro field (Nx=$Nx) settling…"); flush(stdout)
    zf = seed_field(xf, η̄, Δ, κ, Khat); println(" done")
    r_f, za_f = rate.(zf), abs.(zf)

    # ---- micro spiking net ----
    print("  micro spiking (N=$N) settling… (slow)"); flush(stdout)
    rng = Random.MersenneTwister(1)
    xr  = make_positions(N)
    η   = make_excitabilities(N, η̄, Δ; rng=rng)
    K   = make_kernel(xr)
    θset = seed_ring(η, K, a_n, n, κ, xr); println(" done")
    half  = N ÷ 32
    f_sm  = ring_smooth(mean_frequencies(θset, η, K, a_n, n, κ; T=200.0, dt=0.01), half)
    z_r   = ring_smooth(mean_order_parameter(θset, η, K, a_n, n, κ; T=200.0, dt=0.01), half)
    za_r  = abs.(z_r)

    # ---- recentre both bumps to π and resample onto the common grid ----
    cf, cr = centroid(r_f, xf), centroid(f_sm, xr)
    rF  = recenter(r_f, θq, cf);  zF = recenter(zf,  θq, cf)
    rR  = recenter(f_sm, θq, cr); zR = recenter(z_r, θq, cr)
    zaF, zaR = abs.(zF), abs.(zR)

    # ---- quantitative agreement ----
    rmse(a, b) = sqrt(sum(abs2, a .- b) / length(a))
    rate_rmse = rmse(rF, rR);  z_rmse = rmse(zaF, zaR)
    rate_rel  = rate_rmse / maximum(rF)
    # NB: |z| ANTI-correlates with rate — max|z| is the synchronised surround,
    # min|z| is the desynchronised bump centre.
    metrics = [
        ("peak rate",        maximum(rF), maximum(rR)),
        ("surround rate",    minimum(rF), minimum(rR)),
        ("FWHM (rad)",       fwhm(rF, Mc), fwhm(rR, Mc)),
        ("|z| surround max", maximum(zaF), maximum(zaR)),
        ("|z| centre min",   minimum(zaF), minimum(zaR)),
    ]
    println("\n== micro/macro agreement (recentred, common grid Mc=$Mc) ==")
    @printf("  %-16s %10s %10s %10s\n", "quantity", "macro", "micro", "|Δ|")
    for (name, a, b) in metrics
        @printf("  %-16s %10.4f %10.4f %10.4f\n", name, a, b, abs(a-b))
    end
    @printf("  rate-profile RMSE = %.4f  (%.1f%% of peak)\n", rate_rmse, 100rate_rel)
    @printf("  |z|-profile  RMSE = %.4f\n", z_rmse)
    pass = rate_rel < 0.10 && abs(maximum(rF)-maximum(rR)) < 0.05
    println("  GATE: ", pass ? "PASS ✓ (profiles match in shape/width/amplitude)" :
                                "REVIEW — discrepancy exceeds tolerance")

    open(joinpath("figures", "compare_metrics.csv"), "w") do io
        println(io, "quantity,macro,micro,absdiff")
        for (name, a, b) in metrics; println(io, "$name,$a,$b,$(abs(a-b))"); end
        println(io, "rate_profile_rmse,$rate_rmse,,")
        println(io, "rate_profile_rel,$rate_rel,,")
        println(io, "z_profile_rmse,$z_rmse,,")
    end

    # ---- overlay figure: the 3-panel profile, macro vs micro, recentred at π ----
    argF = break_wraps(to_02pi.(arg_laing.(zF)))
    argR = to_02pi.(arg_laing.(zR)); argR[zaR .< 0.02] .= NaN
    xt, yt = pi_ticks(2π), pi_ticks(2π)
    common = (xlims=(0, 2π), xticks=xt, left_margin=8mm, xlabel="position x (recentred, bump at π)")
    p_r = plot(θq, rF, lc=:steelblue, lw=2.5, label="macro field rate(z)", ylabel="firing rate",
               title="(a) bump profile — micro vs macro  (RMSE $(round(100rate_rel,digits=1))% of peak)",
               ylims=(0, 0.5); legend=:topright, common...)
    plot!(p_r, θq, rR, lc=:crimson, lw=2, ls=:dash, label="micro f_k (Eq.10)")
    p_z = plot(θq, zaF, lc=:steelblue, lw=2.5, label="macro |z|", ylabel="synchrony |z|",
               title="(b) synchrony |z|", ylims=(0, 1); legend=:topright, common...)
    plot!(p_z, θq, zaR, lc=:crimson, lw=2, ls=:dash, label="micro |⟨e^{-iθ}⟩|")
    p_a = plot(θq, argF, lc=:steelblue, lw=2.5, label="macro arg z", ylabel="arg(z)",
               title="(c) mean phase arg(z)  (|z|<0.02 masked)", ylims=(0, 2π), yticks=yt;
               legend=:topright, common...)
    plot!(p_a, θq, argR, lc=:crimson, lw=2, ls=:dash, label="micro arg z")
    fig = plot(p_r, p_z, p_a, layout=(3,1), size=(760, 950))
    savefig(fig, joinpath("figures", "compare_profile.png"))
    println("\n  saved figures/compare_profile.png + figures/compare_metrics.csv")
    return pass
end

main()
