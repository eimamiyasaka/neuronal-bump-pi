# Phase-2 validation: single MOVING bumps (Laing & Omel'chenko 2020, Fig. 2).
#
# Before trusting the s(B) sweep (run_sweep.jl), confirm that an individual moving
# bump looks right: a coherent travelling wave whose firing-rate / |z| / arg(z)
# profile matches the paper at B = 0.08, 0.12, 0.16, with the characteristic twist
# count 3 → 2 → 1 in arg(z) as B increases. This is the qualitative companion to
# the quantitative sweep — same model layer (src/moving.jl), separate from the
# Phase-1 static scripts.
#
# Operating point is the MOVING-bump one: Δ = 0.1 (not the Δ = 0.01 static point),
# κ = 2, η̄ = −0.4, n = 2 (Laing Fig. 2).
#
# CORRECTNESS NOTE. A moving bump must be read with INSTANTANEOUS quantities. The
# field's rate(z)/|z|/arg(z) are already instantaneous (a snapshot of the settled
# travelling wave). On the spiking side we therefore show a θ_k snapshot, NOT the
# time-averaged Eq-10 mean_frequencies/mean_order_parameter used for the static
# bump — averaging at a fixed neuron while the bump translates past it would smear
# the profile. Each panel is re-centred into the bump's co-moving frame so the
# three B-values line up at x = π for comparison.

include("../src/moving.jl")    # track_*_centroid, lateral_speed, seed_*_bump, bump_centroid, rate, arg_laing
include("plotting.jl")         # pi_ticks, to_02pi, break_wraps
using Plots
using Plots.PlotMeasures

# Shift a periodic, position-ordered profile so the bump centre xc sits at x = π.
# The grid is uniform, so a co-moving recentre is just an integer circular shift;
# the same shift applies to every profile of the same length (field grid or ring
# neurons), keeping rate/|z|/arg (or the θ snapshot) mutually aligned.
function recenter(profile, xc, npts)
    shift = round(Int, (π - xc) / (2π / npts))   # grid points to bring xc → π
    return circshift(profile, shift)
end

# Twist = net number of 2π by which arg(z) (Laing's convention) DECREASES across
# the ring once (their definition). Computed on the unwrapped argument; reported as
# a signed integer so we can read off 3/2/1 directly.
function twist_count(z)
    al = unwrap(arg_laing.(z))                   # unwrap our own helper (no DSP dep)
    return round(Int, (al[1] - al[end]) / (2π))
end

function main()
    # --- moving-bump operating point (Laing Fig. 2) ---
    Δ  = 0.1
    η̄  = -0.4
    κ  = 2.0
    n  = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)   # unit-area pulse normalisation
    dt = 0.02                                      # Laing's integrator step
    Bs = (0.08, 0.12, 0.16)                        # the three Fig. 2 columns

    Nx = 256                                       # field grid
    N  = 8192                                      # spiking network

    xf = field_positions(Nx)
    xr = make_positions(N)

    # rng seeded so the η draw is reproducible across runs (as in run_ring.jl)
    η = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    # Settle a motionless B = 0 bump ONCE on each side, then continue it to each
    # target B (warm-start, exactly as the sweep does — this lands on the moving-
    # bump branch rather than gambling that a kick forms a bump at large B directly).
    Khat0 = fft(field_kernel(xf))                  # B = 0
    K0    = make_kernel(xr)                         # B = 0
    zf0 = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    θr0 = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)

    # storage for the assembled figures
    field_panels = Plots.Plot[]                    # 3×3 grid: rows rate/|z|/arg, cols B
    ring_panels  = Plots.Plot[]                    # θ_k snapshots, one per B
    xt = pi_ticks(2π)
    yt = pi_ticks(2π)

    # build the field grid (column-major collection so we can slot into the grid)
    rate_ps = Plots.Plot[]; absz_ps = Plots.Plot[]; arg_ps = Plots.Plot[]

    for B in Bs
        # ---- field: continue the B=0 bump to this B; T=300 settles then measures ----
        KhatB = fft(field_kernel(xf; B=B))
        xcF, zF = track_field_centroid(zf0, η̄, Δ, κ, KhatB, cos.(xf), sin.(xf); T=300.0, dt=dt)
        sF = lateral_speed(xcF, dt; frac=0.5)
        xc_f = bump_centroid(rate.(zF), cos.(xf), sin.(xf))   # centre of the snapshot
        twist = twist_count(zF)

        rprof = recenter(rate.(zF),               xc_f, Nx)
        aprof = recenter(abs.(zF),                xc_f, Nx)
        gprof = recenter(arg_laing.(zF),          xc_f, Nx)   # recentre THEN wrap for display
        gdisp = break_wraps(to_02pi.(gprof))

        push!(rate_ps, plot(xf, rprof, title="B=$B  (s=$(round(sF,digits=3)), twist $twist)",
                            ylabel="f", legend=false, ylims=(0, 3.5), xticks=xt, left_margin=6mm))
        push!(absz_ps, plot(xf, aprof, ylabel="|z|", legend=false, ylims=(0, 1), xticks=xt, left_margin=6mm))
        push!(arg_ps,  plot(xf, gdisp, ylabel="arg(z)", xlabel="x", legend=false,
                            ylims=(0, 2π), xticks=xt, yticks=yt, left_margin=6mm))

        # ---- spiking net: continue the B=0 bump to this B; snapshot θ_k ----
        KB = make_kernel(xr; B=B)
        xcR, θR = track_ring_centroid(θr0, η, KB, a_n, n, κ; T=300.0, dt=dt)
        sR = lateral_speed(xcR, dt; frac=0.5)
        xc_r = bump_centroid(pulse(θR, a_n, n), KB.cosx, KB.sinx)
        θshift = recenter(θR, xc_r, N)             # co-moving θ_k (re-ordered neurons)

        push!(ring_panels, scatter(xr, θshift, ms=1, mc=:purple, alpha=0.3, legend=false,
                                   title="B=$B  (s=$(round(sR,digits=3)))",
                                   xlabel="x_k", ylabel="θ_k", ylims=(0, 2π),
                                   xticks=xt, yticks=yt, left_margin=6mm))

        println("B=$B:  field s=", round(sF, digits=4), " (twist $twist)   ",
                "spiking s=", round(sR, digits=4),
                "   |Δs|=", round(abs(sF - sR), digits=4))
    end

    # ---- FIELD figure: Laing Fig. 2 rows 2–4 (rate / |z| / arg), columns = B ----
    field_panels = vcat(rate_ps, absz_ps, arg_ps)     # row-major: all rate, all |z|, all arg
    figF = plot(field_panels..., layout=(3, 3), size=(1100, 850))
    display(figF)
    savefig(figF, joinpath("figures", "moving_field_profiles.png"))
    println("saved moving_field_profiles.png")

    # ---- RING figure: Laing Fig. 2 row 1 (θ_k snapshots), columns = B ----
    figR = plot(ring_panels..., layout=(1, 3), size=(1100, 350))
    display(figR)
    savefig(figR, joinpath("figures", "moving_ring_snapshots.png"))
    println("saved moving_ring_snapshots.png")

    return nothing
end

main()
