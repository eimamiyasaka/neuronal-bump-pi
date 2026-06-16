# ============================================================================
# Ring-attractor explainer animation (Phase 3, realistic-OU path integration).
#
# Tells the head-direction story in one clip: a simple character TURNS IN PLACE at
# its true heading Φ(t)=∫Ω dt, driven by a realistic Ornstein–Uhlenbeck angular-
# velocity command Ω(t) — the only thing the "animal" actually senses. That command
# enters the network as kernel asymmetry B(t)=β·Ω(t); the activity BUMP on the ring
# translates at speed s≈Ω, so its decoded centroid integrates Ω into an internal
# heading that stays locked to the true one — with NO compass input.
#
# Reuses the unchanged Phase-1/2/3 models via the profile-recording drivers in
# pathint.jl (drive_field_record / drive_ring_record). Both substrates are shown side
# by side (the macroscopic field bump and the microscopic spiking bump) so micro/macro
# agreement is visible. Output: figures/ring_attractor.mp4.
#
# The expensive spiking run is CACHED to figures/anim_data.jls (stdlib Serialization);
# delete that file (or run with FORCE=true) to re-simulate. Visual-only tweaks reuse
# the cache and re-render in seconds.
# ============================================================================

include("../src/pathint.jl")
include("plotting.jl")
using Plots
using Plots.PlotMeasures
using Serialization

const CACHE = joinpath("figures", "anim_data.jls")
const FORCE = get(ENV, "FORCE", "false") == "true"

# --- simulate (or load cache): drive both models under the realistic OU command ----
function simulate()
    if isfile(CACHE) && !FORCE
        println("loading cached animation data from $CACHE (set ENV FORCE=true to re-sim)")
        return deserialize(CACHE)
    end

    # operating point — identical to run_pathint.jl (Laing moving-bump point)
    Δ, η̄, κ, n = 0.1, -0.4, 2.0, 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt  = 0.02
    β   = 1 / 3.91                       # Ω→B calibration: unity small-signal gain
    Nx, N = 256, 8192
    T   = 360.0
    stride = 45                          # → 400 frames; @ 30fps ≈ 13 s clip

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    println("seeding B=0 bumps (field + spiking N=$N) …"); flush(stdout)
    Khat0 = fft(field_kernel(xf)); K0 = make_kernel(xr)
    zf0 = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    θr0 = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)

    # the realistic band-limited (OU) self-motion command — same seed as the CSV run
    Ωfun = omega_realistic(T, dt; τ=20.0, σ=0.05, Ωbar=0.0, rng=Random.MersenneTwister(2))
    Bfun = Bfun_from_omega(Ωfun, β)

    println("driving field …"); flush(stdout)
    tsF, R, xcF = drive_field_record(zf0, η̄, Δ, κ, xf, Bfun; T=T, dt=dt, stride=stride)
    println("driving spiking net (N=$N) …"); flush(stdout)
    tsR, xdisp, A, xcR = drive_ring_record(θr0, η, xr, a_n, n, κ, Bfun;
                                           T=T, dt=dt, stride=stride)

    ts = tsF                               # identical grids (same stride/nt)
    Ωt = Ωfun.(ts)
    Φ  = commanded_heading(ts, Ωfun; φ0=0.0)        # true heading (ground truth)
    φF = unwrap(xcF) .- xcF[1]                       # decoded displacement (field)
    φR = unwrap(xcR) .- xcR[1]                       # decoded displacement (spiking)

    data = (; ts, xf, R, xcF, xdisp, A, xcR, Ωt, Φ, φF, φR,
              shiftF=xcF[1], shiftR=xcR[1])
    serialize(CACHE, data)
    println("simulation cached → $CACHE")
    return data
end

# --- drawing helpers ---------------------------------------------------------------
# A ring of neurons coloured by activity, with a decoded "compass needle" (where the
# bump is) and a faint ground-truth tick. `angles` are placed in a frame where the
# bump's start sits at angle 0, so it lines up with the character.
function draw_ring!(plt, angles, act, clim, dec, tru, ttl; showlegend=false)
    plot!(plt, cos.(range(0, 2π; length=120)), sin.(range(0, 2π; length=120)),
          lc=:gray80, lw=1, label="", title=ttl, titlefontsize=10)
    scatter!(plt, cos.(angles), sin.(angles); marker_z=act, ms=3.2,
             markerstrokewidth=0, c=:inferno, clims=clim, colorbar=false, label="")
    plot!(plt, [0, 0.96cos(tru)], [0, 0.96sin(tru)]; lc=:white, lw=2, ls=:dot,
          label="true heading")                                                # true
    plot!(plt, [0, 0.9cos(dec)],  [0, 0.9sin(dec)];  lc=:cyan,  lw=3,
          label="decoded (bump)")                                              # decoded
    scatter!(plt, [0.9cos(dec)], [0.9sin(dec)]; mc=:cyan, ms=5, markerstrokewidth=0, label="")
    plot!(plt; xlims=(-1.35, 1.35), ylims=(-1.35, 1.35), aspect_ratio=1,
          framestyle=:none, background_color_inside=:black,
          legend=(showlegend ? :bottomright : false), legendfontsize=7,
          foreground_color_legend=nothing, background_color_legend=RGBA(0,0,0,0.4),
          legendfontcolor=:white)
end

# The character: a head (filled disc) with a heading arrow, turning in place at Φ.
function draw_character!(plt, Φ, Ω, ttl)
    plot!(plt, cos.(range(0, 2π; length=120)), sin.(range(0, 2π; length=120)),
          lc=:gray85, lw=1, label="", title=ttl, titlefontsize=10)
    # whisker showing field of view, then the body and nose arrow
    plot!(plt, [0, 1.05cos(Φ)], [0, 1.05sin(Φ)]; lc=:gold, lw=2, ls=:dash, label="")
    scatter!(plt, [0.0], [0.0]; mc=:steelblue, ms=26, markerstrokecolor=:navy,
             markerstrokewidth=2, label="")
    plot!(plt, [0, 0.55cos(Φ)], [0, 0.55sin(Φ)]; lc=:navy, lw=5,
          arrow=arrow(:closed, 0.4, 0.4), label="")
    annotate!(plt, 0, -1.25, text("Ω = $(round(Ω, digits=3))  rad/t", 8, :gray30))
    plot!(plt; xlims=(-1.35, 1.35), ylims=(-1.35, 1.35), aspect_ratio=1,
          framestyle=:none, legend=false)
end

# --- render ------------------------------------------------------------------------
function main()
    d = simulate()
    nf = length(d.ts)
    T  = d.ts[end]

    Rclim = (0.0, maximum(d.R))
    Aclim = (0.0, maximum(d.A))
    αF = d.xf    .- d.shiftF             # field display angles (bump start at 0)
    αR = d.xdisp .- d.shiftR             # spiking display angles
    hlim = (minimum(vcat(d.Φ, d.φF, d.φR)) - 0.1, maximum(vcat(d.Φ, d.φF, d.φR)) + 0.1)
    Ωlim = (minimum(d.Ωt) - 0.02, maximum(d.Ωt) + 0.02)

    println("rendering $nf frames …"); flush(stdout)
    anim = @animate for k in 1:nf
        pc = plot(); draw_character!(pc, d.Φ[k], d.Ωt[k], "the animal (true heading Φ)")
        pf = plot(); draw_ring!(pf, αF, d.R[:, k], Rclim, d.φF[k], d.Φ[k],
                                "field bump  (macroscopic)"; showlegend=true)
        pr = plot(); draw_ring!(pr, αR, d.A[:, k], Aclim, d.φR[k], d.Φ[k],
                                "spiking bump  (N=8192)")

        ph = plot(d.ts[1:k], d.Φ[1:k]; lc=:black, lw=2, label="true ∫Ω dt",
                  title="decoded heading tracks the truth", titlefontsize=10,
                  xlabel="t", ylabel="heading (rad)", legend=:topleft,
                  xlims=(0, T), ylims=hlim, left_margin=6mm, bottom_margin=5mm)
        plot!(ph, d.ts[1:k], d.φF[1:k]; lc=:dodgerblue, lw=2, label="field")
        plot!(ph, d.ts[1:k], d.φR[1:k]; lc=:red, lw=1.5, ls=:dash, label="spiking")

        pw = plot(d.ts[1:k], d.Ωt[1:k]; lc=:darkorange, lw=1.5, label="",
                  title="self-motion command Ω(t)  (OU)", titlefontsize=10,
                  xlabel="t", ylabel="Ω (rad/t)", xlims=(0, T), ylims=Ωlim,
                  legend=false, left_margin=6mm, bottom_margin=5mm)
        hline!(pw, [0.0]; lc=:gray70, lw=1, label="")

        l = @layout [a{0.28w} b c; d{0.6w} e]
        plot(pc, pf, pr, ph, pw; layout=l, size=(1500, 950),
             plot_title="Ring attractor integrating self-motion into heading",
             plot_titlefontsize=13)
    end

    out = joinpath("figures", "ring_attractor.mp4")
    mp4(anim, out; fps=30)
    println("saved $out")
    return nothing
end

main()
