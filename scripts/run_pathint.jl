# Phase-3, Step 3.1: velocity-driven path integration — the bump TRANSLATES under a
# time-varying self-motion command Ω(t), and its decoded heading φ(t) tracks the
# commanded heading Φ(t)=∫Ω dt. This is the first genuinely new step (project.md §3.1):
# Phase 2 swept the asymmetry B statically; here B(t)=β·Ω(t) is a dynamic command.
#
# Three commands (modeling note §3): (a) constant Ω — clean translation + quasi-static
# unity-gain check; (b) step changes — abrupt-velocity / lag probe; (c) a realistic
# band-limited (OU) Ω(t) — naturalistic dead-reckoning. Each is run on BOTH the
# macroscopic field and the microscopic spiking net (the recurring closure gate), from
# the SAME settled B=0 bump.
#
# Operating point (unchanged from Phase 2): Δ=0.1, κ=2, η̄=−0.4, n=2. Calibration
# β=1/3.91 sets unity small-signal gain (modeling note §2). Output: heading-tracking
# and instantaneous-speed figures + CSVs in figures/.

include("../src/pathint.jl")   # drive_field/drive_ring, omega_*, Bfun_from_omega, heading_estimate, …
include("plotting.jl")
using Plots
using Plots.PlotMeasures

function main()
    # --- operating point (Laing moving-bump point) ---
    Δ  = 0.1
    η̄  = -0.4
    κ  = 2.0
    n  = 2
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)
    dt = 0.02
    β  = 1 / 3.91                  # Ω→B calibration: unity small-signal gain (note §2)

    Nx = 256                       # field grid
    N  = 8192                      # spiking neurons (expensive side — closure standard)

    xf = field_positions(Nx)
    xr = make_positions(N)
    η  = make_excitabilities(N, η̄, Δ; rng=Random.MersenneTwister(1))

    # --- settle ONE motionless B=0 bump on each side; all commands start here ---
    Khat0 = fft(field_kernel(xf))
    K0    = make_kernel(xr)
    println("seeding B=0 bumps (field + spiking N=$N) …"); flush(stdout)
    zf0 = seed_field_bump(xf, η̄, Δ, κ, Khat0; dt=dt)
    θr0 = seed_ring_bump(η, K0, a_n, n, κ, xr; dt=dt)

    # --- the three commands Ω(t) (kept inside the monostable band; note §6) ---
    T_demo = 300.0                 # ≳ several settling times τ_set≈50
    commands = [
        ("constant", omega_const(0.10),                                              T_demo),
        ("step",     omega_step([100.0, 175.0, 250.0], [0.10, -0.08, 0.06, 0.00]),   320.0),
        ("realistic",omega_realistic(360.0, dt; τ=20.0, σ=0.05, Ωbar=0.0,
                                     rng=Random.MersenneTwister(2)),                 360.0),
    ]

    head_panels  = Plots.Plot[]    # row 1: decoded vs commanded heading
    speed_panels = Plots.Plot[]    # row 2: instantaneous bump speed vs Ω(t)

    for (name, Ωfun, T) in commands
        Bfun = Bfun_from_omega(Ωfun, β)

        # --- drive both models from the same settled bump ---
        tsF, xcF, _ = drive_field(zf0, η̄, Δ, κ, xf, Bfun; T=T, dt=dt)
        tsR, xcR, _ = drive_ring(θr0, η, xr, a_n, n, κ, Bfun; T=T, dt=dt)

        # --- decoded headings (relative to start) and the commanded heading ---
        φF = heading_estimate(xcF); φF .-= φF[1]
        φR = heading_estimate(xcR); φR .-= φR[1]
        Φ  = commanded_heading(tsF, Ωfun; φ0=0.0)         # ideal unity-gain heading

        # --- instantaneous speeds vs the command ---
        vF = instantaneous_speed(tsF, xcF; window=4.0)
        vR = instantaneous_speed(tsR, xcR; window=4.0)
        Ωt = Ωfun.(tsF)

        # --- per-command validation prints ---
        closure = maximum(abs.(φF .- φR))                 # micro/macro heading agreement
        if name == "constant"
            Ω0 = Ωfun(0.0)
            sF = lateral_speed(xcF, dt; frac=0.5)         # settled speed, tail
            sR = lateral_speed(xcR, dt; frac=0.5)
            println("[constant Ω=$Ω0]  field s=", round(sF, digits=4),
                    " (k=", round(sF/Ω0, digits=3), ")   spiking s=", round(sR, digits=4),
                    " (k=", round(sR/Ω0, digits=3), ")   [unity-gain target k≈1]")
        end
        println("[$name]  final decoded φ: field=", round(φF[end], digits=3),
                " spiking=", round(φR[end], digits=3),
                "   commanded Φ=", round(Φ[end], digits=3),
                "   micro/macro max|Δφ|=", round(closure, digits=4))
        @assert all(isfinite, φF) && all(isfinite, φR) "non-finite heading — check settling/drive"

        # --- heading panel ---
        ph = plot(tsF, Φ, lc=:black, lw=2, label="commanded ∫Ω dt", title="$name",
                  xlabel="t", ylabel="Δ heading", legend=:topleft, left_margin=6mm)
        plot!(ph, tsF, φF, lc=:dodgerblue, lw=2, label="field")
        plot!(ph, tsR, φR, lc=:red, lw=1.5, ls=:dash, label="spiking")
        push!(head_panels, ph)

        # --- speed panel ---
        ps = plot(tsF, Ωt, lc=:black, lw=2, label="commanded Ω", title="$name",
                  xlabel="t", ylabel="dφ/dt", legend=:topright, left_margin=6mm)
        plot!(ps, tsF, vF, lc=:dodgerblue, lw=1.5, label="field")
        plot!(ps, tsR, vR, lc=:red, lw=1, alpha=0.6, label="spiking")
        push!(speed_panels, ps)

        # --- persist the heading traces as data ---
        open(joinpath("figures", "pathint_$(name).csv"), "w") do io
            println(io, "t,Omega,commanded,field,spiking")
            for i in eachindex(tsF)
                println(io, tsF[i], ",", Ωt[i], ",", Φ[i], ",", φF[i], ",", φR[i])
            end
        end
    end

    fig = plot(vcat(head_panels, speed_panels)..., layout=(2, 3), size=(1500, 750))
    display(fig)
    savefig(fig, joinpath("figures", "pathint_tracking.png"))
    println("saved pathint_tracking.png + pathint_{constant,step,realistic}.csv")
    return nothing
end

main()
