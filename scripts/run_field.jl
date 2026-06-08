include("../src/field.jl")
using Plots

# Angular distance on the ring (helper for the seeding kick)
function angular_distance(x, x0)
    d = abs.(x .- x0)
    return min.(d, 2π .- d)
end

# Seed a bump with a transient localized external input, then release the network
# and let it relax freely. Returns the order-parameter field after release.
# This tests *bump existence*: a bump that survives release coexists with rest
# (bistability), which is exactly the regime a ring attractor needs.
function kick_then_release(x, η̄, Δ, κ, Khat;
                           A=0.6, σ=0.6, x0=π, T_kick=20.0, T_free=80.0, dt=0.01)
    z0   = fill(rest_state(η̄, Δ), length(x))
    kick = A .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    Zk = simulate_field(z0, η̄, Δ, κ, Khat; T=T_kick, dt=dt, Iext=kick)   # carve bump
    Zf = simulate_field(Zk[:, end], η̄, Δ, κ, Khat; T=T_free, dt=dt)       # release
    return Zf
end

# Classify the released state: a bump = localized high firing rate with a silent
# surround (small r_min) and width below half the ring.
function bump_metrics(x, z)
    r = rate.(z)
    rmax, imax = findmax(r)
    rmin = minimum(r)
    half = 0.5*(rmax + rmin)
    width = count(>(half), r)
    is_bump = (rmax > 0.05) && (rmin < 0.02) && (width < length(x) ÷ 2)
    return rmax, rmin, width, x[imax], is_bump
end

function main()
    Nx = 256
    Δ  = 0.01         # heterogeneity sweet-spot: Δ≲0.015 → oscillon (gamma) regime;
                       # Δ≳0.04 floods the surround. Δ≈0.02 gives a STATIC bump.
    x  = field_positions(Nx)

    # ---- V1: validate the reduction on a single population ----
    println("== reduction sanity (single population) ==")
    for (η̄, I) in ((-0.4, 0.0), (-0.4, 1.0), (0.5, 0.0))
        z = rest_state(η̄, Δ)                       # rest with no input
        # apply constant input by relaxing with a flat Iext
        Kflat = fft(zeros(Nx))                      # zero kernel ⇒ coupling 0
        Zc = simulate_field(fill(z, Nx), η̄, Δ, 1.0, Kflat; T=200.0, dt=0.01, Iext=I)
        println("  η̄=$η̄ I=$I  ->  r=", round(rate(Zc[1, end]), digits=4),
                "  |z|=", round(abs(Zc[1, end]), digits=3))
    end

    # ---- B1: solve for a bump at a representative operating point ----
    η̄, κ = -0.4, 2.0
    Khat = fft(field_kernel(x))                     # symmetric kernel (B = 0)
    # release long enough for high-spatial-frequency grid residual to damp out
    # (the macroscopic bump converges almost immediately; the per-point residual
    #  is only damped by Δ, so it needs T≳200 to settle visually)
    Zf = kick_then_release(x, η̄, Δ, κ, Khat; T_free=250.0)
    rmax, rmin, width, xc, isb = bump_metrics(x, Zf[:, end])
    println("\n== representative bump  (η̄=$η̄, κ=$κ) ==")
    println("  r_max=", round(rmax, digits=3), "  r_min=", round(rmin, digits=4),
            "  width=", width, "/", Nx, "  center=", round(xc, digits=3),
            "  bump=", isb)

    # bump profile + space-time firing-rate map
    rprof = rate.(Zf[:, end])
    p1 = plot(x, rprof, xlabel="position x", ylabel="firing rate r(x)",
              title="released bump profile (η̄=$η̄, κ=$κ)", legend=false)
    p2 = heatmap(rate.(Zf), xlabel="time step", ylabel="grid point",
                 title="r(x,t) after release", clims=(0, maximum(rprof)))
    fig = plot(p1, p2, layout=(2, 1), size=(700, 700))
    savefig(fig, joinpath("figures", "field_bump.png"))
    println("  saved field_bump.png")

    # ---- B2: map the bump regime over (η̄, κ) ----
    println("\n== bump regime map (kick-then-release; ✓ = stable bump w/ silent surround) ==")
    η̄s = (-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0)
    κs = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
    print(rpad("η̄ \\ κ", 8)); for κ in κs; print(rpad(κ, 7)); end; println()
    for η̄v in η̄s
        print(rpad(η̄v, 8))
        for κv in κs
            Z = kick_then_release(x, η̄v, Δ, κv, Khat; T_free=60.0)
            _, rmin, _, _, isb = bump_metrics(x, Z[:, end])
            print(rpad(isb ? "✓" : (rmin >= 0.02 ? "flood" : "·"), 7))
        end
        println()
    end
    println("(· = decays to rest;  flood = whole ring active;  ✓ = localized bump)")

    return x
end

x = main()
