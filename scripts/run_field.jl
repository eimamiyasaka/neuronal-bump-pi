include("../src/field.jl")     # provides rate, arg_laing, angular_distance, etc.
include("plotting.jl")         # pi_ticks, wrap_extend, to_02pi, break_wraps
using Plots
using Plots.PlotMeasures       # for mm margins

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
    Δ  = 0.01         

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
    # release long enough for the high-spatial-frequency transient to damp out.
    # The OA field has no spatial diffusion, so perturbations decay only at rate
    # ~Δ; at Δ=0.01 that means T_free ≳ 1000 (~10/Δ) for a clean static bump 
    # (T_free could probably be less, just needs enough time to damp out). The
    # macroscopic amplitude converges fast; the high-k "fishbone" is the slow part
    # — and it is a transient, NOT an oscillon.
    Zf = kick_then_release(x, η̄, Δ, κ, Khat; T_free=1000.0)
    rmax, rmin, width, xc, isb = bump_metrics(x, Zf[:, end])
    println("\n== representative bump  (η̄=$η̄, κ=$κ) ==")
    println("  r_max=", round(rmax, digits=3), "  r_min=", round(rmin, digits=4),
            "  width=", width, "/", Nx, "  center=", round(xc, digits=3),
            "  bump=", isb)

    # ---- COMPARISON figure: the 3-panel bump profile (Laing Fig. 1 rows 2–4) ----
    # These are exactly the quantities run_ring.jl reports, so they overlay 1:1 in
    # the eventual micro/macro gate (run_compare). arg z uses arg_laing so the sign
    # matches the paper (our z = <e^{-iθ}> is the conjugate of theirs — see field.jl).
    zf    = Zf[:, end]
    rprof = rate.(zf)
    zargp = break_wraps(to_02pi.(arg_laing.(zf)))   # arg in [0,2π); break the 0↔2π seam
    xt = pi_ticks(2π)                               # π-scaled x-axis (field spans 0..2π)
    yt = pi_ticks(2π)                               # π-scaled y-axis for arg (0..2π)
    # rate fixed to Laing's static Fig.1 scale (0–0.5); revisit to (0,5) for moving bumps
    p_r = plot(x, rprof, xlabel="position x", ylabel="firing rate r",
               title="bump profile", legend=false, ylims=(0, 0.5),
               xticks=xt, left_margin=8mm)
    p_z = plot(x, abs.(zf), xlabel="position x", ylabel="|z|",
               title="synchrony |z|", legend=false, ylims=(0, 1),
               xticks=xt, left_margin=8mm)
    p_a = plot(x, zargp, xlabel="position x", ylabel="arg(z)",
               title="mean phase arg(z)", legend=false, ylims=(0, 2π),
               xticks=xt, yticks=yt, left_margin=8mm)
    display(p_r); display(p_z); display(p_a)        # individual panels (REPL)
    fig = plot(p_r, p_z, p_a, layout=(3, 1), size=(700, 900))
    savefig(fig, joinpath("figures", "field_profile.png"))
    println("  saved field_profile.png")

    # ---- ANALYSIS figure: persistence / stationarity (NOT the profile) ----
    # Space-time firing-rate map after release: shows the bump is stationary and
    # persists (a ring attractor), which the single-time profile above cannot.
    persist = heatmap(rate.(Zf), xlabel="time step", ylabel="grid point",
                      title="r(x,t) after release (persistence)", clims=(0, maximum(rprof)))
    display(persist)
    savefig(persist, joinpath("figures", "field_persistence.png"))
    println("  saved field_persistence.png")

    # ---- B2: map the bump regime over (η̄, κ) ----
    println("\n== bump regime map (kick-then-release; ✓ = stable bump w/ silent surround) ==")
    η̄s = (-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0)
    κs = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)         # coupling gain α; Laing Fig.1 uses α=2
    print(rpad("η̄ \\ κ", 8)); for κ in κs; print(rpad(κ, 7)); end; println()
    for η̄v in η̄s
        print(rpad(η̄v, 8))
        for κv in κs
            Z = kick_then_release(x, η̄v, Δ, κv, Khat; T_free=150.0)
            _, rmin, _, _, isb = bump_metrics(x, Z[:, end])
            print(rpad(isb ? "✓" : (rmin >= 0.02 ? "flood" : "·"), 7))
        end
        println()
    end
    println("(· = decays to rest;  flood = whole ring active;  ✓ = localized bump)")

    return x
end

x = main()
