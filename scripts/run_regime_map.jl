# Phase-5, Step 5.1 — regime map: locate the static-bump / oscillon (gamma) / collapse
# boundaries by direct simulation of the exact field, in the (g0, Δ) plane, with the PI
# operating point marked. This is the SIMULATION-BASED first-pass boundary; a publication-
# grade Hopf curve wants AUTO/continuation (deps not present here) — see step5_gate.md.
#
# Method (correctness-critical): each point is SETTLED with T_free ≳ 12/Δ before measuring,
# because at small Δ the field is under-damped and a slow transient ("fishbone", CLAUDE.md)
# masquerades as oscillation if measured too early. classify_regime then reads sustained
# breathing from breathing_amplitude (src/phase5.jl). The low-Δ/high-κ "oscillations" seen
# at short T are exactly such transients (verified decaying under long T), so they are
# correctly excluded here and the genuine, sustained Hopf rides the shunting axis g0.
#
# Operating point: Δ=0.1, κ=2, η̄=−0.4, n=2, Nx=256.

include("../src/phase5.jl")        # breathing_amplitude, classify_regime (+ conductance, core)
using Plots
using Plots.PlotMeasures

function regime_at(x, η̄, Δ, κ, g0; dt=0.02)
    E_E, E_I = matched_reversals(κ, g0)
    syn   = dale_field(x, E_E, E_I; g0=g0)
    T_set = max(300.0, 12/Δ)
    zs    = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=T_set, dt=dt)
    b     = breathing_amplitude(zs, z -> field_step_cond(z, η̄, Δ, κ, syn, dt); T=800.0, dt=dt)
    return classify_regime(b), b
end

function main()
    η̄, κ, n, dt, Nx = -0.4, 2.0, 2, 0.02, 256
    x = field_positions(Nx)

    Δs  = [0.05, 0.10, 0.15, 0.20]
    g0s = [0.10, 0.20, 0.28, 0.32, 0.36, 0.40, 0.45]
    color_of = Dict(:static => :seagreen, :oscillon => :crimson, :collapsed => :gray)

    println("== Step 5.1: regime map over (g0, Δ) — settled T_free≳12/Δ ==")
    println(rpad("Δ\\g0", 8), join([rpad(string(g0), 8) for g0 in g0s]))
    pts_g0 = Float64[]; pts_Δ = Float64[]; pts_c = Symbol[]
    for Δ in Δs
        row = String[]
        for g0 in g0s
            reg, _ = regime_at(x, η̄, Δ, κ, g0; dt=dt)
            push!(pts_g0, g0); push!(pts_Δ, Δ); push!(pts_c, reg)
            push!(row, rpad(reg == :static ? "stat" : reg == :oscillon ? "OSC" : "coll", 8))
        end
        println(rpad(string(Δ), 8), join(row))
    end

    # --- 1-D detail at the operating Δ=0.1: breathing amplitude vs g0 (the Hopf) ---
    println("\n== breathing amplitude vs g0 at Δ=0.1 ==")
    g0_fine = collect(0.20:0.02:0.44)
    osc = Float64[]
    for g0 in g0_fine
        _, b = regime_at(x, η̄, 0.10, κ, g0; dt=dt)
        push!(osc, b.osc)
        println("  g0=", rpad(g0,5), "  osc=", round(b.osc, digits=3))
    end
    g0_star = g0_fine[findfirst(o -> o > 0.02, osc)]
    println("  ⇒ Hopf g0* ≈ ", g0_star)

    # ------------------------------------------------------------------ figure
    # Panel (a): (g0,Δ) regime scatter; Panel (b): osc(g0) at Δ=0.1 with Hopf marked.
    pa = plot(xlabel="conductance gain g0  (shunt strength)", ylabel="heterogeneity Δ",
              title="(a) regime map  (●static  ●oscillon/gamma  ●collapsed)",
              legend=false, xlims=(0.05, 0.5), ylims=(0.02, 0.23), left_margin=8mm, bottom_margin=5mm)
    for reg in (:static, :oscillon, :collapsed)
        idx = findall(==(reg), pts_c)
        isempty(idx) && continue
        scatter!(pa, pts_g0[idx], pts_Δ[idx], mc=color_of[reg], ms=9, msw=0.5,
                 markershape=(reg==:static ? :circle : reg==:oscillon ? :diamond : :xcross))
    end
    scatter!(pa, [0.10], [0.10], mc=:gold, markershape=:star5, ms=14, msw=1.0)   # operating point
    annotate!(pa, 0.10, 0.122, text("operating\npoint", 8, :gold))

    pb = plot(g0_fine, osc, m=:circle, lc=:crimson, mc=:crimson, lw=2,
              xlabel="conductance gain g0", ylabel="breathing amplitude (peak-rate osc)",
              title="(b) gamma Hopf along the shunting axis (Δ=0.1)", legend=:topleft,
              label="sustained osc amplitude", left_margin=9mm, bottom_margin=5mm)
    vline!(pb, [g0_star], lc=:crimson, ls=:dash, lw=1.2, label="Hopf g0*≈$(g0_star)")
    hline!(pb, [0.02], lc=:gray, ls=:dot, lw=1, label="detection floor")

    fig = plot(pa, pb, layout=(1,2), size=(1180, 460))
    savefig(fig, joinpath("figures", "phase5_regime_map.png"))
    println("\n  saved figures/phase5_regime_map.png")
    return (g0_star=g0_star,)
end

main()
