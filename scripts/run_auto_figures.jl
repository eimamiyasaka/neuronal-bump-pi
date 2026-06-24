# Phase-5 AUTO figures: render the continuation results (auto/*.csv) into the
# publication-grade Hopf-boundary figures. Pure presentation — all numerics are
# produced by the auto-07p drivers (auto/run_bump.py, auto/run_homog.py).
#
#   figures/phase5_auto_hopf.png   (a) 1-param bump branch vs g0 with the oscillon
#                                      Hopf g0* marked; (b) (g0,Δ) Hopf locus:
#                                      continuation boundary vs simulation regimes.
#   figures/phase5_auto_homog.png  warm-up/baseline: uniform S-curve vs η̄.
using Plots, Plots.PlotMeasures, DelimitedFiles

readcsv(path) = readdlm(path, ',', Float64; skipstart=1)

const AUTO = joinpath(@__DIR__, "..", "auto")
const FIG  = joinpath(@__DIR__, "..", "figures")

function main()
    # ---- (a) 1-parameter branch: L2-norm vs g0, Hopf marked ----
    br = readcsv(joinpath(AUTO, "bump_branch.csv"))     # g0, L2norm
    g0star = 0.3303      # from continuation (auto/run_bump.py); printed as the headline
    pa = plot(br[:,1], br[:,2], lw=2, lc=:steelblue, label="bump branch (eq.)",
              xlabel="conductance gain g0  (shunt strength)", ylabel="solution L2-norm",
              title="(a) static bump → oscillon Hopf along g0  (Δ=0.1)",
              legend=:topleft, left_margin=8mm, bottom_margin=5mm)
    vline!(pa, [g0star], lc=:crimson, ls=:dash, lw=2, label="oscillon Hopf g0*=$g0star")
    vline!(pa, [0.10], lc=:gold, ls=:dot, lw=2, label="operating point g0=0.1")

    # ---- (b) two-parameter (g0,Δ) Hopf locus: continuation vs simulation ----
    hp = readcsv(joinpath(AUTO, "bump_hopf.csv"))        # g0, Delta
    perm = sortperm(hp[:,2])
    pb = plot(hp[perm,1], hp[perm,2], lw=3, lc=:crimson, label="Hopf locus (continuation)",
              xlabel="conductance gain g0", ylabel="heterogeneity Δ",
              title="(b) (g0,Δ) oscillon-Hopf boundary: AUTO vs simulation",
              legend=:bottomright, left_margin=8mm, bottom_margin=5mm)
    # trusted simulation anchor: the settled-sweep Hopf at Δ=0.1 (run_regime_map.jl)
    scatter!(pb, [0.33], [0.10], mc=:seagreen, ms=9, markershape=:utriangle,
             label="sim Hopf g0*≈0.33 (Δ=0.1)")
    scatter!(pb, [0.10], [0.10], mc=:gold, ms=11, markershape=:star5, label="operating point")
    plot!(pb, xlims=(0.07, 0.37), ylims=(0.045, 0.135))
    annotate!(pb, 0.135, 0.075, text("static\nbump", 9, :seagreen))
    annotate!(pb, 0.345, 0.108, text("oscillon\n(breathing)", 9, :crimson))

    fig = plot(pa, pb, layout=(1,2), size=(1240, 480))
    savefig(fig, joinpath(FIG, "phase5_auto_hopf.png"))
    println("saved figures/phase5_auto_hopf.png")

    # ---- warm-up / baseline: uniform S-curve vs η̄ ----
    if isfile(joinpath(AUTO, "homog_eta.csv"))
        he = readcsv(joinpath(AUTO, "homog_eta.csv"))    # etabar, rate
        ph = plot(he[:,1], he[:,2], lw=2, lc=:purple, label="uniform equilibrium",
                  xlabel="mean drive η̄", ylabel="uniform firing rate",
                  title="warm-up: homogeneous S-curve (bulk bistability)",
                  legend=:topleft, size=(640, 460), left_margin=8mm, bottom_margin=5mm)
        vline!(ph, [-0.4], lc=:gold, ls=:dot, lw=2, label="operating η̄=-0.4")
        savefig(ph, joinpath(FIG, "phase5_auto_homog.png"))
        println("saved figures/phase5_auto_homog.png")
    end
end

main()
