include("../src/ring.jl")
using Plots

# Angular distance calculate (helper since neurons are on a ring)
function angular_distance(x, x0)
    d = abs.(x .- x0)
    return min.(d, 2π .- d)
end

function main()
    # --- parameters (locals, not globals) ---
    N  = 512        # network size; start small; scale to 8192 once correct

    η̄  = -0.4       # excitability distribution: Cauchy centre
    Δ  = 0.01       #                            half-width (heterogeneity)

    κ = 2.0         # coupling strength (global dial)

    n = 2           # pulse sharpness
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)   # derived: unit-area normalisation

    # --- D1: build the fixed scaffolding ---
    x = make_positions(N)
    η = make_excitabilities(N, η̄, Δ)
    K = make_kernel(x)

#     # --- Test D1: kernel row should look like a cosine ---
#     p = plot(x, K[1, :],
#              xlabel = "position xₖ", ylabel = "K[1, k]",
#              title = "Kernel seen by neuron 1", legend = false)
#     display(p)

#     # println(length(η) == N)
#     # display(histogram(η))

#     # --- Test D2: drive from a known θ should be finite & kernel-shaped ---
#     θ_test = fill(float(π), N)      # every neuron at peak pulse
#     I = drive(θ_test, K, a_n, n)
#     println("I finite: ", all(isfinite, I),
#             "   range: ", round(minimum(I), digits=4), " ... ", round(maximum(I), digits=4))
#     # all-at-π => P constant => I should be nearly flat (cos part averages out)

#     # --- Test D3: one population step should move θ and stay finite ---
#     θ1 = step_population(θ_test, η, K, a_n, n, κ, 0.01)
#     println("θ changed: ", θ1 != θ_test,
#             "   all finite: ", all(isfinite, θ1),
#             "   range: ", round(minimum(θ1), digits=3), " ... ", round(maximum(θ1), digits=3))

    # --- D4: seed a localized bump and simulate the population ---
    x0 = π                  # bump center (middle of the ring)
    σ  = 0.5                # bump width
    θ0 = π .* exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))

    Θ_agg = simulate_population(θ0, η, K, a_n, n, κ; T=50.0, dt=0.01)

    # Readout: activity heatmap (neuron on y, time on x)
    activity = 1 .- cos.(Θ_agg)
    h = heatmap(activity, xlabel="time step", ylabel="neuron",
                title="bump activity (1 - cos θ)")
    display(h)
    savefig(h, joinpath("figures", "d4_bump.png"))
    println("D4 done: Θ_agg size ", size(Θ_agg),
            "   activity range ", round(minimum(activity), digits=3),
            " ... ", round(maximum(activity), digits=3))

    return x, η, K   # hand back for later steps
end

x, η, K = main()