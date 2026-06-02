include("../src/ring.jl")
using Plots

function main()
    # --- parameters (locals, not globals) ---
    N  = 512        # network size; start small; scale to 8192 once correct

    η̄  = -0.5       # excitability distribution: Cauchy centre
    Δ  = 0.05       #                            half-width (heterogeneity)

    n = 2           # pulse sharpness
    a_n = 2.0^n * factorial(n)^2 / factorial(2n)   # derived: unit-area normalisation

    # --- D1: build the fixed scaffolding ---
    x = make_positions(N)
    η = make_excitabilities(N, η̄, Δ)
    K = make_kernel(x)

    # --- Test D1: kernel row should look like a cosine ---
    p = plot(x, K[1, :],
             xlabel = "position xₖ", ylabel = "K[1, k]",
             title = "Kernel seen by neuron 1", legend = false)
    display(p)

    # --- Test D2: drive from a known θ should be finite & kernel-shaped ---
    θ_test = fill(float(π), N)      # every neuron at peak pulse
    I = drive(θ_test, K, a_n, n)
    println("I finite: ", all(isfinite, I),
            "   range: ", round(minimum(I), digits=4), " ... ", round(maximum(I), digits=4))
    # all-at-π => P constant => I should be nearly flat (cos part averages out)

    # println(length(η) == N)
    # display(histogram(η))

    return x, η, K   # hand back for later steps
end

x, η, K = main()