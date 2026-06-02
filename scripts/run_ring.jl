include("../src/ring.jl")
using Plots

function main()
    # --- parameters (locals, not globals) ---
    N  = 512        # start small; scale to 8192 once correct
    η̄  = -0.5
    Δ  = 0.05

    # --- D1: build the fixed scaffolding ---
    x = make_positions(N)
    η = make_excitabilities(N, η̄, Δ)
    K = make_kernel(x)

    # --- Test D1: kernel row should look like a cosine ---
    p = plot(x, K[1, :],
             xlabel = "position xₖ", ylabel = "K[1, k]",
             title = "Kernel seen by neuron 1", legend = false)
    display(p)

    # println(length(η) == N)
    # display(histogram(η))

    return x, η, K   # hand back for later steps
end

x, η, K = main()