include("../src/single.jl")
using Plots

# --- Test 1: suprathreshold, should spike periodically ---
ts, θtr, spk = simulate_single(0.5; T=100.0, dt=0.01)
println("η = 0.5: ", length(spk), " spikes, rate ≈ ",
        round(length(spk)/100.0, digits=3), " per unit time")

p1 = plot(ts, θtr, xlabel="time", ylabel="θ", title="η = 0.5 (spiking)",
        legend=false)
hline!(p1, [π], linestyle=:dash)

# --- Test 2: subthreshold, should fall silent ---
ts2, θtr2, spk2 = simulate_single(-0.5; T=100.0, dt=0.01)
println("η = -0.5: ", length(spk2), " spikes (expect 0 after transient)")

p2 = plot(ts2, θtr2, xlabel="time", ylabel="θ", title="η = -0.5 (silent)",
        legend=false)

plot(p1, p2, layout=(2, 1), size=(600, 500))
