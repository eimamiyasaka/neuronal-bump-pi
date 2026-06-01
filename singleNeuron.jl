using Plots

# --- Theta-neuron right-hand side ---
# θ̇ = (1 - cos θ) + (1 + cos θ) * η
thetadot(θ, η) = (1-cos(θ)) + (1 + cos(θ)) * η

# --- One RK4 step of size dt ---
function rk4_step(θ, η, dt)
    k1 = thetadot(θ,            η)
    k2 = thetadot(θ + 0.5dt*k1, η)
    k3 = thetadot(θ + 0.5dt*k2, η)
    k4 = thetadot(θ + dt*k3,    η)
    return θ + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# --- Simulate one neuron, return time, θ-trace, and spike times ---
function simulate(η; T=100.0, dt=0.01, θ0=0.0)
    nt = round(Int, T/dt)
    θ = θ0
    ts     = Vector{Float64}(undef, nt)
    θtrace = Vector{Float64}(undef, nt)
    spikes = Float64[]

    for i in 1:nt
        θ_new = rk4_step(θ, η, dt)
        # spike if θ crossed π upward during this step
        if θ < π && θ_new >= π
            push!(spikes, i*dt)
        end
        θ = mod(θ_new, 2π)
        ts[i]     = i*dt
        θtrace[i] = θ
    end
    return ts, θtrace, spikes
end

# --- Test 1: suprathreshold, should spike periodically ---
ts, θtr, spk = simulate(0.5; T=100.0, dt=0.01)
println("η = 0.5: ", length(spk), " spikes, rate ≈ ",
        round(length(spk)/100.0, digits=3), " per unit time")

p1 = plot(ts, θtr, xlabel="time", ylabel="θ", title="η = 0.5 (spiking)",
        legend=false)
hline!(p1, [π], linestyle=:dash)

# --- Test 2: subthreshold, should fall silent ---
ts2, θtr2, spk2 = simulate(-0.5; T=100.0, dt=0.01)
println("η = -0.5: ", length(spk2), " spikes (expect 0 after transient)")

p2 = plot(ts2, θtr2, xlabel="time", ylabel="θ", title="η = -0.5 (silent)",
        legend=false)


plot(p1, p2, layout=(2, 1), size=(600, 500))