# ============================================================================
# Theta-neuron model + single-neuron integration.
#
# This file holds the canonical theta-neuron right-hand side (`thetadot`), which
# is shared with the ring network in src/ring.jl (that file `include`s this one),
# plus the uncoupled single-neuron RK4 stepper and a single-neuron simulator used
# by scripts/run_single.jl for sanity checks.
# ============================================================================

# --- Theta-neuron right-hand side ---
# θ̇ = (1 - cos θ) + (1 + cos θ) * η
thetadot(θ, η) = (1-cos(θ)) + (1 + cos(θ)) * η

# --- One RK4 step of size dt (uncoupled single neuron: η held constant) ---
function rk4_step(θ, η, dt)
    k1 = thetadot.(θ,            η)
    k2 = thetadot.(θ + 0.5dt*k1, η)
    k3 = thetadot.(θ + 0.5dt*k2, η)
    k4 = thetadot.(θ + dt*k3,    η)
    return θ + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# --- Simulate one neuron, return time, θ-trace, and spike times ---
function simulate_single(η; T=100.0, dt=0.01, θ0=0.0)
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
