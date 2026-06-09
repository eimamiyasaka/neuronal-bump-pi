using FFTW

# ============================================================================
# Macroscopic next-generation (Ott–Antonsen) neural field for the theta-neuron
# ring — the exact mean-field reduction of the spiking network in src/ring.jl.
#
# State: complex order parameter  z(x,t) = <e^{-iθ}>  at each grid point x.
#
#   ż = (i/2)(z-1)² − (1/2)(z+1)²·[ Δ + i(η̄ + κ·I(x,t)) ]
#
# Derivation: write θ̇ = (1+f) − (1-f)cos θ, f = η + κI, in harmonics, apply the
# OA ansatz, and integrate over the Lorentzian η (median η̄, HWHM Δ) by closing
# the contour at the pole η = η̄ − iΔ. See notes/project.md §4.
#
# CONVENTION (matters for comparison to the paper): we track z = <e^{-iθ}>, which
# is the complex CONJUGATE of Laing & Omel'chenko's z = <e^{+iθ}> (their Eq. 5).
# The equation above is the exact conjugate of their Eq. 5, so |z| and the firing
# rate are IDENTICAL to theirs; only arg z is sign-flipped. Negate arg z (see
# `arg_laing`) when overlaying on their Fig. 1/2 row 4.
#
# Readouts (closed form in z):
#   firing rate   r = (1/π) Re[(1-z)/(1+z)]     (Laing Eq. 9)
#   mean pulse    P̄ = a_n <(1-cosθ)^n>          (used for the coupling, Laing Eq. 2)
# ============================================================================

# Grid of Nx evenly spaced positions on the ring [0, 2π)
field_positions(Nx) = [2π*(j-1)/Nx for j in 1:Nx]

# Population firing rate from the order parameter: the flux of neurons through
# θ = π. This is Laing & Omel'chenko Eq. (9) — the *instantaneous* frequency
# f(x,t). They write it as (1/π)Re[(1-z̄)/(1+z̄)] for their z = <e^{+iθ}>; since
# our z = <e^{-iθ}> is their z̄, (1-z)/(1+z) here equals their (1-z̄)/(1+z̄), so
# this returns exactly the same number.
rate(z) = (1/π) * real((1 - z) / (1 + z))

# Argument of z in Laing & Omel'chenko's convention (z = <e^{+iθ}>). Our z is the
# conjugate <e^{-iθ}>, so arg flips sign; negate to plot/compare against their
# Fig. 1/2 row 4. (|z| and rate(z) need no such correction — see header.)
arg_laing(z) = -angle(z)

# Angular (shortest-arc) distance on the ring [0, 2π). Shared by the seeding
# kicks in both run_field.jl and run_ring.jl (both `include` this file).
function angular_distance(x, x0)
    d = abs.(x .- x0)
    return min.(d, 2π .- d)
end

# Mean pulse P̄ = a_n <(1-cosθ)^n>, written in z for n = 2 (a_2 = 2/3):
#   <(1-cosθ)²> = 3/2 − 2 Re z + (1/2) Re z²   ⇒   P̄ = 1 − (4/3)Re z + (1/3)Re z²
# (matches pulse() in src/ring.jl in the mean-field sense)
meanpulse(z) = 1 - (4/3)*real(z) + (1/3)*real(z^2)

# Coupling kernel sampled on the grid offsets, ready for FFT convolution.
# K(x) = 0.1 + 0.3 cos x + B sin x   (B = 0 ⇒ symmetric; B ≠ 0 ⇒ velocity asymmetry)
field_kernel(x; B=0.0) = 0.1 .+ 0.3 .* cos.(x) .+ B .* sin.(x)

# Coupling field  I(x) = (1/Nx) Σ_k K(x − x_k) P̄_k  via circular FFT convolution.
# Khat = fft(field_kernel(x)) is precomputed once (kernel is time-independent).
function coupling_field(z, Khat)
    P = meanpulse.(z)
    return real(ifft(Khat .* fft(P))) .* (2π / length(z))
end

# Field right-hand side: ż at every grid point. Iext is an optional external
# input field (scalar or per-position vector), e.g. a transient bump-seeding kick.
function field_rhs(z, η̄, Δ, κ, Khat; Iext=0.0)
    I = coupling_field(z, Khat) .+ Iext
    f = η̄ .+ κ .* I                         # total drive felt by each population
    return (im/2) .* (z .- 1).^2 .- 0.5 .* (z .+ 1).^2 .* (Δ .+ im .* f)
end

# One RK4 step of size dt for the whole field
function field_step(z, η̄, Δ, κ, Khat, dt; Iext=0.0)
    k1 = field_rhs(z,                η̄, Δ, κ, Khat; Iext=Iext)
    k2 = field_rhs(z .+ 0.5dt .* k1, η̄, Δ, κ, Khat; Iext=Iext)
    k3 = field_rhs(z .+ 0.5dt .* k2, η̄, Δ, κ, Khat; Iext=Iext)
    k4 = field_rhs(z .+ dt  .* k3,   η̄, Δ, κ, Khat; Iext=Iext)
    return z .+ (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
end

# Integrate the field for duration T; Iext is held fixed across the run (use a
# vector to impose a localized external drive, or 0.0 for the free network).
function simulate_field(z0, η̄, Δ, κ, Khat; T=100.0, dt=0.01, Iext=0.0)
    nt = round(Int, T/dt)
    z = copy(z0)
    Z = Matrix{ComplexF64}(undef, length(z0), nt)
    for i in 1:nt
        z = field_step(z, η̄, Δ, κ, Khat, dt; Iext=Iext)
        Z[:, i] = z
    end
    return Z
end

# Rest-state order parameter: the spatially uniform quiescent fixed point,
# obtained by relaxing a single population with no coupling (I = 0).
function rest_state(η̄, Δ; T=200.0, dt=0.005)
    z = 0.0 + 0.0im
    nt = round(Int, T/dt)
    for _ in 1:nt
        f(zz) = (im/2)*(zz-1)^2 - 0.5*(zz+1)^2*(Δ + im*η̄)
        k1 = f(z); k2 = f(z + 0.5dt*k1); k3 = f(z + 0.5dt*k2); k4 = f(z + dt*k3)
        z += (dt/6)*(k1 + 2k2 + 2k3 + k4)
    end
    return z
end
