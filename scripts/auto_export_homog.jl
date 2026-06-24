# Compute the spatially-UNIFORM conductance equilibrium (the bump's surround state)
# and export it for homog.f90's STPNT. Matched reversals ⇒ g0-independent drive:
#   P = meanpulse(z),  f = η̄ + 0.2π κ P,  G = 1.4π g0 P
#   ż = (i/2)(z−1)² − ½(z+1)²(Δ+if) + (G/2)(z²−1)
# Relax from the uncoupled rest state to the coupled uniform fixed point.
include("../src/conductance.jl")   # meanpulse, rest_state

function homog_rhs(z, η̄, Δ, κ, g0)
    P = meanpulse(z)
    f = η̄ + 0.2π*κ*P
    G = 1.4π*g0*P
    return (im/2)*(z-1)^2 - 0.5*(z+1)^2*(Δ + im*f) + 0.5*G*(z^2 - 1)
end

function main()
    η̄, Δ, κ, g0 = -0.4, 0.10, 2.0, 0.10
    z, dt = rest_state(η̄, Δ), 0.005
    for _ in 1:200_000
        k1 = homog_rhs(z, η̄, Δ, κ, g0)
        k2 = homog_rhs(z + 0.5dt*k1, η̄, Δ, κ, g0)
        k3 = homog_rhs(z + 0.5dt*k2, η̄, Δ, κ, g0)
        k4 = homog_rhs(z + dt*k3, η̄, Δ, κ, g0)
        z += (dt/6)*(k1 + 2k2 + 2k3 + k4)
    end
    res = abs(homog_rhs(z, η̄, Δ, κ, g0))
    println("uniform equilibrium z=", z, "  |ż|=", res, "  rate=", round(rate(z), digits=4))
    @assert res < 1e-10
    mkpath("auto")
    open(joinpath("auto", "homog_init.dat"), "w") do io
        println(io, "# uniform conductance equilibrium for AUTO homog STPNT")
        println(io, "# etabar Delta kappa g0  then  u w")
        println(io, "$η̄ $Δ $κ $g0")
        println(io, "$(real(z)) $(imag(z))")
    end
    println("wrote auto/homog_init.dat")
end

main()
