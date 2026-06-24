# Export validated static bumps at a LIST of Δ (all at g0=0.05), each an
# independently-seeded, symmetrised, machine-precision equilibrium — the robust
# starting solutions for the (g0,Δ) Hopf-locus sweep (auto/run_hopf_locus.py).
# Re-seeding per Δ avoids the branch-contamination that Δ-continuation suffers
# (the bump branch folds at high Δ and returns on the unstable branch).
#
# Writes auto/bump_init_D###.dat (### = 1000Δ) + auto/locus_targets.txt manifest.
include("../src/conductance.jl")

# reduced_rhs is identical to scripts/auto_export_bump.jl; redefined here so this
# script is standalone.
function reduced_rhs(zh::Vector{ComplexF64}, xh::Vector{Float64}, η̄, Δ, κ, g0)
    H = length(zh); N = 2*(H-1); M = 2π/N
    P  = meanpulse.(zh)
    S0 = P[1] + P[H] + 2*sum(@view P[2:H-1])
    Sc = P[1]*cos(xh[1]) + P[H]*cos(xh[H]) + 2*sum((@view P[2:H-1]) .* cos.(@view xh[2:H-1]))
    dz = similar(zh)
    @inbounds for j in 1:H
        c = cos(xh[j]); f = η̄ + κ*M*(0.1*S0 + 0.3*c*Sc); G = g0*M*(0.7*S0 + 0.3*c*Sc)
        z = zh[j]
        dz[j] = (im/2)*(z-1)^2 - 0.5*(z+1)^2*(Δ + im*f) + 0.5*G*(z^2 - 1)
    end
    return dz
end

function export_one(Δ; η̄=-0.4, κ=2.0, Nx=256, g0=0.05)
    x = field_positions(Nx)
    E_E, E_I = matched_reversals(κ, g0)
    syn = dale_field(x, E_E, E_I; g0=g0)
    z = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=max(500.0, 14/Δ), dt=0.02)
    refl(j) = j == 1 ? 1 : Nx + 2 - j
    zsym = [(z[j] + z[refl(j)]) / 2 for j in 1:Nx]
    H = Nx ÷ 2 + 1; xh = x[1:H]; zh = ComplexF64.(zsym[1:H])
    mism = maximum(abs, reduced_rhs(zh, xh, η̄, Δ, κ, g0) .- field_rhs_cond(zsym, η̄, Δ, κ, syn)[1:H])
    res  = maximum(abs, reduced_rhs(zh, xh, η̄, Δ, κ, g0))
    r = rate.(zsym)
    @assert mism < 1e-9 "Δ=$Δ: reduced/full mismatch $mism"
    if maximum(r) - minimum(r) < 0.05            # no localized bump (past the existence fold)
        println("  Δ=$Δ  contrast≈0 — bump does not exist at g0=$g0; skipping")
        return nothing
    end
    fname = "bump_init_D$(lpad(round(Int, 1000Δ), 3, '0')).dat"
    open(joinpath("auto", fname), "w") do io
        println(io, "# static bump even half-grid, Δ=$Δ g0=$g0")
        println(io, "# H Nx etabar Delta kappa g0")
        println(io, "$H $Nx $η̄ $Δ $κ $g0")
        for j in 1:H; println(io, "$(real(zh[j])) $(imag(zh[j]))"); end
    end
    println("  Δ=$Δ  |reduced ż|=$(round(res,sigdigits=2))  mism=$(round(mism,sigdigits=2))  ",
            "peak=$(round(maximum(r),digits=3))  contrast=$(round(maximum(r)-minimum(r),digits=3))  -> auto/$fname")
    return Δ, fname
end

function main()
    mkpath("auto")
    Δs = [0.05, 0.07, 0.10, 0.13, 0.16, 0.18, 0.20]
    println("== exporting locus starting bumps (g0=0.05) ==")
    rows = filter(!isnothing, [export_one(Δ) for Δ in Δs])
    open(joinpath("auto", "locus_targets.txt"), "w") do io
        for (Δ, fname) in rows; println(io, "$Δ $fname"); end
    end
    println("wrote auto/locus_targets.txt ($(length(rows)) targets)")
end

main()
