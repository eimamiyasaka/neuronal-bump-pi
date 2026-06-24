# ============================================================================
# AUTO/XPPAUT pipeline — Step 5.1 (publication-grade Hopf), bump exporter + validator.
#
# Produces the starting solution and validates the reduced model that the AUTO
# continuation (auto/bump.f90) integrates. NOTHING is ported to Fortran until the
# reduced even-grid RHS here is shown to equal the verified field code to machine
# precision — that check is the go/no-go gate for trusting the AUTO result.
#
# The continuation parameter is the conductance gain g0 under MATCHED reversals
# E_E,E_I = (κ/g0, −κ/g0) — exactly run_regime_map.jl's setup. Under matching, g0
# cancels from the drive and survives only in the shunt term, so the static-bump
# drive is g0-independent and g0 multiplies a single linear term:
#
#   M  = 2π/N                          (ring integration measure)
#   S0 = Σ_k P_k ,  Sc = Σ_k P_k cos x_k ,  Ss ≡ 0   (even bump ⇒ sine-sum vanishes)
#   f_j = η̄ + κ·M·(0.1·S0 + 0.3·cos x_j·Sc)          (= current-based drive)
#   G_j = g0·M·(0.7·S0 + 0.3·cos x_j·Sc)             (shunt, ∝ g0)
#   ż_j = (i/2)(z_j−1)² − ½(z_j+1)²(Δ+i f_j) + (G_j/2)(z_j²−1)
#
# Reflection symmetry about x=π lets us carry only the half-grid j=1..H, H=N/2+1
# (x∈[0,π]); the translation (Goldstone) zero mode is odd and thus excluded, so the
# remaining linearization is non-singular and the only oscillatory instability is the
# even breathing (oscillon) mode — the gamma Hopf we want to locate.
#
# Output: auto/bump_init.dat  — H, params, then H rows "u_j  w_j" (Re/Im z on x∈[0,π]).
# ============================================================================

include("../src/conductance.jl")   # field_positions, rest_state, dale_field,
                                   # seed_field_bump_cond, field_rhs_cond, rate, meanpulse

# Reduced even-grid RHS (matched reversals; g0 only in the shunt). zh, xh on x∈[0,π].
function reduced_rhs(zh::Vector{ComplexF64}, xh::Vector{Float64}, η̄, Δ, κ, g0)
    H = length(zh); N = 2*(H-1); M = 2π/N
    P  = meanpulse.(zh)
    S0 = P[1] + P[H] + 2*sum(@view P[2:H-1])
    Sc = P[1]*cos(xh[1]) + P[H]*cos(xh[H]) + 2*sum((@view P[2:H-1]) .* cos.(@view xh[2:H-1]))
    dz = similar(zh)
    @inbounds for j in 1:H
        c = cos(xh[j])
        f = η̄ + κ*M*(0.1*S0 + 0.3*c*Sc)
        G = g0*M*(0.7*S0 + 0.3*c*Sc)
        z = zh[j]
        dz[j] = (im/2)*(z-1)^2 - 0.5*(z+1)^2*(Δ + im*f) + 0.5*G*(z^2 - 1)
    end
    return dz
end

function main()
    η̄, κ, Δ, Nx = -0.4, 2.0, 0.10, 256        # operating point (run_regime_map.jl)
    g0_start    = 0.05                          # deep in the static pocket (well below g0*≈0.33)
    x = field_positions(Nx)
    E_E, E_I = matched_reversals(κ, g0_start)
    syn = dale_field(x, E_E, E_I; g0=g0_start)

    println("== Seeding conductance static bump (Δ=$Δ, g0=$g0_start, Nx=$Nx) ==")
    z = seed_field_bump_cond(x, η̄, Δ, κ, syn; T_free=max(400.0, 12/Δ), dt=0.02)

    # (1) confirm it is a settled equilibrium of the FULL field
    res_full = field_rhs_cond(z, η̄, Δ, κ, syn)
    println("  max|ż| at seeded bump (full field) = ", maximum(abs, res_full))

    # (2) enforce exact evenness about x=π: z_j = z_{N+2-j}  (reflection index)
    refl(j) = j == 1 ? 1 : Nx + 2 - j
    zsym = [(z[j] + z[refl(j)]) / 2 for j in 1:Nx]
    asym = maximum(abs, z .- zsym)
    println("  pre-symmetrisation asymmetry max|z_j - z_refl| = ", asym)

    H  = Nx ÷ 2 + 1
    xh = x[1:H]                # x ∈ [0, π]
    zh = ComplexF64.(zsym[1:H])

    # (3) GO/NO-GO: reduced even-grid RHS must equal the full field RHS on the half grid
    res_full_sym = field_rhs_cond(zsym, η̄, Δ, κ, syn)
    res_reduced  = reduced_rhs(zh, xh, η̄, Δ, κ, g0_start)
    mism = maximum(abs, res_reduced .- res_full_sym[1:H])
    println("  max|reduced_rhs − full_rhs| on half grid = ", mism)
    @assert mism < 1e-10 "Reduced RHS does not match the field code — fix before Fortran!"

    # (4) the symmetrised bump is still an equilibrium of the reduced model
    println("  max|reduced ż| at symmetrised bump = ", maximum(abs, res_reduced))

    # (5) bump sanity: peak rate, surround rate, contrast
    r = rate.(zsym)
    println("  bump peak rate=", round(maximum(r),digits=4),
            "  surround=", round(minimum(r),digits=4),
            "  contrast=", round(maximum(r)-minimum(r),digits=4))

    # --- export the half-grid initial solution for AUTO (auto/bump.f90 STPNT) ---
    mkpath("auto")
    open(joinpath("auto","bump_init.dat"), "w") do io
        println(io, "# next-gen field static bump, even half-grid x in [0,pi], for AUTO STPNT")
        println(io, "# H Nx etabar Delta kappa g0_start")
        println(io, "$H $Nx $η̄ $Δ $κ $g0_start")
        for j in 1:H
            println(io, "$(real(zh[j])) $(imag(zh[j]))")
        end
    end
    println("  wrote auto/bump_init.dat  (H=$H rows, NDIM=", 2H, ")")
    return nothing
end

main()
