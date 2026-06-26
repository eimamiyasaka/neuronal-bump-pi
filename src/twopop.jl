# ============================================================================
# EXPLORATORY two-population (E–I) next-generation neural-field SPIKE — FIELD ONLY.
#
# Status: this is an EXPLORATORY field-level spike, NOT a gated result. It probes
# whether a genuinely DIVISIVE shunting-inhibition regime with an intact bump is
# reachable in a TWO-POPULATION excitatory–inhibitory architecture — something the
# single-population Dale-split-kernel model (src/conductance.jl) can only do in a
# BOUNDED way (aI ≲ 0.1, see notes/writeupAssist/step5_divisive_gate.md). The
# microscopic spiking model and the micro/macro agreement gate are OUT OF SCOPE
# here and are named as required future work in the write-up.
#
# This layer MIRRORS src/conductance.jl, generalised from ONE order-parameter field
# z(x,t) to TWO coupled fields z_E(x,t), z_I(x,t) — the exact OA reduction of a
# two-population theta/QIF network with conductance (Dale) synapses. Each population
# obeys the SAME closed field equation as the single-population conductance model
# (one shunting term (G/2)(z²−1)); the only new structure is that E and I are
# distinct fields with FOUR first-harmonic (rank-3) conductance kernels between them:
#
#   for a ∈ {E, I}:
#     g^E_a = g0·(2π/N) Σ W_{aE}(x−y) P_E(y)      excitatory conductance onto a (≥0)
#     g^I_a = g0·(2π/N) Σ W_{aI}(x−y) P_I(y)      inhibitory conductance onto a (≥0)
#     J_a   = E_exc·g^E_a + E_inh·g^I_a           reversal-weighted drive  (E_exc>0>E_inh)
#     G_a   = g^E_a + g^I_a                        total conductance (shunt)
#     ż_a   = (i/2)(z_a−1)² − ½(z_a+1)²[Δ_a + i(η̄_a + J_a)] + (G_a/2)(z_a²−1)
#
#   W_{ab} = c_{ab} + a1_{ab}·cos(x−y)  (kernel FROM population b ONTO population a),
#   ALL coefficients ≥ 0 with c ≥ a1 ≥ 0 so every conductance kernel is ≥ 0 (Dale).
#   Classic E–I Mexican-hat: W_EE / W_IE = localized excitation (E drives both pops),
#   W_EI / W_II = inhibition (I shunts both pops); the DIVISIVE knob is the inhibitory
#   conductance gain (g0 and/or the I-kernel strength). Unlike the single kernel, the
#   I→E kernel W_EI is non-negativity-bounded ONLY by ITSELF (c_EI ≥ a1_EI), not tied
#   to the excitatory kernel by a net-kernel constraint — so it can be far more
#   strongly localized than the single-population aI ≲ 0.1 cap.
#
# Reduction to the validated single-population bump (the mandatory validation anchor,
# see scripts/run_twopop_spike.jl Part 0): make the two populations IDENTICAL clones
# (η̄_E=η̄_I, Δ_E=Δ_I, matched reversals) with W_EE=W_IE=K_E, W_EI=W_II=K_I (the
# conductance.jl Dale split) and seed BOTH with the same kick. Then z_I ≡ z_E for all
# time and z_E's equation is bit-identical to field_rhs_cond — so rate(z_E) reproduces
# seed_field_bump_cond to machine precision.
#
# Layer convention (as moving.jl / conductance.jl / phase5.jl): MODEL/MEASUREMENT
# only — NO Plots; parameters are passed explicitly (no globals); FFT convolution
# with the 2π/N integration measure; RK4 recomputing the coupling at every stage.
# ============================================================================

include("phase5.jl")   # full field core (rate, meanpulse, coupling_field, rest_state,
                        # angular_distance, field_positions, bump_centroid, lateral_speed,
                        # matched_reversals) + classify_regime / breathing_amplitude readouts.
                        # This exploratory layer adds the two-population field on top.

# --- Two-population Dale-split synapse config (field side) -------------------
# Stores the four kernel transforms (fft for the FFT convolution in coupling_field)
# plus the shared conductance gain g0 and reversal potentials. Tiny (four length-Nx
# complex vectors). W_{ab} = kernel from population b onto population a.
struct TwoPopFieldSyn
    WhatEE::Vector{ComplexF64}     # fft(W_EE): excitation onto E from E
    WhatEI::Vector{ComplexF64}     # fft(W_EI): inhibition onto E from I
    WhatIE::Vector{ComplexF64}     # fft(W_IE): excitation onto I from E
    WhatII::Vector{ComplexF64}     # fft(W_II): inhibition onto I from I
    g0::Float64                    # conductance gain (shunt strength)
    E_exc::Float64                 # excitatory reversal (> 0)
    E_inh::Float64                 # inhibitory reversal (< 0)
end

# Build a config from first-harmonic kernel coefficients W_ab = c_ab + a1_ab·cos x.
# Caller is responsible for Dale non-negativity (c ≥ a1 ≥ 0 ⇒ W ≥ 0 everywhere); the
# script asserts the resulting conductances stay ≥ 0. Defaults give the conductance.jl
# Dale split on BOTH pathways (W_EE=W_IE=K_E=0.4+0.3cos, W_EI=W_II=K_I=0.3) — i.e. the
# validation-anchor clone config when the two populations are otherwise identical.
function twopop_field(x; g0=1.0, E_exc=20.0, E_inh=-20.0,
                      cEE=0.4, aEE=0.3, cEI=0.3, aEI=0.0,
                      cIE=0.4, aIE=0.3, cII=0.3, aII=0.0)
    WhatEE = fft(cEE .+ aEE .* cos.(x))
    WhatEI = fft(cEI .+ aEI .* cos.(x))
    WhatIE = fft(cIE .+ aIE .* cos.(x))
    WhatII = fft(cII .+ aII .* cos.(x))
    return TwoPopFieldSyn(WhatEE, WhatEI, WhatIE, WhatII, g0, E_exc, E_inh)
end

# --- The four conductances at a state (zE, zI). All ≥ 0 (kernels ≥ 0, P ≥ 0). ----
# Returned named for clarity; also used by the run script's Dale sanity check.
function twopop_conductances(zE, zI, syn::TwoPopFieldSyn)
    gE_onE = syn.g0 .* coupling_field(zE, syn.WhatEE)   # exc onto E (from E)
    gI_onE = syn.g0 .* coupling_field(zI, syn.WhatEI)   # inh onto E (from I)
    gE_onI = syn.g0 .* coupling_field(zE, syn.WhatIE)   # exc onto I (from E)
    gI_onI = syn.g0 .* coupling_field(zI, syn.WhatII)   # inh onto I (from I)
    return (gE_onE=gE_onE, gI_onE=gI_onE, gE_onI=gE_onI, gI_onI=gI_onI)
end

# ============================================================================
# Field RHS for the pair. Mirrors field_rhs_cond TWICE (once per population) with the
# cross-coupling above; returns (żE, żI). Iext{E,I} are optional external kicks (used
# by the seeding protocol). `shunt` toggles the (G/2)(z²−1) term off (debug / the
# matched-reversal exact closure check).
# ============================================================================
function twopop_rhs(zE, zI, η̄E, η̄I, ΔE, ΔI, κ, syn::TwoPopFieldSyn;
                    IextE=0.0, IextI=0.0, shunt::Bool=true)
    g = twopop_conductances(zE, zI, syn)
    fE = η̄E .+ syn.E_exc .* g.gE_onE .+ syn.E_inh .* g.gI_onE .+ κ .* IextE   # η̄_E + J_E
    fI = η̄I .+ syn.E_exc .* g.gE_onI .+ syn.E_inh .* g.gI_onI .+ κ .* IextI   # η̄_I + J_I
    żE = (im/2) .* (zE .- 1).^2 .- 0.5 .* (zE .+ 1).^2 .* (ΔE .+ im .* fE)
    żI = (im/2) .* (zI .- 1).^2 .- 0.5 .* (zI .+ 1).^2 .* (ΔI .+ im .* fI)
    if shunt
        GE = g.gE_onE .+ g.gI_onE
        GI = g.gE_onI .+ g.gI_onI
        żE = żE .+ 0.5 .* GE .* (zE.^2 .- 1)
        żI = żI .+ 0.5 .* GI .* (zI.^2 .- 1)
    end
    return żE, żI
end

# One RK4 step of the pair (recomputes the coupling at all four stages).
function twopop_step(zE, zI, η̄E, η̄I, ΔE, ΔI, κ, syn, dt;
                     IextE=0.0, IextI=0.0, shunt::Bool=true)
    k1E, k1I = twopop_rhs(zE,              zI,              η̄E,η̄I,ΔE,ΔI,κ,syn; IextE=IextE,IextI=IextI,shunt=shunt)
    k2E, k2I = twopop_rhs(zE.+0.5dt.*k1E,  zI.+0.5dt.*k1I,  η̄E,η̄I,ΔE,ΔI,κ,syn; IextE=IextE,IextI=IextI,shunt=shunt)
    k3E, k3I = twopop_rhs(zE.+0.5dt.*k2E,  zI.+0.5dt.*k2I,  η̄E,η̄I,ΔE,ΔI,κ,syn; IextE=IextE,IextI=IextI,shunt=shunt)
    k4E, k4I = twopop_rhs(zE.+dt.*k3E,     zI.+dt.*k3I,     η̄E,η̄I,ΔE,ΔI,κ,syn; IextE=IextE,IextI=IextI,shunt=shunt)
    zEn = zE .+ (dt/6) .* (k1E .+ 2k2E .+ 2k3E .+ k4E)
    zIn = zI .+ (dt/6) .* (k1I .+ 2k2I .+ 2k3I .+ k4I)
    return zEn, zIn
end

# Evolve the pair for duration T, returning only the final (zE, zI) (O(Nx) memory).
function evolve_twopop(zE0, zI0, η̄E, η̄I, ΔE, ΔI, κ, syn;
                       T=200.0, dt=0.02, IextE=0.0, IextI=0.0, shunt::Bool=true)
    nt = round(Int, T/dt)
    zE, zI = copy(zE0), copy(zI0)
    for _ in 1:nt
        zE, zI = twopop_step(zE, zI, η̄E, η̄I, ΔE, ΔI, κ, syn, dt;
                             IextE=IextE, IextI=IextI, shunt=shunt)
    end
    return zE, zI
end

# ============================================================================
# Seeding (mirror seed_field_bump_cond for two populations): relax both fields to
# rest → transient localized Gaussian kick → release/settle. Kick amplitudes are
# SEPARATE: science seeds kick E only (AI=0) and lets I follow through W_IE; the
# validation anchor kicks both identically (AE=AI) to hold z_I ≡ z_E.
# ============================================================================
function seed_twopop_bump(x, η̄E, η̄I, ΔE, ΔI, κ, syn;
                          AE=0.6, AI=0.0, σ=0.6, x0=π, T_kick=20.0, T_free=200.0, dt=0.02, shunt::Bool=true)
    zE0 = fill(rest_state(η̄E, ΔE), length(x))
    zI0 = fill(rest_state(η̄I, ΔI), length(x))
    bump  = exp.(-(angular_distance(x, x0).^2) ./ (2σ^2))
    kickE = AE .* bump
    kickI = AI .* bump
    zEk, zIk = evolve_twopop(zE0, zI0, η̄E,η̄I,ΔE,ΔI,κ,syn; T=T_kick, dt=dt, IextE=kickE, IextI=kickI, shunt=shunt)  # carve
    zEf, zIf = evolve_twopop(zEk, zIk, η̄E,η̄I,ΔE,ΔI,κ,syn; T=T_free, dt=dt,                              shunt=shunt)  # settle
    return zEf, zIf
end

# ============================================================================
# Oscillon / breathing detector for the pair — the two-population analogue of
# phase5.jl's breathing_amplitude. Advances (zE, zI) under the supplied pair map
# stepfun2: (zE, zI) -> (zE, zI), and measures the E-population peak-rate oscillation
# over the tail. Returns the SAME NamedTuple shape as breathing_amplitude so the
# generic classify_regime can be reused unchanged. REQUIRES a settled seed.
# ============================================================================
function breathing_amplitude_twopop(zE0, zI0, stepfun2; T=600.0, dt=0.02, tail=0.3)
    nt = round(Int, T/dt)
    i0 = floor(Int, (1 - tail) * nt)
    zE, zI = copy(zE0), copy(zI0)
    peaks     = Float64[]
    contrasts = Float64[]
    for i in 1:nt
        zE, zI = stepfun2(zE, zI)
        if i >= i0
            r = rate.(zE)
            push!(peaks, maximum(r))
            push!(contrasts, maximum(r) - minimum(r))
        end
    end
    pm = sum(peaks) / length(peaks)
    return (osc = (maximum(peaks) - minimum(peaks)) / pm, peak = pm,
            cmin = minimum(contrasts), cmax = maximum(contrasts))
end
