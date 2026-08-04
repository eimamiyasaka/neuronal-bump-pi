# neuron-sim — a ring attractor for angular path integration in next-generation neural fields

Julia code for a head-direction ring attractor built from **theta (QIF) neurons**, studied in the
"next-generation" neural-field framework of Laing & Omel'chenko (2020, *Moving bumps in theta neuron
networks*). Every result is carried by a **micro/macro pair** — a spiking network and its exact
Ott–Antonsen mean-field reduction — that must agree.

**Headline result.** The conductance **shunting gain `g0`** is a single axis that unifies three things
usually treated separately: the path-integration gain, network synchrony `|z|`, and the distance to a
Hopf bifurcation at `g0* = 0.33026` (AUTO continuation) where the static bump becomes a breathing
**oscillon** (gamma regime). Raising `g0` lowers PI gain (`ds/dB|₀` 3.91 → 2.40), lowers synchrony
(0.895 → 0.848), and raises the per-realization drift scale (0.182 → 0.217) — one knob, several
currencies. The bump still path-integrates *above* the Hopf (moving oscillon, decoded gain 0.988).

## Models

| File | What it is |
|---|---|
| `src/single.jl` | single theta neuron: `θ̇ = (1-cosθ) + (1+cosθ)(η + κI)` |
| `src/ring.jl` | **microscopic** spiking ring: N theta neurons, pulse coupling, Lorentzian `η`, RK4 |
| `src/field.jl` | **macroscopic** exact OA reduction: `ż = (i/2)(z-1)² − ½(z+1)²[Δ + i(η̄+κI)]` |
| `src/moving.jl` | moving-bump measurement (centroid → lateral speed) and `B`-sweeps |
| `src/continuation.jl` | Newton + pseudo-arclength continuation of the travelling-wave equation |
| `src/pathint.jl` | velocity-driven path integration: `B(t) = β·Ω(t)`, heading readouts |
| `src/conductance.jl` | conductance/Dale synapses (`g·(E−v)`), the shunting gain `g0` |
| `src/phase5.jl` | oscillon / regime / drift readouts |
| `src/drift_adjoint.jl` | Nakao adjoint phase-sensitivity → the finite-N drift law |
| `src/twopop.jl` | exploratory two-population E–I divisive shunting (field only) |
| `src/calibration.jl` | dimensionless → Hz / deg / deg·s⁻¹ via `τ_m` |

Kernel is first-harmonic Mexican-hat, `K = 0.1 + 0.3cos(x−y) + B·sin(x−y)`; `B` is the velocity
asymmetry knob. Static-bump operating point: `η̄ = −0.4, κ = 2, Δ = 0.01, n = 2`.

## Running

Julia 1.12+, deps pinned in `Project.toml` / `Manifest.toml` (FFTW, Plots). Always activate the project:

```bash
julia --project=. scripts/run_compare.jl          # micro/macro static-bump agreement gate
julia --project=. scripts/run_sweep.jl            # s(B) speed diagram + continuation (Laing Fig. 3)
julia --project=. scripts/run_pathint.jl          # bump dead-reckons ∫Ω dt
julia --project=. scripts/run_conductance_gain.jl # PI gain under conductance synapses
julia --project=. scripts/run_shunting_sweep.jl   # headline: synchrony, gain, band vs g0
julia --project=. scripts/run_oscillon_pi.jl      # path integration across the gamma Hopf
```

There is **no test suite**: each script self-validates with embedded sanity checks (printed ranges,
finiteness, bump metrics) and writes figures + CSVs to `figures/`. First run precompiles FFTW/Plots
(a few minutes); iterate by `include`ing scripts in a warm REPL. `display()` is a no-op headless —
inspect the saved PNGs.

## Bifurcation continuation (`auto/`)

The oscillon Hopf is obtained by continuation, not only simulation. `auto/bump.f90` is a bit-faithful
NDIM=258 port of the field right-hand side (reflection-symmetric half grid, which removes the
translation Goldstone mode) run under a locally built **auto-07p** on WSL. See `auto/README.md`.

```bash
julia --project=. scripts/auto_export_bump.jl
cd auto && source env.sh && python3 run_bump.py && python3 run_hopf_locus.py
```

## Status

Phases 1–5 are gated (micro/macro agreement, travelling waves, path integration, conductance/Dale
synapses, the shunting contribution). Known limitations, owned rather than hidden: the bump is ~198°
wide where head-direction tuning is ~90° (needs E–I, see `scripts/run_bump_width.jl`); synapses are
instantaneous; the two-population E–I extension is field-only and not yet gated against a spiking
network.
