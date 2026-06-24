# AUTO-07p continuation pipeline — Phase-5 publication-grade Hopf boundary

Locates, by numerical continuation (not brute-force simulation), the **gamma /
oscillon Hopf** at which the static next-gen field bump becomes a breathing bump,
along the shunting axis `g0`, and traces that Hopf in the `(g0, Δ)` plane. This is
the AUTO/XPPAUT deliverable deferred in `notes/writeupAssist/step5_gate.md`.

## What is continued

The exact OA field under conductance/Dale synapses (`src/conductance.jl`), reduced
to the **reflection-symmetric half grid** so the translation (Goldstone) zero mode
is excluded and the corrector is non-singular — the only oscillatory instability
detected is then the even **breathing (oscillon)** mode. Under matched reversals
`E_E,E_I=(κ/g0,−κ/g0)`, `g0` cancels from the drive and appears only in the shunt,
so it is a clean continuation parameter. Full derivation: header of
`scripts/auto_export_bump.jl`.

The Fortran RHS (`bump.f90`) is a **direct port of the Julia field code**, validated
to `1.3e-15` before any continuation was trusted (the export script prints the check).

## Files

| file | role |
|------|------|
| `bump.f90`, `c.bump` | NDIM=258 field model (even half grid, N=256) + AUTO constants |
| `homog.f90`, `c.homog` | NDIM=2 spatially-uniform mean field — toolchain warm-up + bulk baseline |
| `bump_init.dat`, `homog_init.dat` | starting equilibria, written by the Julia exporters |
| `run_bump.py` | 1-param Hopf in `g0`, then 2-param `(g0,Δ)` Hopf locus |
| `run_homog.py` | uniform S-curve vs `etabar`; bulk-Hopf search vs `g0` |
| `env.sh`, `build_auto07p.sh` | local auto-07p build + environment |

## Reproduce

```bash
# 0. prerequisites (WSL/Ubuntu): one-time
sudo apt install -y gfortran make
bash auto/build_auto07p.sh                 # builds ~/auto-07p (no sudo)

# 1. starting solutions (Windows or WSL Julia)
julia --project=. scripts/auto_export_bump.jl     # -> auto/bump_init.dat  (+ validation)
julia --project=. scripts/auto_export_homog.jl    # -> auto/homog_init.dat

# 2. continuation
cd auto && source env.sh
python3 run_bump.py        # oscillon Hopf g0* + (g0,Δ) locus  -> bump_*.csv
python3 run_homog.py       # warm-up / bulk baseline           -> homog_*.csv

# 3. figures (Windows Julia)
julia --project=. scripts/run_auto_figures.jl
```

## Result

Continuation Hopf: **g0\* = 0.3303** at Δ=0.1, confirming the simulation value
`g0*≈0.33` (`run_regime_map.jl`). See `notes/writeupAssist/step5_auto_gate.md`.

## Note on auto-07p

This auto-07p build has a py2-ism (`import AUTOExceptions`) that surfaces only on
invalid data-column keys; the drivers use valid keys (`g0`, `Delta`, `L2-NORM`,
`TY`). Plotting modules are disabled (no Tkinter) — figures are made in Julia.
