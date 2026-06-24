#!/usr/bin/env python
# ============================================================================
# AUTO-07p driver — homogeneous (uniform) conductance mean field. WARM-UP:
# verifies the toolchain on the project's own equations and maps the BULK
# bifurcations (saddle-node bistability; any bulk Hopf), which are distinct from
# the localized oscillon Hopf (run_bump.py).
#
# (1) Continue the uniform equilibrium in etabar (both directions) -> the S-curve
#     whose folds (LP) are the rest/active bistability underlying the bump.
# (2) From the active branch at operating etabar=-0.4, continue in g0; test for a
#     bulk Hopf (expected ABSENT — the sustained oscillation is the spatial oscillon).
#
# Run from inside auto/ after sourcing env.sh:   python3 run_homog.py
# ============================================================================
import auto, math

def rate(u, w):                       # Laing Eq. 9 firing rate from z=u+iw
    z = complex(u, w)
    return (1.0 / math.pi) * ((1.0 - z) / (1.0 + z)).real

def write_csv(fname, header, rows):
    with open(fname, 'w') as f:
        f.write(header + '\n')
        for row in rows:
            f.write(','.join(repr(v) for v in row) + '\n')

print("=== (1) uniform equilibrium vs etabar (S-curve, both directions) ===")
fwd = auto.run(e='homog', c='homog', UZR={'etabar': -0.4})
bwd = auto.run(e='homog', c='homog', DS='-')
S = auto.merge(fwd + bwd)
auto.save(S, 'homog_eta')
for s in S:
    if s['TY'] in ('LP', 'HB'):
        print(f"  {s['TY']:3s} at etabar = {s['etabar']:.5f}  rate = {rate(s['U(1)'], s['U(2)']):.4f}")
eta = list(S['etabar']); u = list(S['U(1)']); w = list(S['U(2)'])
write_csv('homog_eta.csv', 'etabar,rate', [(e, rate(ui, wi)) for e, ui, wi in zip(eta, u, w)])
print(f"  wrote homog_eta.csv ({len(eta)} pts)")

print("\n=== (2) active branch: continue in g0 at etabar=-0.4, seek bulk Hopf ===")
try:
    start = fwd('UZ1')
    g0run = auto.run(start, ICP=['g0'], ISP=2, ILP=1,
                     DS=0.01, DSMAX=0.02, NMX=2000, RL0=0.0, RL1=1.0)
    auto.save(g0run, 'homog_g0')
    nhb = sum(1 for s in g0run if s['TY'] == 'HB')
    for s in g0run:
        if s['TY'] == 'HB':
            print(f"  bulk HB at g0 = {s['g0']:.5f}")
    if nhb == 0:
        print("  no bulk Hopf on the uniform branch over g0 in [0,1] "
              "(consistent with the oscillon being a SPATIAL instability).")
except Exception as e:
    print(f"  g0 restart skipped/failed: {type(e).__name__}: {e}")
print("=== done ===")
