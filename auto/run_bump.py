#!/usr/bin/env python
# ============================================================================
# AUTO-07p driver — oscillon (gamma) Hopf of the next-gen field bump.
#
# (1) 1-parameter equilibrium continuation of the static bump in the shunting
#     gain g0 (Delta=0.1 fixed), detecting the Hopf bifurcation g0* where the
#     static bump becomes a breathing oscillon.
# (2) 2-parameter continuation of that Hopf locus in the (g0, Delta) plane.
#
# Run from inside auto/ after sourcing auto/env.sh:   python3 run_bump.py
# Outputs: bump_eq.* / bump_hopf_*.* (AUTO solution files), bump_branch.csv
# (g0, L2norm), bump_hopf.csv (g0, Delta).
# ============================================================================
import auto

def write_csv(fname, header, rows):
    with open(fname, 'w') as f:
        f.write(header + '\n')
        for row in rows:
            f.write(','.join(repr(v) for v in row) + '\n')

print("=== (1) equilibrium continuation of the bump in g0 (Delta=0.1) ===")
eq = auto.run(e='bump', c='bump')
auto.save(eq, 'bump_eq')

hopf_g0 = []
for s in eq:
    ty = s['TY']
    if ty in ('HB', 'LP'):
        print(f"  {ty:3s} at g0 = {s['g0']:.5f}")
        if ty == 'HB':
            hopf_g0.append(s['g0'])

g0 = list(eq['g0']); l2 = list(eq['L2-NORM'])
write_csv('bump_branch.csv', 'g0,L2norm', zip(g0, l2))
print(f"  wrote bump_branch.csv ({len(g0)} pts);  g0 in [{min(g0):.3f},{max(g0):.3f}]")

hb = eq('HB')
if len(hb) == 0:
    print("  NO Hopf detected on the branch — inspect bump_eq.")
    raise SystemExit
g0star = min(h['g0'] for h in hb)
print(f"\n  >>> oscillon Hopf at g0* = {g0star:.5f}   (simulation: ~0.33)\n")
print("  (the (g0,Delta) Hopf locus is traced by run_hopf_locus.py)")
print("=== done ===")
