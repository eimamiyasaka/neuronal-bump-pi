#!/usr/bin/env python
# ============================================================================
# AUTO-07p driver — (g0, Delta) oscillon-Hopf BOUNDARY.
#
# For each Delta, start from an INDEPENDENTLY re-seeded, validated static bump
# (scripts/auto_export_bumps_locus.jl -> bump_init_D###.dat) and run the trusted
# 1-parameter g0 continuation, recording the first Hopf g0*(Delta). Re-seeding per
# Delta (rather than continuing in Delta) avoids branch contamination from the
# bump fold at high Delta. STPNT always reads 'bump_init.dat', so we copy each
# per-Delta file into place before its run.
#
# Output: bump_hopf.csv (g0, Delta).  Run from auto/ after sourcing env.sh.
# ============================================================================
import auto, shutil

def write_csv(fname, header, rows):
    with open(fname, 'w') as f:
        f.write(header + '\n')
        for row in rows:
            f.write(','.join(repr(v) for v in row) + '\n')

targets = []
with open('locus_targets.txt') as f:
    for line in f:
        d, fname = line.split()
        targets.append((float(d), fname))

print("=== (g0,Delta) Hopf locus: 1-param g0->HB from a re-seeded bump per Delta ===")
boundary = []
for D, fname in targets:
    shutil.copyfile(fname, 'bump_init.dat')          # STPNT reads bump_init.dat
    r = auto.run(e='bump', c='bump', ISP=2, ILP=1, MXBF=0, ISW=1,
                 DS=0.01, DSMIN=1e-6, DSMAX=0.02, NMX=900, RL0=0.0, RL1=0.45)
    hbs = r('HB')
    if len(hbs) == 0:
        print(f"  Delta={D:.3f}: no Hopf in g0<=0.45")
        continue
    g0star = min(h['g0'] for h in hbs)
    boundary.append((g0star, D))
    print(f"  Delta={D:.3f}  ->  g0* = {g0star:.5f}")

boundary.sort(key=lambda p: p[1])
write_csv('bump_hopf.csv', 'g0,Delta', boundary)
print(f"wrote bump_hopf.csv ({len(boundary)} pts)")
# restore the Delta=0.1 starting file for run_bump.py reproducibility
shutil.copyfile('bump_init_D100.dat', 'bump_init.dat')
print("=== done ===")
