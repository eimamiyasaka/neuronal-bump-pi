"""Digitize Laing & Omel'chenko Fig. 3 (notes/image3.png) and compare it quantitatively
to our pseudo-arclength continuation (figures/continuation_bs.csv, written by run_sweep.jl).
Validation utility for the Step-2.3 gate (notes/writeupAssist/step2_3_gate.md); run from the
project root: `python scripts/digitize_fig3.py`. Auxiliary analysis only — not part of the
Julia simulation pipeline.

Calibration (from frame detection): plot box cols [78,450] = B in [0,0.2],
rows [27,194] = s in [1.3,0]. We extract the SOLID BLACK curve pixels (red/blue markers
are excluded by the black mask), map them to (B,s), and for each point on OUR curve
measure the distance to the nearest Laing curve pixel — i.e. does our curve lie on the
published one. An overlay figure is saved for visual confirmation."""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- calibration ---
xL, xR, yT, yB = 78, 450, 27, 194
Bmax, smax = 0.2, 1.3
col2B = lambda c: (c - xL) / (xR - xL) * Bmax
row2s = lambda r: (yB - r) / (yB - yT) * smax

im = np.asarray(Image.open("notes/image3.png").convert("RGB")).astype(int)
R, G, Bb = im[:, :, 0], im[:, :, 1], im[:, :, 2]
# black (not coloured): all channels low AND low saturation
mx = np.maximum(np.maximum(R, G), Bb)
mn = np.minimum(np.minimum(R, G), Bb)
black = (mx < 90) & ((mx - mn) < 45)
# interior only, with a 3px margin off the frame to drop spines/ticks
interior = np.zeros_like(black)
interior[yT + 3:yB - 2, xL + 3:xR - 2] = True
black &= interior

ys, xs = np.where(black)
Bd = col2B(xs); sd = row2s(ys)
print(f"digitized {len(Bd)} black-curve pixels")
print(f"  B range [{Bd.min():.3f},{Bd.max():.3f}]  s range [{sd.min():.3f},{sd.max():.3f}]")

# --- our continuation ---
rows = np.genfromtxt("figures/continuation_bs.csv", delimiter=",", names=True)
Bc, sc = rows["B"], rows["s"]
ok = np.isfinite(Bc) & np.isfinite(sc)
Bc, sc = Bc[ok], sc[ok]
inwin = (Bc >= 0) & (Bc <= Bmax) & (sc >= 0) & (sc <= smax)
Bc, sc = Bc[inwin], sc[inwin]
print(f"our continuation: {len(Bc)} in-window points  s in [{sc.min():.3f},{sc.max():.3f}]")

# --- distance: each of OUR points -> nearest Laing curve pixel (in (B,s) data units) ---
# (robust to arrows / to Laing arcs we don't trace, since we query from our curve)
P = np.column_stack([Bd, sd])
d = np.empty(len(Bc))
for i, (b, s) in enumerate(zip(Bc, sc)):
    d[i] = np.sqrt(((P - [b, s]) ** 2).sum(axis=1)).min()
print("\nour-curve -> Laing-curve distance (data units, (B,s) plane):")
print(f"  RMS  = {np.sqrt((d**2).mean()):.4f}")
print(f"  mean = {d.mean():.4f}   median = {np.median(d):.4f}   max = {d.max():.4f}")
print(f"  (pixel scale: 1 px ~ {Bmax/(xR-xL):.4f} in B, {smax/(yB-yT):.4f} in s)")

# --- feature checks vs Laing's one tabulated point (knot ~ (0.1,0.85)) ---
print(f"\nLaing's only tabulated s(B) value: knot ~ (0.10, 0.85)")
print(f"  our knot (upper-branch leftmost) was (0.087, 0.868)")

# Laing's OWN small-B slope, read off the digitized lower arc (0.01 < B <= 0.04).
m = (Bd > 0.01) & (Bd <= 0.04)
slope_laing = (Bd[m] * sd[m]).sum() / (Bd[m] ** 2).sum()   # least-squares through origin
print(f"\nLaing digitized small-B slope (0.01<B<=0.04, {m.sum()} px): {slope_laing:.3f}"
      f"   (ours 3.91, classical guide 3.33)")

# Laing's OWN knot = leftmost point of the digitized upper structure (s > 0.78).
up = sd > 0.78
iu = np.argmin(Bd[up])
print(f"Laing digitized knot (upper-structure leftmost): "
      f"({Bd[up][iu]:.3f}, {sd[up][iu]:.3f})   (ours (0.087, 0.868))")

# --- overlay figure ---
plt.figure(figsize=(7, 4))
plt.scatter(Bd, sd, s=4, c="0.6", label="Laing Fig. 3 (digitized black curve)")
plt.plot(Bc, sc, "r.", ms=2, label="our continuation")
plt.xlabel("B"); plt.ylabel("s"); plt.xlim(0, Bmax); plt.ylim(0, smax)
plt.legend(loc="upper left", fontsize=8); plt.title("continuation vs digitized Laing Fig. 3")
plt.tight_layout(); plt.savefig("figures/laing_overlay.png", dpi=130)
print("\nsaved figures/laing_overlay.png")
