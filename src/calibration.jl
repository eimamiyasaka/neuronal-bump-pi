# ============================================================================
# Dimensional-calibration layer — convert the dimensionless theta/QIF readouts to
# physical units (Hz, deg, deg/s, ms) for empirical comparison with HD-cell tuning
# and behavioral angular-velocity data.
#
# The theta/QIF model is non-dimensional: there is exactly ONE free physical scale,
# the membrane time constant τ_m, and fixing it sets every other unit. Two facts:
#
#   (1) time is measured in units of τ_m   →  1 t.u. = τ_m  (we carry τ_m in ms);
#   (2) the ring coordinate x ∈ [0,2π) maps to the full 0–360° heading circle, so
#       ring-radians ARE heading-radians (a full loop = 2π ring-rad = 360° heading).
#
# From those, the conversions below follow. Defaults τ_m = 10 ms (typical cortical /
# HD-cell membrane time constant; pass τ_m=20 for the slow end of the range). NOTHING
# in the model changes — this is a pure unit transformation of EXISTING measurements
# (rate(z), mean_frequencies_*, bump_centroid/lateral_speed, the commanded Ω, drift).
#
# Layer convention (as in moving.jl / pathint.jl / conductance.jl): pure MEASUREMENT
# functions, NO Plots. Presentation + the chosen τ_m live in scripts/run_calibration.jl.
#
# Units cheat-sheet (r = spikes per t.u.; ω = ring-rad per t.u.; t = t.u.):
#   time          t·τ_m                  [ms]            (·/1000 → s)
#   firing rate   r·1000/τ_m             [Hz]
#   heading       x·180/π                [deg]
#   ang. velocity ω·(180/π)·1000/τ_m     [deg/s]         (·60 → deg/min)
# ============================================================================

const RAD_TO_DEG = 180 / π

# --- time -------------------------------------------------------------------
# dimensionless t (t.u.) → physical ms / s, given τ_m in ms. Broadcasts over arrays.
time_ms(t; τ_m=10.0) = t .* τ_m
time_s(t;  τ_m=10.0) = t .* (τ_m / 1000)

# --- firing rate ------------------------------------------------------------
# r in spikes per t.u. (the closed-form rate(z), or the micro mean_frequencies_*,
# both of which return cycles/spikes per t.u.) → Hz. 1 t.u. = τ_m ms, so r per t.u.
# is r/τ_m per ms = 1000·r/τ_m per second.
rate_hz(r; τ_m=10.0) = r .* (1000 / τ_m)

# --- heading angle ----------------------------------------------------------
# ring / heading radians → heading degrees (the map is identity up to 180/π).
heading_deg(x) = x .* RAD_TO_DEG

# --- angular velocity (bump speed s, commanded Ω, drift rate) ---------------
# ω in ring-radians per t.u. → deg/s and deg/min. Same factor as a heading-angle
# rate: convert the radians to degrees (·180/π) and the per-t.u. to per-second
# (·1000/τ_m). Used for s(B), the commanded Ω axis, and bump-drift rates alike.
angvel_degpers(ω;  τ_m=10.0)  = ω .* (RAD_TO_DEG * 1000 / τ_m)
angvel_degpermin(ω; τ_m=10.0) = angvel_degpers(ω; τ_m=τ_m) .* 60

# --- bump width -------------------------------------------------------------
# Full-width-at-half-maximum of a localized ring profile, in heading degrees. Half-max
# is taken above the profile's OWN baseline (its min — at Δ>0 the surround sits at the
# rest-rate floor, not zero, see step4_1_gate.md), and the width is the total angular
# measure of the supra-half region. Wrap-safe: it counts grid points, so a bump that
# straddles the 0/2π seam is measured correctly. `prof` is sampled on evenly-spaced
# points of [0,2π) (firing rate, |z|, …). This is the model-side counterpart of an
# HD-cell tuning-curve width.
function fwhm_deg(prof)
    lo, hi = minimum(prof), maximum(prof)
    half = lo + 0.5 * (hi - lo)
    return count(>=(half), prof) / length(prof) * 360.0
end
