# ============================================================================
# Shared plotting helpers for the Phase-1 comparison figures (run_field.jl,
# run_ring.jl). Presentation only — no model/measurement logic lives here.
# Kept out of src/field.jl, which deliberately carries no Plots dependency.
# ============================================================================
using Plots

# π-scaled axis ticks: positions at every π/2 in [0, xmax], labelled in units of
# π ("0", "π/2", "π", "3π/2", "2π", "5π/2", ...). Used for the position x-axes and
# for the arg-z y-axis (which runs 0..2π). Pass e.g. pi_ticks(2π) or pi_ticks(2.5π).
function pi_ticks(xmax)
    labels = ["0", "π/2", "π", "3π/2", "2π", "5π/2", "3π"]
    n = floor(Int, xmax / (π/2) + 1e-9)           # number of π/2 steps that fit
    pos = [k * (π/2) for k in 0:n]
    return (pos, labels[1:length(pos)])
end

# Periodic right-extension of a position-indexed series by `pad` radians: the ring
# is 2π-periodic, so we copy the leading `pad` of data onto [2π, 2π+pad). This lets
# the spiking panels run slightly past 2π and show the (possibly drifted) bump whole
# across the 0/2π seam. `x` is the [0,2π) grid; works for any real vector `v`.
function wrap_extend(x, v, pad)
    k = round(Int, length(x) * pad / (2π))        # grid points spanning `pad`
    x_ext = vcat(x, x[1:k] .+ 2π)
    v_ext = vcat(v, v[1:k])
    return x_ext, v_ext
end

# Remap an angle into [0, 2π) (arg_laing returns (−π, π]; the paper plots 0..2π).
to_02pi(a) = mod(a, 2π)

# Insert NaN where consecutive samples jump by more than π, so a phase curve that
# wraps across the 0↔2π seam is broken instead of drawn as a vertical streak. For
# smooth curves (the field arg); the ring arg is intentionally left un-broken since
# it is genuine noise where |z|≈0. Returns a copy of `v` with breaks inserted.
function break_wraps(v; thresh=π)
    out = copy(float.(v))
    for i in 2:length(out)
        if abs(out[i] - out[i-1]) > thresh
            out[i-1] = NaN
        end
    end
    return out
end
