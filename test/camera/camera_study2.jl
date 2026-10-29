# Run from the repository root: julia -t auto --project=. test/camera/camera_study2.jl (about 1.5 hours on 32 threads; the
# output is camera_study2_output.txt). The populations use untuned profile parameters: the study is of the camera.
# Second camera study: a ring band wide enough for the n = 1 light of every region, against a reference that
# resolves it radially (0.0005 in s) and azimuthally (720 cells); candidates for production; weight cutoffs.
using KernelAbstractions, StaticArrays, Printf, Random
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "..", "..", "scripts", "SlowLight.jl")); using .SlowLight
const KA = KernelAbstractions
backend = CPU(); a = 0.94; θo = deg2rad(30.0); L = LUNIT
νs = (NU[1], NU[4], NU[7])
rng = Xoshiro(1)
pops = (disk = disk_population(rng, (-600.0, 600.0), 40; a), funnel = funnel_population(rng, (-600.0, 600.0), 40; a),
        jet = jet_population(rng, (-600.0, 600.0), 40; a))
xs = (disk = [log(4.6e7), log(2.8e11), -0.18, -2.6], funnel = [log(1e7), log(1e11), 0.0, -2.0], jet = [log(1e7), log(2e11), 0.0, -2.0])
band = (0.93, 1.55)
cams = [("reference", 240, 1e-6, (inner = 0.01, ring = (band..., 0.0005), outer = 0.01, arc = 0.03, nψring = 720)),
        ("r.002 ψ720", 160, 1e-6, (inner = 0.03, ring = (band..., 0.002), outer = 0.03, arc = 0.06, nψring = 720)),
        ("r.002 ψ240", 160, 1e-6, (inner = 0.03, ring = (band..., 0.002), outer = 0.03, arc = 0.06, nψring = 240)),
        ("r.004 ψ240", 160, 1e-6, (inner = 0.04, ring = (band..., 0.004), outer = 0.04, arc = 0.1, nψring = 240)),
        ("r.004 ψ120", 160, 1e-6, (inner = 0.04, ring = (band..., 0.004), outer = 0.04, arc = 0.1, nψring = 120)),
        ("r.006 ψ180", 160, 1e-6, (inner = 0.04, ring = (band..., 0.006), outer = 0.04, arc = 0.1, nψring = 180)),
        ("r.004 ψ240 N120", 120, 1e-6, (inner = 0.04, ring = (band..., 0.004), outer = 0.04, arc = 0.1, nψring = 240)),
        ("r.004 ψ240 N240", 240, 1e-6, (inner = 0.04, ring = (band..., 0.004), outer = 0.04, arc = 0.1, nψring = 240)),
        ("r.004 ψ240 c1e-4", 160, 1e-4, (inner = 0.04, ring = (band..., 0.004), outer = 0.04, arc = 0.1, nψring = 240)),
        ("r.004 ψ240 c1e-3", 160, 1e-3, (inner = 0.04, ring = (band..., 0.004), outer = 0.04, arc = 0.1, nψring = 240))]
ts = (0.0, 300.0)
results = Dict{Tuple{String,Symbol,Float64},Array{Float64,3}}()
for (lab, N, cutoff, kw) in cams
    rc = ring_camera(a, θo; ρmax = 80.0, kw...)
    cache, tab, w, τh = camera_setup(backend, rc, a, θo, Val(N))
    out = KA.allocate(backend, OrderState{3,Float64}, npixels(cache))
    for name in keys(pops), t in ts
        P = plasma(pops[name], xs[name])
        Pa, box, _ = frame_parcels(P, t, tab; cutoff)
        el = @elapsed render_orders!(out, cache, Pa, box, τh, t, νs, L; cutoff)
        F = frame_flux(out, w, νs); results[(lab, name, t)] = F
        if lab == "reference"
            s_sorted = rc.s[cache.perm_host]
            I1 = [((st.n >= 2 ? st.s1[2] : st.st[2].S) - (st.n >= 1 ? st.s0[2] : st.st[2].S))[1] * w[j] for (j, st) in enumerate(out)]
            q = sortperm(s_sorted); cum = cumsum(I1[q]) ./ sum(I1)
            qs = [s_sorted[q][findfirst(>=(f), cum)] for f in (0.001, 0.01, 0.99, 0.999)]
            @printf("  %s t=%g: order-1 light of band 6 at s quantiles 0.1%%/1%%/99%%/99.9%%: %.4f %.4f %.4f %.4f; outside the band %.2e\n", name, t, qs...,
                    sum(I1[j] for j in eachindex(I1) if !(band[1] <= s_sorted[j] <= band[2])) / sum(I1))
        end
        @printf("%-17s N %3d %7d rays %-6s t=%5g %6.1f s | band 6: all %.4f  n0 %.4f  n1 %.4f  n≥2 %.4f\n", lab, N, length(rc.area), name, t, el,
                F[1, 1, 2], F[1, 2, 2], F[1, 3, 2] - F[1, 2, 2], F[1, 1, 2] - F[1, 3, 2])
        flush(stdout)
    end
end
println("\nrelative to the reference (Δ/F_all for all and n0, Δ/F_n1 for n1):")
for (lab, N, cutoff, kw) in cams[2:end], name in keys(pops), t in ts
    R = results[("reference", name, t)]; F = results[(lab, name, t)]
    println(rpad(lab, 17), rpad(name, 7), @sprintf("t=%-5g", t), join((@sprintf(" | b%d all %+.1e n0 %+.1e n1 %+.1e (n1 %.3f)", band, F[1, 1, b] / R[1, 1, b] - 1, (F[1, 2, b] - R[1, 2, b]) / R[1, 1, b],
        ((F[1, 3, b] - F[1, 2, b]) - (R[1, 3, b] - R[1, 2, b])) / (R[1, 3, b] - R[1, 2, b]), (R[1, 3, b] - R[1, 2, b]) / R[1, 1, b]) for (b, band) in ((1, 3), (2, 6), (3, 9)))))
end
