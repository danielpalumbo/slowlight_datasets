# Run from the repository root: julia -t auto --project=. test/camera/camera_study3.jl (about 10 minutes; the output is
# camera_study3_output.txt). It compares with camera_study2.jl's reference, whose band-6 numbers it carries.
# Third camera study: a ring band graded toward the critical curve (n ≥ 2) against study 2's reference (same seeded
# populations), and the seven-band cost of each at the production cutoff.
using KernelAbstractions, StaticArrays, Printf, Random
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "..", "..", "scripts", "SlowLight.jl")); using .SlowLight
const KA = KernelAbstractions
backend = CPU(); a = 0.94; θo = deg2rad(30.0); L = LUNIT
rng = Xoshiro(1)
pops = (disk = disk_population(rng, (-600.0, 600.0), 40; a), funnel = funnel_population(rng, (-600.0, 600.0), 40; a),
        jet = jet_population(rng, (-600.0, 600.0), 40; a))
xs = (disk = [log(4.6e7), log(2.8e11), -0.18, -2.6], funnel = [log(1e7), log(1e11), 0.0, -2.0], jet = [log(1e7), log(2e11), 0.0, -2.0])
# study 2's reference at band 6 (all, n0, n1, n≥2)
ref = Dict((:disk, 0.0) => (0.8761, 0.8213, 0.0498, 0.0050), (:disk, 300.0) => (0.8110, 0.7638, 0.0426, 0.0046),
           (:funnel, 0.0) => (2.3689, 1.7889, 0.5620, 0.0180), (:funnel, 300.0) => (2.2898, 1.6849, 0.5928, 0.0121),
           (:jet, 0.0) => (1.4439, 1.4274, 0.0155, 0.0010), (:jet, 300.0) => (0.6371, 0.6272, 0.0092, 0.0007))
band = (0.93, 1.55)
cams = [("graded .0005", (inner = 0.04, ring = (band..., 0.006), core = (0.0005, 0.1), outer = 0.04, arc = 0.1, nψring = 120)),
        ("graded .001", (inner = 0.04, ring = (band..., 0.006), core = (0.001, 0.1), outer = 0.04, arc = 0.1, nψring = 120)),
        ("uniform .006", (inner = 0.04, ring = (band..., 0.006), outer = 0.04, arc = 0.1, nψring = 120))]
for (lab, kw) in cams
    rc = ring_camera(a, θo; ρmax = 80.0, kw...)
    cache, tab, w, τh = camera_setup(backend, rc, a, θo, Val(120))
    out3 = KA.allocate(backend, OrderState{3,Float64}, npixels(cache)); out7 = KA.allocate(backend, OrderState{NB,Float64}, npixels(cache))
    for name in keys(pops), t in (0.0, 300.0)
        P = plasma(pops[name], xs[name])
        Pa, box, _ = frame_parcels(P, t, tab)
        render_orders!(out3, cache, Pa, box, τh, t, (NU[1], NU[4], NU[7]), L)
        F = frame_flux(out3, w, (NU[1], NU[4], NU[7]))
        all_, n0, n1, n2 = F[1, 1, 2], F[1, 2, 2], F[1, 3, 2] - F[1, 2, 2], F[1, 1, 2] - F[1, 3, 2]
        R = ref[(name, t)]
        Pa4, box4, _ = frame_parcels(P, t, tab; cutoff = 1e-4)
        el = @elapsed render_orders!(out7, cache, Pa4, box4, τh, t, NU, L; cutoff = 1e-4)
        @printf("%-13s %6d rays %-6s t=%5g | band 6 vs reference: all %+.1e n0 %+.1e n1 %+.1e n≥2 %.4f (ref %.4f) | 7 bands at 1e-4: %.1f s\n",
                lab, length(rc.area), name, t, all_ / R[1] - 1, (n0 - R[2]) / R[1], (n1 - R[3]) / R[3], n2, R[4], el)
        flush(stdout)
    end
end
