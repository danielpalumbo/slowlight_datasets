# Checks of scripts/SlowLight.jl against KerrSplat's own consumer:
#   1. moving splats at vr = 0 are KerrSplat's polarized splats to the bit (RadiativeTransport, per band);
#   2. the multi-band consumer reproduces RadiativeTransport over moving splats band by band (radially moving ones too);
#   3. its order-0 and order-≤1 parts of a thin disk equal KerrSplat's rays truncated with nmax = 0 and 1 (slab 0.5 M);
#   4. the frame boxes are conservative (stochastic populations of all three regions against unculled rendering);
#   5. τ½ is the Mino time between successive polar turning points of the marched rays.
#     julia -t auto --project=. test/test_slowlight.jl
using KernelAbstractions, StaticArrays, Printf, Test, Random
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "..", "scripts", "SlowLight.jl")); using .SlowLight
const KA = KernelAbstractions
backend = CPU()
a = 0.94; θo = deg2rad(30.0); L = LUNIT
rows = split.(readlines(joinpath(@__DIR__, "..", "datasets", "test_sgra_disk20", "splats.csv"))[2:end], ",")
p21 = reduce(hcat, [parse.(Float64, r) for r in rows])
cam = Geodesics.Camera((-20.0, 20.0), (-20.0, 20.0), 64)
cache = GeodesicCache(backend, cam, Val(160); store_samples = false)
regenerate!(cache, a, θo; marcher = Fused(64))
buf = KA.allocate(backend, accumulator_type(Float64, -1), npixels(cache))
τh = half_orbit_times(cache)
single(model, ν) = (fill!(buf, zero(eltype(buf))); fused_march!(RadiativeTransport(model, ν, L), buf, cache); map(st -> observed_stokes(st, ν), buf))
wide = LookbackTable([1.0, 1e4], [-1e4, -1e4], [1e4, 1e4])     # every parcel, every time
relmax(x, y) = maximum(norm.(x .- y)) / maximum(norm.(y))
using LinearAlgebra: norm
t = 37.0
@testset "moving splats at rest are KerrSplat's" begin
    pm = at_rest(p21)
    for ν in NU
        @test single(MovingSplats(pm, t), ν) == single(PolarizedSplats(p21, t), ν)
    end
end
@testset "coefficients with the Bessel functions passed in = KerrSplat's" begin
    rng0 = Xoshiro(3); worst = 0
    for _ in 1:20000
        ne, Θe, B = exp(10 + 6 * rand(rng0)), exp(-1 + 6 * rand(rng0)), exp(5 * rand(rng0)); ν = exp(log(5e10) + 3 * rand(rng0)); θ = π * rand(rng0)
        ref = thermal_synchrotron(ne, Θe, B, ν, θ)
        got = thermal_synchrotron_k(ne, Θe, B, ν, θ, KerrSplat.Transfer.besselk012(inv(Θe))...)
        worst += ref != got
    end
    @test worst == 0
    @test thermal_synchrotron_k(1e6, 30.0, 20.0, 2e11, 0.0, KerrSplat.Transfer.besselk012(1 / 30)...) == thermal_synchrotron(1e6, 30.0, 20.0, 2e11, 0.0)
end
# a moving set: half the disk's parcels lifted 2 M off the plane, born at t = 30 M and moving outward after it
pm = at_rest(p21)
for i in 1:2:size(pm, 2)
    pm[22, i] = 0.3; pm[3, i] = 2.0; pm[12, i] = log(5.0); pm[23, i] = log(15.0); pm[11, i] = 30.0
end
out = KA.allocate(backend, OrderState{NB,Float64}, npixels(cache))
render_orders!(out, cache, frame_parcels(pm, t, wide)[1:2]..., τh, t, NU, L)
@testset "multi-band consumer = RadiativeTransport, band by band" begin
    for (b, ν) in enumerate(NU)
        ref = single(MovingSplats(pm, t), ν)
        got = [st.st[b].S * νhat(ν)^3 for st in out]
        @test relmax(got, ref) < 1e-12
        @printf("band %d: max |Δ|/max = %.2e\n", BANDS[b], relmax(got, ref))
    end
end
# a thin disk (vertical scale 0.05 M): orders against KerrSplat's slab truncation
pt = copy(p21); pt[3, :] .= 0.0; pt[6, :] .= log(0.05)
ptm = at_rest(pt)
render_orders!(out, cache, frame_parcels(ptm, t, wide)[1:2]..., τh, t, NU, L)
wb = KA.allocate(backend, accumulator_type(Float64, 0), npixels(cache))
@testset "orders = KerrSplat's nmax truncation for a thin disk" begin
    for (b, ν) in ((1, NU[1]), (4, NU[4]), (7, NU[7]))
        for nmax in (0, 1)
            fill!(wb, zero(eltype(wb)))
            polarized_image!(wb, cache, pt, t, ν, L; nmax, slab = 0.5)
            ref = [observed_stokes(w, ν) for w in wb]
            got = [(nmax == 0 ? (st.n >= 1 ? st.s0[b] : st.st[b].S) : (st.n >= 2 ? st.s1[b] : st.st[b].S)) * νhat(ν)^3 for st in out]
            @printf("band %d, nmax %d: max |Δ|/max = %.2e, flux ratio %.12f\n", BANDS[b], nmax, relmax(got, ref), sum(g[1] for g in got) / sum(r[1] for r in ref))
            @test relmax(got, ref) < 1e-10
        end
        full = [st.st[b].S * νhat(ν)^3 for st in out]
        @printf("band %d: order 0 %.4f, order 1 %.4f, orders ≥ 2 %.4f of the flux\n", BANDS[b],
                sum((st.n >= 1 ? st.s0[b] : st.st[b].S)[1] for st in out) * νhat(ν)^3 / sum(f[1] for f in full),
                sum(((st.n >= 2 ? st.s1[b] : st.st[b].S) - (st.n >= 1 ? st.s0[b] : st.st[b].S))[1] for st in out) * νhat(ν)^3 / sum(f[1] for f in full),
                sum((st.st[b].S - (st.n >= 2 ? st.s1[b] : st.st[b].S))[1] for st in out) * νhat(ν)^3 / sum(f[1] for f in full))
    end
end
# speed: seven bands in one pass against seven single-band passes
Pa, box = frame_parcels(pm, t, wide)
render_orders!(out, cache, Pa, box, τh, t, NU, L)
t1 = @elapsed render_orders!(out, cache, Pa, box, τh, t, NU, L)
t7 = @elapsed for ν in NU; single(MovingSplats(pm, t), ν); end
@printf("seven bands: one multi-band pass %.3f s, seven single passes %.3f s\n", t1, t7)
# the boxes of frame_parcels are conservative: stochastic populations of all three regions, tight lookback window
using Random
rng = Xoshiro(7)
P = hcat(plasma(disk_population(rng, (-300.0, 300.0), 30; a), [log(4e7), log(2.8e11), -0.18, -2.6]),
         plasma(funnel_population(rng, (-300.0, 300.0), 30; a), [log(3e6), log(2e11), 0.0, -2.0]),
         plasma(jet_population(rng, (-300.0, 300.0), 30; a), [log(3e7), log(1e11), 0.0, -2.0]))
scache = GeodesicCache(backend, cam, Val(160); store_samples = true)
regenerate!(scache, a, θo; marcher = Recurrence(64))
tab = LookbackTable(scache)
@printf("lookback range within 10 M: %.1f to %.1f M; within 100 M: %.1f to %.1f M\n", tab(10.0)..., tab(100.0)...)
@testset "frame boxes are conservative" begin
    for tt in (-40.0, 0.0, 55.0)
        Pa, box, cols = frame_parcels(P, tt, tab)
        render_orders!(out, cache, Pa, box, τh, tt, NU, L)
        for b in (1, 4, 7)
            ref = single(MovingSplats(P, tt), NU[b])
            got = [st.st[b].S * νhat(NU[b])^3 for st in out]
            @printf("t = %5.1f: %d of %d parcels in the frame; band %d max |Δ|/max = %.2e\n", tt, length(cols), size(P, 2), BANDS[b], relmax(got, ref))
            @test relmax(got, ref) < 1e-12
        end
    end
end

# τ½ against the marched rays: successive sign changes of the polar momentum are τ½ apart (to the sample spacing)
@testset "half-orbit Mino time" begin
    S = Geodesics.host(scache.samples); pc = Geodesics.host(scache.consts)
    τhs = half_orbit_times(scache)
    worst = 0.0; nrays = 0
    for j in axes(S.t, 1)
        Δτ = pc.τ_total[j] / (160 + 1)
        flips = [k for k in 2:160 if S[j, k].ok && S[j, k - 1].ok && S[j, k].νθ != S[j, k - 1].νθ]
        length(flips) >= 2 || continue
        nrays += 1
        worst = max(worst, maximum(abs((flips[q + 1] - flips[q]) * Δτ - τhs[j]) / Δτ for q in 1:length(flips)-1))
    end
    @printf("τ½ against turning points on %d rays: worst |Δ| = %.2f sample spacings\n", nrays, worst)
    @test nrays > 100
    @test worst <= 1.0 + 1e-9
end
