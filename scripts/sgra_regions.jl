# A day of slow-light light curves of Sgr A* from a model in which one region of the spacetime gives most of the light:
# the equatorial disk, the funnel (the magnetosphere above and below the hole), or the jet. Each model holds parcels
# of all three regions (scripts/SlowLight.jl): its dominant region keeps 0.8 NACTIVE of its parcels alive on average and
# the other two 0.1 NACTIVE each, every region with the profiles tuned by scripts/tune_regions.jl so that, on
# average, the three give 80%, 10% and 10% of the target spectrum on ALMA bands 3–9. The light curves are the total
# Stokes flux densities with slow light, and the same split by lensing order: the light that reaches the observer
# directly (order 0), after one half-orbit about the hole (order 1, the n = 1 ring and its kin), and after more.
#
#     julia -t auto --project=. scripts/sgra_regions.jl --model disk|funnel|jet [--dt 1] [--tmax <a day>] [--cpu]
#           [--out datasets/sgra_<model>_dominated] [--resume] [--tuning datasets/sgra_regions_tuning.toml] [--share-snapshots 16]
#
# Writes, under --out, for each band: lightcurves_band<b>_<ν>GHz.csv (band, frequency, time, Stokes I, Q, U, V in Jy)
# and lightcurves_by_order_band<b>_<ν>GHz.csv (the same per lensing order); then spectrum.csv, parcels.csv (every
# parcel's region and parameters) and metadata.toml; scripts/plot_regions.jl draws overview.png from them. A
# checkpoint of the frames done is kept in scratch/ (--resume continues from it).
using CUDA, KernelAbstractions, StaticArrays, LinearAlgebra, Statistics, Printf, Dates, TOML, Random, Serialization
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "SlowLight.jl")); using .SlowLight
include(joinpath(@__DIR__, "mirror.jl")); using .Mirror
const KA = KernelAbstractions
const Tr = KerrSplat.Transfer

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
backend = ("--cpu" in ARGS || !CUDA.functional()) ? CPU() : CUDABackend()
model = Symbol(getstr("--model", "disk")); model in REGIONS || error("--model must be one of $(REGIONS)")
dt = getopt("--dt", 1.0); tmax = getopt("--tmax", TMAX)
outdir = getstr("--out", joinpath(@__DIR__, "..", "datasets", "sgra_$(model)_dominated")); mkpath(outdir)
checkpoint = joinpath(@__DIR__, "..", "scratch", basename(normpath(outdir)) * "_checkpoint.jls"); mkpath(dirname(checkpoint))
const MODEL_SEED = (disk = 101, funnel = 202, jet = 303)

a = SPIN; θo = deg2rad(INCLINATION)
tuning = TOML.parsefile(getstr("--tuning", joinpath(@__DIR__, "..", "datasets", "sgra_regions_tuning.toml")))
x = NamedTuple{REGIONS}(Tuple(Float64.(tuning[string(r)]["x"]) for r in REGIONS))
nactive = NamedTuple{REGIONS}(Tuple((r == model ? 0.8 : 0.1) * NACTIVE[r] for r in REGIONS))
P, region = model_parcels(nactive, x, (0.0, tmax); a, seed = MODEL_SEED[model])
@info "model" model nactive parcels = [count(==(r), region) for r in REGIONS]; flush(stderr)

rc = ring_camera(a, θo; PRODUCTION_CAMERA...)
cache, tab, w, τh = camera_setup(backend, rc, a, θo, Val(PRODUCTION_CAMERA_SAMPLES))
out = KA.allocate(backend, OrderState{NB,Float64}, npixels(cache))
dev(x) = (y = KA.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
"Stokes flux densities (4, part, band) of the columns `cols` of P at time t (part: all, order 0, orders ≤ 1)."
function frame(t, cols = axes(P, 2))
    Pa, box, _ = frame_parcels(P[:, cols], t, tab; cutoff = CUTOFF)
    render_orders!(out, cache, dev(Pa), dev(box), τh, t, NU, LUNIT; cutoff = CUTOFF)
    return frame_flux(out, w, NU)
end
@info "camera" rays = length(rc.area) samples = PRODUCTION_CAMERA_SAMPLES; flush(stderr)

# ---- the light curves, with a checkpoint every 50 frames
times = collect(0.0:dt:tmax)
lc = fill(NaN, 4, 3, NB, length(times))
start = 1
if "--resume" in ARGS && isfile(checkpoint)
    ck = deserialize(checkpoint)
    (ck.times == times && ck.P == P) || error("the checkpoint is of another run")
    lc .= ck.lc; start = ck.next
    @info "resuming" frame = start; flush(stderr)
end
t_lc = time()
for k in start:length(times)
    lc[:, :, :, k] = frame(times[k])
    if k % 50 == 0 || k == length(times)
        serialize(checkpoint * ".tmp", (; times, P, lc, next = k + 1)); mv(checkpoint * ".tmp", checkpoint; force = true)
        el = time() - t_lc; done = k - start + 1
        @info "light curves" frame = k of = length(times) seconds_per_frame = round(el / done, digits = 2) hours_left = round(el / done * (length(times) - k) / 3600, digits = 2); flush(stderr)
    end
end

# ---- how much of the light each region gives: each region alone at snapshots through the day
nshare = getopt("--share-snapshots", 16)
snap = [(k - 0.5) * tmax / nshare for k in 1:nshare]
share = zeros(length(REGIONS), NB)
for t in snap
    tot = frame(t)[1, 1, :]
    for (q, r) in enumerate(REGIONS)
        share[q, :] .+= frame(t, findall(==(r), region))[1, 1, :] ./ tot ./ length(snap)
    end
end
@info "region shares (band 3, 6, 9)" share[:, [1, 4, 7]]; flush(stderr)

# ---- outputs
I = lc[1, 1, :, :]                                          # (band, time)
order1 = lc[1, 3, :, :] .- lc[1, 2, :, :]
for b in 1:NB
    open(joinpath(outdir, band_file("lightcurves", b)), "w") do io
        println(io, "band,nu_GHz,t_M,t_s,I_Jy,Q_Jy,U_Jy,V_Jy")
        for (k, t) in enumerate(times)
            @printf(io, "%d,%.1f,%.3f,%.3f,%.6e,%.6e,%.6e,%.6e\n", BANDS[b], NU[b] / 1e9, t, t * TUNIT, lc[1, 1, b, k], lc[2, 1, b, k], lc[3, 1, b, k], lc[4, 1, b, k])
        end
    end
    open(joinpath(outdir, band_file("lightcurves_by_order", b)), "w") do io
        println(io, "band,nu_GHz,t_M,t_s,", join(("$(s)_n$(o)_Jy" for o in ("0", "1", "2plus") for s in ("I", "Q", "U", "V")), ","))
        for (k, t) in enumerate(times)
            n0 = lc[:, 2, b, k]; n1 = lc[:, 3, b, k] .- lc[:, 2, b, k]; n2 = lc[:, 1, b, k] .- lc[:, 3, b, k]
            @printf(io, "%d,%.1f,%.3f,%.3f,%s\n", BANDS[b], NU[b] / 1e9, t, t * TUNIT, join((@sprintf("%.6e", v) for v in vcat(n0, n1, n2)), ","))
        end
    end
end
Imean = vec(mean(I; dims = 2)); Istd = vec(std(I; dims = 2))
open(joinpath(outdir, "spectrum.csv"), "w") do io
    println(io, "band,nu_GHz,target_Jy,target_frac_err,mean_Jy,std_Jy,min_Jy,max_Jy,order1_fraction,orders2plus_fraction,", join(("$(r)_fraction" for r in REGIONS), ","))
    for b in 1:NB
        @printf(io, "%d,%.1f,%.4f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n", BANDS[b], NU[b] / 1e9, FTARGET[b], FRACERR[b], Imean[b], Istd[b], minimum(I[b, :]), maximum(I[b, :]),
                sum(order1[b, :]) / sum(I[b, :]), sum(lc[1, 1, b, :] .- lc[1, 3, b, :]) / sum(I[b, :]), join((@sprintf("%.4f", share[q, b]) for q in eachindex(REGIONS)), ","))
    end
end
open(joinpath(outdir, "parcels.csv"), "w") do io
    println(io, "region,", join(string.(MOVING_SPLAT_PARAMS), ","))
    for i in axes(P, 2)
        println(io, region[i], ",", join((@sprintf("%.10g", v) for v in P[:, i]), ","))
    end
end
manifest = TOML.parsefile(joinpath(@__DIR__, "..", "Manifest.toml"))
ks = only(manifest["deps"]["KerrSplat"])
meta = Dict(
    "created" => string(now()), "julia" => string(VERSION), "backend" => string(nameof(typeof(backend))),
    "kerrsplat" => Dict("repo" => get(ks, "repo-url", ""), "rev" => get(ks, "repo-rev", "")),
    "source" => Dict("name" => "Sgr A*", "mass_Msun" => M_SOLAR, "distance_kpc" => D_CM / Tr.PC / 1e3, "GM_over_c2_cm" => LUNIT, "GM_over_c3_s" => TUNIT),
    "spacetime" => Dict("spin" => a, "inclination_deg" => INCLINATION),
    "camera" => Dict("kind" => "annuli about the critical curve (scripts/SlowLight.jl ring_camera)", "rays" => length(rc.area),
                     "samples_per_ray" => PRODUCTION_CAMERA_SAMPLES, "critical_curve_centre_alpha_M" => rc.αc,
                     (string(k) => (v isa Tuple ? collect(v) : v) for (k, v) in pairs(PRODUCTION_CAMERA))...),
    "light_curves" => Dict("t0_M" => first(times), "tmax_M" => last(times), "dt_M" => dt, "frames" => length(times), "hours" => last(times) * TUNIT / 3600),
    "model" => Dict("dominant_region" => string(model), "seed" => MODEL_SEED[model], "parcels" => size(P, 2),
                    (string(r) => Dict("nactive" => nactive[r], "parcels" => count(==(r), region), "x" => x[r], "n0_cm3" => exp(x[r][1]), "T0_K" => exp(x[r][2]),
                                       "temperature_slope" => x[r][3], "density_slope" => x[r][4], "share_band3" => share[q, 1], "share_band6" => share[q, 4],
                                       "share_band9" => share[q, 7]) for (q, r) in enumerate(REGIONS))...),
    "bands" => Dict("band" => collect(BANDS), "nu_GHz" => collect(NU) ./ 1e9, "target_Jy" => collect(FTARGET), "target_frac_err" => collect(FRACERR)),
)
open(io -> TOML.print(io, meta; sorted = true), joinpath(outdir, "metadata.toml"), "w")
@info "written" outdir; flush(stderr)
mirror_dataset(outdir)
