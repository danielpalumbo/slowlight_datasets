# Movies of a region dataset (scripts/sgra_regions.jl) in one band: images on a uniform grid through the day, split
# by lensing order as the light curves are (all the light, the light that reached the screen directly, and the light
# that went half an orbit about the hole), and the grid's light curves by order. The model is redrawn exactly as
# sgra_regions.jl draws it (same tuning, same seed). The grid's flux densities are a check on the dataset's ring
# camera: they differ from it by the light outside the field and the grid's coarser sampling of the n = 1 ring.
#
#     julia -t auto --project=. scripts/movie_regions.jl --model disk|funnel|jet [--band 6] [--res 240] [--fov <M>]
#           [--dt 2] [--tmax <a day>] [--cubes movies] [--out datasets/sgra_<model>_dominated] [--resume] [--cpu]
#
# Writes under --cubes (untracked; a gigabyte or so a model) <dataset>_band<b>_{all,n0,n1}.npy: float32 NumPy arrays of
# shape (frames, res, res) in Jy per pixel, indexed [frame, β, α] (β up, α right: imshow with origin = "lower"); and
# in the dataset's directory movie_lightcurves_by_order_band<b>_<ν>GHz.csv (the grid's flux densities by order,
# columns as the dataset's lightcurves_by_order files) and movie_metadata.toml. scripts/draw_movie.jl makes the mp4.
using CUDA, KernelAbstractions, StaticArrays, Statistics, Printf, Dates, TOML, Serialization, Mmap
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "SlowLight.jl")); using .SlowLight
include(joinpath(@__DIR__, "mirror.jl")); using .Mirror
const KA = KernelAbstractions
const Tr = KerrSplat.Transfer

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
backend = ("--cpu" in ARGS || !CUDA.functional()) ? CPU() : CUDABackend()
model = Symbol(getstr("--model", "disk")); model in REGIONS || error("--model must be one of $(REGIONS)")
const MOVIE_FOV = (disk = 40.0, funnel = 32.0, jet = 60.0)     # M: 98.5–100% of the light, pixels of 0.13–0.25 M
b = findfirst(==(getopt("--band", 6)), BANDS); ν = NU[b]
res = getopt("--res", 240); fov = getopt("--fov", MOVIE_FOV[model]); dt = getopt("--dt", 2.0); tmax = getopt("--tmax", TMAX)
const MODEL_SEED = (disk = 101, funnel = 202, jet = 303)       # as scripts/sgra_regions.jl
name = "sgra_$(model)_dominated"
datadir = joinpath(@__DIR__, "..", "datasets", name)
outdir = getstr("--out", datadir); mkpath(outdir)
cubedir = getstr("--cubes", joinpath(@__DIR__, "..", "movies")); mkpath(cubedir)
checkpoint = joinpath(@__DIR__, "..", "scratch", "$(name)_movie_band$(BANDS[b])_checkpoint.jls"); mkpath(dirname(checkpoint))

a = SPIN; θo = deg2rad(INCLINATION)
tuning = TOML.parsefile(joinpath(@__DIR__, "..", "datasets", "sgra_regions_tuning.toml"))
x = NamedTuple{REGIONS}(Tuple(Float64.(tuning[string(r)]["x"]) for r in REGIONS))
nactive = NamedTuple{REGIONS}(Tuple((r == model ? 0.8 : 0.1) * NACTIVE[r] for r in REGIONS))
P, region = model_parcels(nactive, x, (0.0, TMAX); a, seed = MODEL_SEED[model])
@info "model" model parcels = size(P, 2) band = BANDS[b] res fov dt; flush(stderr)

# ---- the camera: a uniform grid (its pixels include both edges, so they are fov/(res − 1) apart)
cam = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
sc = GeodesicCache(CPU(), cam, Val(160); store_samples = true); regenerate!(sc, a, θo; marcher = Recurrence(64))
tab = LookbackTable(sc; margin = 5.0); sc = nothing; GC.gc()
cache = GeodesicCache(backend, cam, Val(160); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
τh = half_orbit_times(cache)
out = KA.allocate(backend, OrderState{1,Float64}, npixels(cache))
dev(x) = (y = KA.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
Δ = fov / (res - 1)
Ωpix = (Δ * LUNIT / D_CM)^2
w = fill(Ωpix, npixels(cache))
jy = νhat(ν)^3 * Ωpix / Tr.JY                                  # Jy per pixel of a ray's (invariant) Stokes I

"""
    npy_cube(path, nx, ny, nt) -> Array{Float32,3}

A float32 NumPy file of C-order shape (nt, ny, nx), memory-mapped as the Julia array (nx, ny, nt); created (zeros)
when absent, its header checked when present.
"""
function npy_cube(path, nx, ny, nt)
    dict = "{'descr': '<f4', 'fortran_order': False, 'shape': ($nt, $ny, $nx), }"
    hdr = dict * " "^mod(-(11 + length(dict)), 64) * "\n"           # the data start on a 64-byte boundary
    head = take!(let io = IOBuffer(); write(io, 0x93, "NUMPY", 0x01, 0x00, htol(UInt16(length(hdr))), hdr); io; end)
    fresh = !isfile(path)
    io = open(path, fresh ? "w+" : "r+")
    fresh ? write(io, head) : (read(io, length(head)) == head || error("$path is not a cube of this shape"))
    return Mmap.mmap(io, Array{Float32,3}, (nx, ny, nt), length(head); grow = fresh)
end

times = collect(0.0:dt:tmax); nt = length(times)
parts = ("all", "n0", "n1")
cubes = NamedTuple{Symbol.(parts)}(Tuple(npy_cube(joinpath(cubedir, "$(name)_band$(BANDS[b])_$(p).npy"), res, res, nt) for p in parts))
lc = fill(NaN, 4, 3, nt)
start = 1
if "--resume" in ARGS && isfile(checkpoint)
    ck = deserialize(checkpoint)
    (ck.times == times && ck.res == res && ck.fov == fov && ck.ν == ν) || error("the checkpoint is of another run")
    lc .= ck.lc; start = ck.next
    @info "resuming" frame = start; flush(stderr)
end

# ---- the frames, with a checkpoint every 50
t_start = time()
for k in start:nt
    t = times[k]
    Pa, box, _ = frame_parcels(P, t, tab; cutoff = CUTOFF)
    render_orders!(out, cache, dev(Pa), dev(box), τh, t, (ν,), LUNIT; cutoff = CUTOFF)
    acc = Array(out)
    lc[:, :, k] = frame_flux(acc, w, (ν,))[:, :, 1]
    I = [s.st[1].S[1] for s in acc]
    I0 = [s.n >= 1 ? s.s0[1][1] : s.st[1].S[1] for s in acc]
    I1 = [s.n >= 2 ? s.s1[1][1] : s.st[1].S[1] for s in acc] .- I0
    for (cube, v) in zip(cubes, (I, I0, I1))
        cube[:, :, k] = reshape(Float32.(jy .* Geodesics.unsort(cache, v)), res, res)
    end
    if k % 50 == 0 || k == nt
        foreach(Mmap.sync!, cubes)
        serialize(checkpoint * ".tmp", (; times, res, fov, ν, lc, next = k + 1)); mv(checkpoint * ".tmp", checkpoint; force = true)
        el = time() - t_start; done = k - start + 1
        @info "movie frames" frame = k of = nt seconds_per_frame = round(el / done, digits = 2) hours_left = round(el / done * (nt - k) / 3600, digits = 2); flush(stderr)
    end
end

# ---- the grid's light curves by order, and how they compare with the dataset's (ring camera) at the same times
lcfile = "movie_" * band_file("lightcurves_by_order", b)
open(joinpath(outdir, lcfile), "w") do io
    println(io, "band,nu_GHz,t_M,t_s,", join(("$(s)_n$(o)_Jy" for o in ("0", "1", "2plus") for s in ("I", "Q", "U", "V")), ","))
    for (k, t) in enumerate(times)
        n0 = lc[:, 2, k]; n1 = lc[:, 3, k] .- lc[:, 2, k]; n2 = lc[:, 1, k] .- lc[:, 3, k]
        @printf(io, "%d,%.1f,%.3f,%.3f,%s\n", BANDS[b], ν / 1e9, t, t * TUNIT, join((@sprintf("%.6e", v) for v in vcat(n0, n1, n2)), ","))
    end
end
ref = Dict(parse(Float64, f[3]) => parse.(Float64, f[5:end]) for f in split.(readlines(joinpath(datadir, band_file("lightcurves_by_order", b)))[2:end], ","))
ratio(col, part) = [part(k) / ref[t][col] for (k, t) in enumerate(times) if haskey(ref, t)]
r0 = ratio(1, k -> lc[1, 2, k]); r1 = ratio(5, k -> lc[1, 3, k] - lc[1, 2, k])
stats(r) = Dict("mean" => mean(r), "min" => minimum(r), "max" => maximum(r))
@info "grid over ring camera, Stokes I" order0 = round.((mean(r0), extrema(r0)...), digits = 4) order1 = round.((mean(r1), extrema(r1)...), digits = 4); flush(stderr)
meta = Dict(
    "created" => string(now()), "model" => string(model), "spin" => a, "inclination_deg" => INCLINATION, "band" => BANDS[b], "nu_GHz" => ν / 1e9, "frames" => nt, "t0_M" => first(times), "tmax_M" => last(times),
    "dt_M" => dt, "dt_s" => dt * TUNIT, "res" => res, "fov_M" => fov, "pixel_M" => Δ, "pixel_uas" => Δ * LUNIT / D_CM * 180 / π * 3600e6,
    "samples_per_ray" => 160, "cutoff" => CUTOFF,
    "cubes" => Dict("files" => ["$(name)_band$(BANDS[b])_$(p).npy" for p in parts], "parts" => ["all the light", "order 0 (direct)", "order 1 (one half-orbit)"],
                    "units" => "Jy per pixel", "dtype" => "float32", "shape" => [nt, res, res], "index" => "[frame, beta, alpha], beta up and alpha right (origin lower)",
                    "alpha_beta_range_M" => [-fov / 2, fov / 2], "where" => "not in git (about $(round(Int, 3 * 4 * res^2 * nt / 1e6)) MB); regenerate with scripts/movie_regions.jl"),
    "lightcurves" => lcfile, "mp4" => "movie_band$(BANDS[b])_$(round(Int, ν / 1e9))GHz.mp4",
    "grid_over_ring_camera_I" => Dict("order0" => stats(r0), "order1" => stats(r1)),
)
open(io -> TOML.print(io, meta; sorted = true), joinpath(outdir, "movie_metadata.toml"), "w")
@info "written" outdir lcfile cubes = cubedir; flush(stderr)
mirror_dataset(joinpath(outdir, lcfile)); mirror_dataset(joinpath(outdir, "movie_metadata.toml"))
