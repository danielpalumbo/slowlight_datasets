# A 20-splat accretion disk for the inner 20 M of Sgr A*, tuned to its mean spectrum on ALMA bands 3–9, and its slow-light
# light curves in full Stokes at the seven bands.
#
# The disk: twenty one-zone thermal parcels (KerrSplat's 21-parameter polarized splats) on a golden-angle spiral whose
# centres run log-uniformly in radius from 2 M to 16 M, each a Gaussian of radial, azimuthal and vertical scales
# 0.25 r, 0.5 r and 0.3 r elongated along its orbit, carrying power-law profiles in the manner of the semi-analytic RIAF
# of Broderick & Loeb (2006) at its centre (peak density n₀ r^sn, electron temperature T₀ r^sT, a field in
# equipartition with the ions at plasma β, B²/8π = n mₚc²/(12 β r), pitched 45° between toroidal and vertical), moving
# on prograde Keplerian circular orbits (the pattern rate Ω = 1/(r^{3/2} + a) and the matching ZAMO velocity, so the
# pattern and the fluid move together). The four profile parameters (n₀, T₀, sT, sn) are tuned by Levenberg–Marquardt
# on the logarithm of the total flux density at t = 0 against the target spectrum below, with β held at 3 (Bower et al.
# 2019: β ≃ 1–10 with the X-ray limit; left free, β runs off along the spectrum's degeneracy between a dense, weakly
# magnetized plasma and a tenuous one with kilogauss fields). The light curves are then rendered with slow light (every
# sample of every ray sees the plasma at its own emission time t_obs − t̃), so the orbits of the inner parcels appear
# as they were when the light left them.
#
#     julia -t auto --project=. scripts/test_sgra_disk20.jl [--cpu] [--res 128] [--fov 48] [--nsamples 200]
#           [--tmax 500] [--dt 2] [--spin 0.94] [--inclination 30] [--beta 3] [--no-tune] [--out datasets/test_sgra_disk20]
#
# The CUDA backend is used when a functional GPU is present (`--cpu` forces the CPU backend of KernelAbstractions).
# Writes, under --out: lightcurves_band<b>_<ν>GHz.csv for each band (band, frequency, time, Stokes I, Q, U, V in Jy),
# spectrum.csv (target, the light curves' mean, standard deviation and range, the tuning snapshot), splats.csv (the 20
# parameter columns), metadata.toml (the source, spacetime, camera, tuned normalizations and KerrSplat commit) and
# overview.png.
using CUDA, KernelAbstractions, StaticArrays, LinearAlgebra, Statistics, Printf, Dates, TOML
using CairoMakie
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
const Tr = KerrSplat.Transfer
include(joinpath(@__DIR__, "mirror.jl")); using .Mirror

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
backend = ("--cpu" in ARGS || !CUDA.functional()) ? CPU() : CUDABackend()
res = getopt("--res", 128); fov = getopt("--fov", 48.0); nsamples = getopt("--nsamples", 200)
tmax = getopt("--tmax", 500.0); dt = getopt("--dt", 2.0)
a = getopt("--spin", 0.94); incl = getopt("--inclination", 30.0); θo = deg2rad(incl)
β = getopt("--beta", 3.0)
tune = !("--no-tune" in ARGS)
outdir = getstr("--out", joinpath(@__DIR__, "..", "datasets", "test_sgra_disk20")); mkpath(outdir)

# ---- the source: GRAVITY Collaboration (2019), M = 4.152×10⁶ M☉ at D = 8.178 kpc
M_solar = 4.152e6; D = 8.178e3 * Tr.PC
L = gravitational_radius(M_solar)                         # GM/c² [cm]
tg = L / Tr.CL                                            # GM/c³ [s]

# ---- the target spectrum on ALMA's standard continuum frequencies (Technical Handbook) for bands 3–9. Anchors, all
# from 2017: 2.0 ± 0.2 Jy at 98 GHz and 2.4 ± 0.2 Jy at 213–229 GHz during the EHT campaign (EHT Collaboration 2022,
# Paper II, Table 2), and the simultaneous 233 → 678 GHz index α = −0.26 ± 0.02 of Bower et al. (2019). Below 233 GHz
# a power law through the two EHT points (α = +0.21); above it Bower's index. The fractional uncertainties weight the
# tuning: 10% through band 7, more at the high bands, where the flux scale and Sgr A*'s mean are less certain.
const BANDS = [3, 4, 5, 6, 7, 8, 9]
const NU = [97.5, 145.0, 203.0, 233.0, 343.5, 405.0, 679.0] .* 1e9
target_flux(ν) = ν <= 233e9 ? 2.4 * (ν / 233e9)^(log(2.4 / 2.0) / log(233 / 97.5)) : 2.4 * (ν / 233e9)^-0.26
const FTARGET = target_flux.(NU)
const FRACERR = [0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.20]

# ---- the model
"ZAMO-frame γβ of a prograde Keplerian circular orbit at Boyer–Lindquist radius `r` in the equatorial plane of spin `a`."
function keplerian_γβ(r, a)
    Σ = r^2; Δ = r^2 - 2r + a^2; A = (r^2 + a^2)^2 - a^2 * Δ
    v = sqrt(A / Σ) * (1 / (r^1.5 + a) - 2a * r / A) / sqrt(Σ * Δ / A)      # ϖ (Ω − ω) / α
    return v / sqrt(1 - v^2)
end

const ΘUNIT = Tr.KBOL / (Tr.ME * Tr.CL^2)                # Θe per kelvin

"""
    disk_splats(n0, T0, β; a, sn = -1.1, sT = -0.84, n = 20, rin = 2.0, rout = 16.0, shape = (0.25, 0.5, 0.3), pitch = π/4)

The `NPOLARIZEDPARAMS × n` parameter matrix of the disk: centres on a golden-angle spiral at radii log-uniform from
`rin` to `rout`, scales `shape .* r` (radial, azimuthal, vertical) with the long axis along the orbit, the profiles at
each centre (peak density `n0` r^`sn` [cm⁻³], temperature `T0` r^`sT` [K], field from plasma β; the defaults are
Broderick & Loeb's RIAF slopes), the field `pitch` from the vertical toward the orbit, Keplerian orbits, and temporal
envelopes wider than any movie.
"""
function disk_splats(n0, T0, β; a, sn = -1.1, sT = -0.84, n = 20, rin = 2.0, rout = 16.0, shape = (0.25, 0.5, 0.3), pitch = π / 4)
    p = zeros(NPOLARIZEDPARAMS, n)
    golden = π * (3 - sqrt(5))
    for i in 1:n
        r = rin * (rout / rin)^((i - 1) / (n - 1)); φ = i * golden
        ne = n0 * r^sn
        Θe = T0 * ΘUNIT * r^sT
        B = sqrt(8π * ne * Tr.MP * Tr.CL^2 / (β * 12 * r))
        p[:, i] = [r * cos(φ), r * sin(φ), 0.0,                            # centre at t₀
                   log(shape[1] * r), log(shape[2] * r), log(shape[3] * r),  # scales along r̂, φ̂, ẑ
                   cos(φ / 2), 0.0, 0.0, sin(φ / 2),                         # a rotation by φ about the spin axis
                   0.0, log(1e9),                                            # t₀, ln of the temporal envelope
                   log(ne), log(Θe), log(B), pitch, π / 2,                   # B in the (φ̂, −θ̂) plane
                   0.0, keplerian_γβ(r, a), 0.0, 1 / (r^1.5 + a)]            # ZAMO γβ (r̂, φ̂, −θ̂), pattern rate
    end
    return p
end

# ---- the camera and the renderer
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(backend, camera, Val(nsamples); store_samples = false)
regenerate!(cache, a, θo; marcher = Fused(64))
Δα = fov / (res - 1)                                       # the grid includes both edges
jy_per_pixel = pixel_solid_angle(Δα, L, D) / Tr.JY
buf = KernelAbstractions.allocate(backend, accumulator_type(Float64, -1), npixels(cache))
dev(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)

"Total (I, Q, U, V) [Jy] of the parameters `pd` (on the backend) at observation time `t` [M] and frequency `ν` [Hz]."
function total_stokes(pd, t, ν)
    fill!(buf, zero(eltype(buf)))
    polarized_image!(buf, cache, pd, t, ν, L)
    return sum(Array(map(st -> observed_stokes(st, ν), buf))) .* jy_per_pixel
end
spectrum(p, t) = (pd = dev(p); [total_stokes(pd, t, ν)[1] for ν in NU])

# ---- tuning: Levenberg–Marquardt over x = (ln n₀, ln T₀, sT, sn) on the log flux densities at t = 0
# the tuning of datasets/test_sgra_disk20 (128² pixels, 2026-10-28): --no-tune renders exactly that model
x_start = [log(4.586630597227289e7), log(2.8353939846708246e11), -0.17769164968184994, -2.5954431290743853]
model(x) = disk_splats(exp(x[1]), exp(x[2]), β; a, sT = x[3], sn = x[4])
function residuals(x)
    F = spectrum(model(x), 0.0)
    return (log.(F) .- log.(FTARGET)) ./ FRACERR, F
end
function levenberg_marquardt(x; iters = 12, h = 0.02, tol = 1e-4)
    r, F = residuals(x); χ2 = sum(abs2, r); λ = 1e-2
    @info "tuning" iteration = 0 χ2 n0 = exp(x[1]) T0 = exp(x[2]) sT = x[3] sn = x[4] flux = round.(F, digits = 3)
    for it in 1:iters
        J = reduce(hcat, [(residuals(x .+ h .* (eachindex(x) .== k))[1] .- r) ./ h for k in eachindex(x)])
        A = J' * J; g = J' * r
        accepted = false
        while λ < 1e6
            δ = -((A + λ * Diagonal(diag(A))) \ g)
            r2, F2 = residuals(x .+ δ); χ2new = sum(abs2, r2)
            if χ2new < χ2
                done = (χ2 - χ2new) < tol * χ2 || norm(δ) < 1e-3
                x = x .+ δ; r = r2; F = F2; χ2 = χ2new; λ = max(λ / 3, 1e-6); accepted = true
                @info "tuning" iteration = it χ2 n0 = exp(x[1]) T0 = exp(x[2]) sT = x[3] sn = x[4] flux = round.(F, digits = 3)
                done && return x, F, χ2
                break
            end
            λ *= 4
        end
        accepted || return x, F, χ2
    end
    return x, F, χ2
end
t_start = time()
if tune
    x, F0, χ2 = levenberg_marquardt(x_start)
else
    x = x_start; r0, F0 = residuals(x); χ2 = sum(abs2, r0)
end
n0, T0, sT, sn = exp(x[1]), exp(x[2]), x[3], x[4]
p = model(x)
@info "tuned" n0 T0 sT sn β χ2 seconds = round(time() - t_start, digits = 1)
# the share of the flux in the outer 4 M of the field at the lowest and highest bands: a check that the field holds the source
αs = range(-fov / 2, fov / 2; length = res)
rim = [max(abs(α), abs(β′)) > fov / 2 - 4 for α in αs, β′ in αs]
rim_fraction = map((NU[1], NU[end])) do ν
    I = [s[1] for s in Array(polarized_image(cache, dev(p), 0.0, ν, L))]
    sum(I[rim]) / sum(I)
end
@info "flux in the outer 4 M of the field" band3 = rim_fraction[1] band9 = rim_fraction[2]

# ---- the light curves
times = collect(0.0:dt:tmax)
lc = zeros(4, length(times), length(NU))                   # (Stokes, time, band) [Jy]
pd = dev(p)
t_lc = time()
for (k, t) in enumerate(times)
    for (b, ν) in enumerate(NU)
        lc[:, k, b] = total_stokes(pd, t, ν)
    end
    k % 20 == 0 && @info "light curves" frame = k of = length(times) seconds = round(time() - t_lc, digits = 1)
end
@info "light curves done" frames = length(times) * length(NU) seconds = round(time() - t_lc, digits = 1)

# ---- outputs
for (b, ν) in enumerate(NU)
    open(joinpath(outdir, @sprintf("lightcurves_band%d_%gGHz.csv", BANDS[b], ν / 1e9)), "w") do io
        println(io, "band,nu_GHz,t_M,t_s,I_Jy,Q_Jy,U_Jy,V_Jy")
        for (k, t) in enumerate(times)
            @printf(io, "%d,%.1f,%.3f,%.3f,%.6e,%.6e,%.6e,%.6e\n", BANDS[b], ν / 1e9, t, t * tg, lc[1, k, b], lc[2, k, b], lc[3, k, b], lc[4, k, b])
        end
    end
end
Imean = vec(mean(lc[1, :, :]; dims = 1)); Istd = vec(std(lc[1, :, :]; dims = 1))
Imin = vec(minimum(lc[1, :, :]; dims = 1)); Imax = vec(maximum(lc[1, :, :]; dims = 1))
open(joinpath(outdir, "spectrum.csv"), "w") do io
    println(io, "band,nu_GHz,target_Jy,target_frac_err,mean_Jy,std_Jy,min_Jy,max_Jy,tuning_snapshot_Jy")
    for b in eachindex(NU)
        @printf(io, "%d,%.1f,%.4f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f\n", BANDS[b], NU[b] / 1e9, FTARGET[b], FRACERR[b], Imean[b], Istd[b], Imin[b], Imax[b], F0[b])
    end
end
open(joinpath(outdir, "splats.csv"), "w") do io
    println(io, join(string.(POLARIZED_SPLAT_PARAMS), ","))
    for i in axes(p, 2)
        println(io, join((@sprintf("%.10g", v) for v in p[:, i]), ","))
    end
end
manifest = TOML.parsefile(joinpath(@__DIR__, "..", "Manifest.toml"))
ks = only(manifest["deps"]["KerrSplat"])
meta = Dict(
    "created" => string(now()), "julia" => string(VERSION), "backend" => string(nameof(typeof(backend))),
    "kerrsplat" => Dict("repo" => get(ks, "repo-url", get(ks, "path", "")), "rev" => get(ks, "repo-rev", "")),
    "source" => Dict("name" => "Sgr A*", "mass_Msun" => M_solar, "distance_kpc" => D / Tr.PC / 1e3, "GM_over_c2_cm" => L, "GM_over_c3_s" => tg),
    "spacetime" => Dict("spin" => a, "inclination_deg" => incl),
    "camera" => Dict("fov_M" => fov, "pixels" => res, "pixel_M" => Δα, "samples_per_ray" => nsamples,
                     "rim_flux_fraction_band3" => rim_fraction[1], "rim_flux_fraction_band9" => rim_fraction[2]),
    "light_curves" => Dict("t0_M" => first(times), "tmax_M" => last(times), "dt_M" => dt, "frames" => length(times)),
    "model" => Dict("splats" => size(p, 2), "n0_cm3" => n0, "T0_K" => T0, "density_slope" => sn, "temperature_slope" => sT, "beta" => β,
                    "rin_M" => 2.0, "rout_M" => 16.0, "shape_over_r" => [0.25, 0.5, 0.3], "field_pitch_from_vertical_deg" => 45.0,
                    "tuned" => tune, "tuning_chi2" => χ2),
    "bands" => Dict("band" => BANDS, "nu_GHz" => NU ./ 1e9, "target_Jy" => FTARGET, "target_frac_err" => FRACERR),
)
open(io -> TOML.print(io, meta; sorted = true), joinpath(outdir, "metadata.toml"), "w")

# ---- the overview figure: the disk from above, the 233 GHz image, the spectrum, and the light curves band by band
ink = colorant"#0b0b0b"; ink2 = colorant"#52514e"; muted = colorant"#898781"; grid = colorant"#e1e0d9"
c_model = colorant"#2a78d6"; c_target = colorant"#eb6834"
set_theme!(Theme(fontsize = 13, backgroundcolor = colorant"#fcfcfb", textcolor = ink,
                 Axis = (backgroundcolor = colorant"#fcfcfb", xgridcolor = grid, ygridcolor = grid, xminorgridvisible = false,
                         leftspinecolor = colorant"#c3c2b7", bottomspinecolor = colorant"#c3c2b7", topspinevisible = false,
                         rightspinevisible = false, xtickcolor = muted, ytickcolor = muted, xticklabelcolor = ink2, yticklabelcolor = ink2,
                         titlefont = :regular, titlealign = :left, titlesize = 13)))
fig = Figure(size = (1500, 1150))
left = fig[1, 1] = GridLayout(); right = fig[1, 2] = GridLayout()
# the disk from above at t = 0: each parcel's 1σ ellipse
ax = Axis(left[1, 1], aspect = DataAspect(), title = "20 parcels at t = 0, seen from above (1σ)", xlabel = "x [M]", ylabel = "y [M]")
for i in axes(p, 2)
    cx, cy = p[1, i], p[2, i]; φ = atan(cy, cx); sr, sφ = exp(p[4, i]), exp(p[5, i])
    ell = [Point2f(cx + sr * cos(s) * cos(φ) - sφ * sin(s) * sin(φ), cy + sr * cos(s) * sin(φ) + sφ * sin(s) * cos(φ)) for s in range(0, 2π; length = 64)]
    poly!(ax, ell, color = (c_model, 0.18), strokecolor = c_model, strokewidth = 1)
end
rh = 1 + sqrt(1 - a^2)
poly!(ax, Circle(Point2f(0), Float32(rh)), color = ink)
lines!(ax, [Point2f(20cos(s), 20sin(s)) for s in range(0, 2π; length = 200)], color = muted, linestyle = :dash, linewidth = 1)
text!(ax, 14.5, -16.5, text = "r = 20 M", color = ink2, fontsize = 11)
limits!(ax, -24, 24, -24, 24)
# the 233 GHz image at t = 0 as brightness temperature
img = Array(polarized_image(cache, pd, 0.0, NU[4], L))
Tb = [s[1] * Tr.CL^2 / (2 * Tr.KBOL * NU[4]^2) / 1e10 for s in img]
ax = Axis(left[2, 1], aspect = DataAspect(), title = @sprintf("band 6 (233 GHz) at t = 0, i = %g°", incl), xlabel = "α [M]", ylabel = "β [M]")
hm = heatmap!(ax, αs, αs, Tb, colormap = :afmhot, colorrange = (0, quantile(vec(Tb), 0.999)))
Colorbar(left[2, 2], hm, label = "Tb [10¹⁰ K]", width = 10)
limits!(ax, -18, 18, -18, 18)
# the spectrum
ax = Axis(left[3, 1:2], xscale = log10, yscale = log10, title = "spectrum on ALMA bands 3–9", xlabel = "ν [GHz]", ylabel = "Fν [Jy]",
          xticks = [100, 200, 300, 500, 700], yticks = [1.5, 2, 2.5, 3])
band!(ax, NU ./ 1e9, Imin, Imax, color = (c_model, 0.15))
lines!(ax, NU ./ 1e9, Imean, color = c_model, linewidth = 2)
scatter!(ax, NU ./ 1e9, Imean, color = c_model, markersize = 9, label = "model: light-curve mean (band: range)")
errorbars!(ax, NU ./ 1e9, FTARGET, FTARGET .* FRACERR, color = c_target, whiskerwidth = 6)
scatter!(ax, NU ./ 1e9, FTARGET, color = c_target, marker = :diamond, markersize = 10, label = "target (EHT 2017; Bower et al. 2019)")
ylims!(ax, 1.3, 3.2)
axislegend(ax, position = :lb, framevisible = false, labelsize = 11)
rowsize!(left, 3, Relative(0.3))
# the light curves, one band per row, each over its own mean on a common scale so that the amplitudes compare
axes_lc = map(eachindex(NU)) do b
    axb = Axis(right[b, 1], xlabel = b == length(NU) ? @sprintf("observer time t [M]   (1 M = GM/c³ = %.1f s)", tg) : "",
               xticklabelsvisible = b == length(NU))
    hlines!(axb, [1.0], color = muted, linewidth = 1)
    lines!(axb, times, lc[1, :, b] ./ Imean[b], color = c_model, linewidth = 1.5)
    text!(axb, 0.01, 0.95, text = @sprintf("band %d · %.1f GHz   mean %.2f Jy, rms %.1f%%", BANDS[b], NU[b] / 1e9, Imean[b], 100 * Istd[b] / Imean[b]),
          space = :relative, align = (:left, :top), color = ink2, fontsize = 11)
    xlims!(axb, first(times), last(times))
    axb
end
linkyaxes!(axes_lc...)
Label(right[1:length(NU), 0], "Fν / ⟨Fν⟩", rotation = π / 2, fontsize = 13)
Label(right[0, 1], "slow-light Stokes I light curves, each over its mean", halign = :left, font = :regular, fontsize = 13, tellwidth = false)
rowgap!(right, 6)
colsize!(fig.layout, 1, Relative(0.33))
save(joinpath(outdir, "overview.png"), fig, px_per_unit = 2)
@info "written" outdir total_seconds = round(time() - t_start, digits = 1)
mirror_dataset(outdir)
