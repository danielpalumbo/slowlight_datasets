# The overview figure of a region dataset (scripts/sgra_regions.jl): a band-6 image of the model, its spectrum against
# the target, the share of the light each lensing order carries per band, the day's light curves band by band, and
# three hours of band 6 (mid-day) split into all the light and the light that has gone half an orbit about the hole.
#
#     julia -t auto --project=. scripts/plot_regions.jl datasets/sgra_<model>_dominated [--cpu]
using CUDA, KernelAbstractions, StaticArrays, Statistics, Printf, TOML
using CairoMakie
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "SlowLight.jl")); using .SlowLight
include(joinpath(@__DIR__, "mirror.jl")); using .Mirror
const KA = KernelAbstractions
const Tr = KerrSplat.Transfer

dir = ARGS[1]
backend = ("--cpu" in ARGS || !CUDA.functional()) ? CPU() : CUDABackend()
meta = TOML.parsefile(joinpath(dir, "metadata.toml"))
model = meta["model"]["dominant_region"]
"The numeric columns after the first of a CSV file with a header, as a matrix."
function readcsv(name)
    rows = split.(readlines(joinpath(dir, name))[2:end], ",")
    return reduce(vcat, [permutedims(parse.(Float64, r[2:end])) for r in rows])
end

# ---- the data
"The light curves of every band (one file per band), stacked band by band."
readbands(prefix) = reduce(vcat, [readcsv(band_file(prefix, b)) for b in 1:NB])
lc = readbands("lightcurves")                         # columns after `band`: nu, t_M, t_s, I, Q, U, V
bo = readbands("lightcurves_by_order")                # nu, t_M, t_s, then IQUV of orders 0, 1, ≥2
sp = readcsv("spectrum.csv")
nt = size(lc, 1) ÷ NB
t_M = lc[1:nt, 2]; t_h = lc[1:nt, 3] ./ 3600
I = reshape(lc[:, 4], nt, NB)
I0 = reshape(bo[:, 4], nt, NB); I1 = reshape(bo[:, 8], nt, NB)
Imean = sp[:, 4]; Imin = sp[:, 6]; Imax = sp[:, 7]; f1 = sp[:, 8]; f2 = sp[:, 9]
P = permutedims(readcsv("parcels.csv"))                     # parameter rows × parcels

# ---- the band-6 image at the time of the day's median band-6 flux
a = meta["spacetime"]["spin"]; θo = deg2rad(meta["spacetime"]["inclination_deg"])
fov = model == "jet" ? 60.0 : model == "funnel" ? 24.0 : 36.0; res = 240
cam = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
sc = GeodesicCache(CPU(), cam, Val(160); store_samples = true); regenerate!(sc, a, θo; marcher = Recurrence(64))
tab = LookbackTable(sc; margin = 5.0); sc = nothing; GC.gc()
cache = GeodesicCache(backend, cam, Val(160); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
τh = half_orbit_times(cache)
t_img = t_M[sortperm(I[:, 4])[nt ÷ 2]]
Pa, box, _ = frame_parcels(P, t_img, tab; cutoff = CUTOFF)
dev(x) = (y = KA.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
out = KA.allocate(backend, OrderState{1,Float64}, npixels(cache))
render_orders!(out, cache, dev(Pa), dev(box), τh, t_img, (NU[4],), LUNIT; cutoff = CUTOFF)
Iimg = reshape(Geodesics.unsort(cache, [st.st[1].S[1] * νhat(NU[4])^3 for st in Array(out)]), res, res)
Tb = Iimg .* Tr.CL^2 ./ (2 * Tr.KBOL * NU[4]^2) ./ 1e10
xs = range(-fov / 2, fov / 2; length = res)
αc, ψc, ρc = critical_curve(a, θo)

# ---- the figure
ink = colorant"#0b0b0b"; ink2 = colorant"#52514e"; muted = colorant"#898781"; grid = colorant"#e1e0d9"
c1 = colorant"#2a78d6"; c2 = colorant"#eb6834"; c3 = colorant"#1baf7a"
set_theme!(Theme(fontsize = 13, backgroundcolor = colorant"#fcfcfb", textcolor = ink,
                 Axis = (backgroundcolor = colorant"#fcfcfb", xgridcolor = grid, ygridcolor = grid, xminorgridvisible = false,
                         leftspinecolor = colorant"#c3c2b7", bottomspinecolor = colorant"#c3c2b7", topspinevisible = false,
                         rightspinevisible = false, xtickcolor = muted, ytickcolor = muted, xticklabelcolor = ink2, yticklabelcolor = ink2,
                         titlefont = :regular, titlealign = :left, titlesize = 13)))
fig = Figure(size = (1600, 1250))
left = fig[1, 1] = GridLayout(); right = fig[1, 2] = GridLayout()
Label(fig[0, 1:2], @sprintf("Sgr A*, %s-dominated: a day of slow-light light curves on ALMA bands 3–9 (a = %g, i = %g°)", model, a, rad2deg(θo)),
      fontsize = 16, halign = :left, tellwidth = false)
# image
ax = Axis(left[1, 1], aspect = DataAspect(), title = @sprintf("band 6 (233 GHz) at t = %.0f M (%.2f h)", t_img, t_img * TUNIT / 3600), xlabel = "α [M]", ylabel = "β [M]")
hm = heatmap!(ax, xs, xs, Tb, colormap = :afmhot, colorrange = (0, quantile(vec(Tb), 0.998)))
lines!(ax, [Point2f(αc + ρ * cos(ψ), ρ * sin(ψ)) for (ψ, ρ) in zip(vcat(ψc, ψc[1]), vcat(ρc, ρc[1]))], color = (:white, 0.5), linestyle = :dash, linewidth = 1)
Colorbar(left[1, 2], hm, label = "Tb [10¹⁰ K]", width = 10)
limits!(ax, -fov / 2, fov / 2, -fov / 2, fov / 2)
# spectrum
νG = collect(NU) ./ 1e9
ax = Axis(left[2, 1:2], xscale = log10, yscale = log10, title = "spectrum: the day's mean and range against the target", xlabel = "ν [GHz]", ylabel = "Fν [Jy]",
          xticks = [100, 200, 300, 500, 700], yticks = [1, 1.5, 2, 2.5, 3, 4])
band!(ax, νG, Imin, Imax, color = (c1, 0.15))
lines!(ax, νG, Imean, color = c1, linewidth = 2)
scatter!(ax, νG, Imean, color = c1, markersize = 9, label = "model: the day's mean (band: range)")
errorbars!(ax, νG, collect(FTARGET), collect(FTARGET .* FRACERR), color = c2, whiskerwidth = 6)
scatter!(ax, νG, collect(FTARGET), color = c2, marker = :diamond, markersize = 10, label = "target (EHT 2017; Bower et al. 2019)")
ylims!(ax, 0.8 * min(minimum(Imin), minimum(FTARGET)), 1.2 * max(maximum(Imax), maximum(FTARGET)))
axislegend(ax, position = :lb, framevisible = false, labelsize = 11)
# share by lensing order
ax = Axis(left[3, 1:2], xscale = log10, title = "share of the day's light that has gone around the hole", xlabel = "ν [GHz]", ylabel = "fraction",
          xticks = [100, 200, 300, 500, 700])
lines!(ax, νG, f1, color = c2, linewidth = 2); scatter!(ax, νG, f1, color = c2, markersize = 8)
lines!(ax, νG, f2, color = c3, linewidth = 2); scatter!(ax, νG, f2, color = c3, markersize = 8)
text!(ax, νG[end], f1[end], text = " order 1", color = ink2, align = (:left, :center), fontsize = 11)
text!(ax, νG[end], f2[end], text = " orders ≥ 2", color = ink2, align = (:left, :center), fontsize = 11)
xlims!(ax, 85, 1000); ylims!(ax, 0, 1.15 * max(maximum(f1), maximum(f2)))
rowsize!(left, 1, Relative(0.45))
colsize!(fig.layout, 1, Relative(0.34))
# the day, band by band, each over its mean on a common scale
Label(right[1, 1], "the day: total flux density over its mean, band by band", halign = :left, tellwidth = false, fontsize = 13)
rel = I ./ Imean'
lo, hi = extrema(rel)
axes_day = map(1:NB) do b
    ax = Axis(right[b + 1, 1], ylabel = @sprintf("B%d", BANDS[b]), ylabelrotation = 0, xticklabelsvisible = b == NB, xlabel = b == NB ? "t [h]" : "",
              yticks = [0.5, 1.0, 1.5, 2.0, 2.5])
    lines!(ax, t_h, rel[:, b], color = c1, linewidth = 1)
    text!(ax, t_h[end], hi, text = @sprintf("%.2f Jy, rms %.0f%%", Imean[b], 100 * std(I[:, b]) / Imean[b]), align = (:right, :top), color = ink2, fontsize = 10)
    ylims!(ax, lo - 0.05, hi + 0.05); xlims!(ax, 0, t_h[end])
    ax
end
linkxaxes!(axes_day...)
# three hours of band 6 by order
span = min(3.0, t_h[end]); mid = t_h[end] / 2
k1 = argmin(abs.(t_h .- (mid - span / 2))); k2 = argmin(abs.(t_h .- (mid + span / 2)))
Label(right[NB + 2, 1], @sprintf("band 6, hours %.1f–%.1f: all the light (top) and the order-1 light alone (bottom)", t_h[k1], t_h[k2]),
      halign = :left, tellwidth = false, fontsize = 13)
axz1 = Axis(right[NB + 3, 1], ylabel = "Fν [Jy]", xticklabelsvisible = false)
lines!(axz1, t_h[k1:k2], I[k1:k2, 4], color = c1, linewidth = 1.5)
axz2 = Axis(right[NB + 4, 1], ylabel = "order 1 [Jy]", xlabel = "t [h]")
lines!(axz2, t_h[k1:k2], I1[k1:k2, 4], color = c2, linewidth = 1.5)
linkxaxes!(axz1, axz2); xlims!(axz2, t_h[k1], t_h[k2])
for r in 2:NB + 1
    rowgap!(right, r, 2)
end
rowsize!(right, NB + 3, Relative(0.14)); rowsize!(right, NB + 4, Relative(0.1))
save(joinpath(dir, "overview.png"), fig, px_per_unit = 1.5)
@info "saved" joinpath(dir, "overview.png")
mirror_dataset(joinpath(dir, "overview.png"))
