# The mp4 of a region dataset's movie (scripts/movie_regions.jl): the band's images through the day, all the light, the
# direct light (order 0) and the light that went half an orbit about the hole (order 1), side by side in brightness
# temperature, over two scrolling hours of the grid's light curves by order.
#
#     julia --project=. scripts/draw_movie.jl datasets/sgra_<model>_dominated [--cubes movies] [--fps 30]
#
# Writes movie_band<b>_<ν>GHz.mp4 in the dataset's directory (not in git; copied to the datasets mirror, if set).
using Statistics, Printf, TOML, Mmap
using CairoMakie
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "SlowLight.jl")); using .SlowLight
include(joinpath(@__DIR__, "mirror.jl")); using .Mirror
const Tr = KerrSplat.Transfer

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
dir = ARGS[1]
cubedir = getstr("--cubes", joinpath(@__DIR__, "..", "movies"))
fps = getopt("--fps", 30)
meta = TOML.parsefile(joinpath(dir, "movie_metadata.toml"))
model = meta["model"]; ν = meta["nu_GHz"] * 1e9; fov = meta["fov_M"]; res = meta["res"]; band = meta["band"]
a = meta["spin"]; θo = deg2rad(meta["inclination_deg"])

"A float32 NumPy array of C-order shape (nt, ny, nx), memory-mapped read-only as the Julia array (nx, ny, nt)."
function npy_read(path)
    io = open(path)
    read(io, 6) == UInt8[0x93, codeunits("NUMPY")...] || error("$path is not a NumPy file")
    read(io, 2); hdr = String(read(io, ltoh(read(io, UInt16))))
    occursin("'<f4'", hdr) && occursin("'fortran_order': False", hdr) || error("$path: expected little-endian float32 in C order")
    nt, ny, nx = parse.(Int, match(r"'shape': \((\d+), (\d+), (\d+)\)", hdr).captures)
    return Mmap.mmap(io, Array{Float32,3}, (nx, ny, nt), position(io))
end
cube_all, cube_n0, cube_n1 = (npy_read(joinpath(cubedir, f)) for f in meta["cubes"]["files"])
nt = size(cube_all, 3)

# ---- the light curves (Stokes I by order) and the brightness temperature of a pixel's flux density
rows = [parse.(Float64, split(l, ",")) for l in readlines(joinpath(dir, meta["lightcurves"]))[2:end]]
t_M = [r[3] for r in rows]; t_h = [r[4] for r in rows] ./ 3600
I0 = [r[5] for r in rows]; I1 = [r[9] for r in rows]; I2 = [r[13] for r in rows]; Iall = I0 .+ I1 .+ I2
Ωpix = (meta["pixel_M"] * LUNIT / D_CM)^2
tb = Float32(Tr.JY / Ωpix * Tr.CL^2 / (2 * Tr.KBOL * ν^2) / 1e10)    # 10¹⁰ K per (Jy per pixel)
# colour scales held through the movie: only the brightest tenth of the frames saturate
frame_max(c) = [maximum(@view c[:, :, k]) for k in 1:nt]
vmax = tb * quantile(frame_max(cube_all), 0.9)
vmax1 = tb * quantile(frame_max(cube_n1), 0.9)

# ---- the figure
ink = colorant"#0b0b0b"; ink2 = colorant"#52514e"; muted = colorant"#898781"; grid = colorant"#e1e0d9"
c1 = colorant"#2a78d6"; c2 = colorant"#eb6834"; c3 = colorant"#1baf7a"
set_theme!(Theme(fontsize = 15, backgroundcolor = colorant"#fcfcfb", textcolor = ink,
                 Axis = (backgroundcolor = colorant"#fcfcfb", xgridcolor = grid, ygridcolor = grid,
                         leftspinecolor = colorant"#c3c2b7", bottomspinecolor = colorant"#c3c2b7", topspinevisible = false,
                         rightspinevisible = false, xtickcolor = muted, ytickcolor = muted, xticklabelcolor = ink2, yticklabelcolor = ink2,
                         titlefont = :regular, titlealign = :left, titlesize = 15)))
fig = Figure(size = (1600, 960))
k = Observable(1)
clock(t) = (s = round(Int, t * TUNIT); @sprintf("%02d:%02d:%02d", s ÷ 3600, (s % 3600) ÷ 60, s % 60))
Label(fig[0, 1:5], @sprintf("Sgr A*, %s-dominated · band %d (%.4g GHz) · a = %g, i = %g°", model, band, ν / 1e9, a, rad2deg(θo)),
      fontsize = 19, halign = :left, tellwidth = false)
Label(fig[0, 1:5], @lift(@sprintf("t = %.0f M   %s", t_M[$k], clock(t_M[$k]))), fontsize = 19, halign = :right, tellwidth = false)
xs = range(-fov / 2, fov / 2; length = res)
αc, ψc, ρc = critical_curve(a, θo)
ring = [Point2f(αc + ρ * cos(ψ), ρ * sin(ψ)) for (ψ, ρ) in zip(vcat(ψc, ψc[1]), vcat(ρc, ρc[1]))]
panels = ((cube_all, "all the light", vmax), (cube_n0, "direct (n = 0)", vmax), (cube_n1, "one half-orbit (n = 1)", vmax1))
for (q, (cube, title, vm)) in enumerate(panels)          # columns: images 1, 2 and 4, colour bars 3 and 5
    ax = Axis(fig[1, (1, 2, 4)[q]], aspect = DataAspect(), title = title, xlabel = "α [M]", ylabel = q == 1 ? "β [M]" : "",
              xgridvisible = false, ygridvisible = false)
    hm = heatmap!(ax, xs, xs, @lift(tb .* @view(cube[:, :, $k])), colormap = :afmhot, colorrange = (0, vm), interpolate = true)
    lines!(ax, ring, color = (:white, 0.35), linestyle = :dash, linewidth = 1)
    limits!(ax, -fov / 2, fov / 2, -fov / 2, fov / 2)
    q >= 2 && Colorbar(fig[1, (0, 3, 5)[q]], hm, label = q == 3 ? "Tb [10¹⁰ K] (own scale)" : "Tb [10¹⁰ K] (both)", width = 10)
end
# two scrolling hours of the light curves
now_h = @lift(t_h[$k])
win = @lift(($now_h - 1, $now_h + 1))
ax1 = Axis(fig[2, 1:5], ylabel = "Fν [Jy]", xticklabelsvisible = false, title = "the grid's light curves, two hours about now")
lines!(ax1, t_h, Iall, color = c1, linewidth = 2, label = "all")
lines!(ax1, t_h, I0, color = c3, linewidth = 1.5, label = "direct (n = 0)")
vlines!(ax1, now_h, color = ink2, linewidth = 1)
axislegend(ax1, position = :lt, framevisible = false, orientation = :horizontal, labelsize = 13)
ax2 = Axis(fig[3, 1:5], ylabel = "n = 1 [Jy]", xlabel = "t [h]", yticks = WilkinsonTicks(3))
lines!(ax2, t_h, I1, color = c2, linewidth = 2)
vlines!(ax2, now_h, color = ink2, linewidth = 1)
ylims!(ax1, 0, 1.05 * maximum(Iall)); ylims!(ax2, 0, 1.05 * maximum(I1))
on(win) do lims
    xlims!(ax1, lims...); xlims!(ax2, lims...)
end
notify(k)
rowsize!(fig.layout, 1, Relative(0.58)); rowsize!(fig.layout, 3, Relative(0.13))
colgap!(fig.layout, 3, 30)

mp4 = joinpath(dir, meta["mp4"])
CairoMakie.activate!(px_per_unit = 1.2)                          # 1920 × 1152
t0 = time()
record(fig, mp4, 1:nt; framerate = fps) do i
    k[] = i
    i % 200 == 0 && (@info "drawing" frame = i of = nt seconds = round(time() - t0, digits = 1); flush(stderr))
end
@info "saved" mp4 megabytes = round(filesize(mp4) / 1e6, digits = 1) seconds = round(time() - t0, digits = 1); flush(stderr)
mirror_dataset(mp4)
