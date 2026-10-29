# Tunes each emitting region of the Sgr A* region models to the target spectrum: the profile parameters
# x = (ln n₀, ln T₀, sT, sn) of a population of that region alone, keeping NACTIVE[region] parcels alive on average, are
# fitted by Levenberg–Marquardt so that its mean spectrum over snapshots spread through a day matches the target on
# ALMA bands 3–9 (the log flux densities, weighted by FRACERR), with weak priors (σ = 1.5) holding the slopes near
# their starting values where the spectrum does not decide them. Every model of scripts/sgra_regions.jl draws its
# regions with these parameters: its dominant region keeping 0.8 NACTIVE parcels alive and the other two 0.1 NACTIVE,
# so that on average they give 80%, 10% and 10% of the target.
#
#     julia -t auto --project=. scripts/tune_regions.jl [--regions disk,funnel,jet] [--snapshots 16] [--cpu]
#
# Writes datasets/sgra_regions_tuning.toml (read by scripts/sgra_regions.jl).
using CUDA, KernelAbstractions, StaticArrays, LinearAlgebra, Statistics, Printf, Dates, TOML, Random
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
include(joinpath(@__DIR__, "SlowLight.jl")); using .SlowLight
const KA = KernelAbstractions

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
backend = ("--cpu" in ARGS || !CUDA.functional()) ? CPU() : CUDABackend()
regions = Symbol.(split(getstr("--regions", "disk,funnel,jet"), ","))
K = getopt("--snapshots", 16)
outfile = joinpath(@__DIR__, "..", "datasets", "sgra_regions_tuning.toml")

a = SPIN; θo = deg2rad(INCLINATION)
rc = ring_camera(a, θo; TUNING_CAMERA...)
cache, tab, w, τh = camera_setup(backend, rc, a, θo, Val(TUNING_CAMERA_SAMPLES))
out = KA.allocate(backend, OrderState{NB,Float64}, npixels(cache))
dev(x) = (y = KA.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
times = [(k - 0.5) * TMAX / K for k in 1:K]
@info "tuning" regions K rays = length(rc.area) samples = TUNING_CAMERA_SAMPLES; flush(stderr)

"Mean total flux density per band [Jy] of the parcels `P` over the snapshot times, and the spectrum of each snapshot."
function mean_spectrum(P)
    S = zeros(NB, K)
    for (k, t) in enumerate(times)
        Pa, box, _ = frame_parcels(P, t, tab; cutoff = CUTOFF)
        render_orders!(out, cache, dev(Pa), dev(box), τh, t, NU, LUNIT; cutoff = CUTOFF)
        S[:, k] = frame_flux(out, w, NU)[1, 1, :]
    end
    return vec(mean(S; dims = 2)), S
end

"""
    spectral_residuals(pop, x, x0) -> (r, F, S)

The tuning's residuals for the profile parameters `x` of a population: the log mean flux densities against the
target in units of FRACERR, then the slopes' departures from their starting values `x0` in units of 1.5; with the
mean spectrum and the snapshots' spectra. (A function of its own, not a closure in the loop below: a closure that
assigns names the loop also assigns would write the loop's variables, and the Jacobian would subtract the new
residuals from themselves.)
"""
function spectral_residuals(pop, x, x0)
    Fm, Sk = mean_spectrum(plasma(pop, x))
    res = vcat((log.(Fm) .- log.(collect(FTARGET))) ./ collect(FRACERR), (x[3:4] .- x0[3:4]) ./ 1.5)
    return res, Fm, Sk
end

results = isfile(outfile) ? TOML.parsefile(outfile) : Dict{String,Any}()
for region in regions
    pop = SlowLight.GENERATORS[region](Xoshiro(TUNING_SEED), births_span(region, 0.0, TMAX), NACTIVE[region]; a)
    x0 = X_START[region]
    residuals(x) = spectral_residuals(pop, x, x0)
    x = copy(x0)
    r, F, S = residuals(x); χ2 = sum(abs2, r); λ = 1e-2; h = 0.02
    @info "tuning $region" iteration = 0 χ2 x flux = round.(F, digits = 3) parcels = length(pop); flush(stderr)
    t_start = time()
    for it in 1:15
        J = reduce(hcat, [(residuals(x .+ h .* (eachindex(x) .== k))[1] .- r) ./ h for k in eachindex(x)])
        A = J' * J; g = J' * r
        accepted = false; done = false
        while λ < 1e6
            δ = -((A + λ * Diagonal(diag(A))) \ g)
            r2, F2, S2 = residuals(x .+ δ); χ2new = sum(abs2, r2)
            if χ2new < χ2
                done = (χ2 - χ2new) < 1e-3 * χ2 || norm(δ) < 1e-3
                x = x .+ δ; r = r2; F = F2; S = S2; χ2 = χ2new; λ = max(λ / 3, 1e-6); accepted = true
                @info "tuning $region" iteration = it χ2 x flux = round.(F, digits = 3) minutes = round((time() - t_start) / 60, digits = 1); flush(stderr)
                break
            end
            λ *= 4
        end
        (accepted && !done) || break
    end
    rms = vec(std(S; dims = 2)) ./ F
    results[string(region)] = Dict("x" => x, "n0_cm3" => exp(x[1]), "T0_K" => exp(x[2]), "temperature_slope" => x[3], "density_slope" => x[4],
                                   "chi2" => χ2, "mean_flux_Jy" => F, "snapshot_rms_fraction" => rms, "nactive" => NACTIVE[region],
                                   "parcels" => length(pop), "seed" => TUNING_SEED, "snapshots" => K, "created" => string(now()))
    open(io -> TOML.print(io, results; sorted = true), outfile, "w")
    @info "tuned $region" x χ2 flux = round.(F, digits = 3) snapshot_rms = round.(rms, digits = 3); flush(stderr)
end
