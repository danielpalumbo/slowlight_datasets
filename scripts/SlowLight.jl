# Shared machinery for the slow-light datasets: the source and the ALMA bands, splats that may move radially, a
# fused-march consumer that renders every band in one pass and splits each ray by lensing order, and a camera whose
# points follow the critical curve so that the n = 1 ring is resolved at a fraction of the cost of a uniform grid.
module SlowLight

using KernelAbstractions, StaticArrays, LinearAlgebra, Krang, Random
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
using KerrSplat.Splats: quaternion_rotation, WEIGHT_CUTOFF, SUPPORT_RADIUS2
using KerrSplat.Transfer: _scalar, EE, ME, CL, dexter_II, dexter_IQ, dexter_IV, planck, besselk012
const KA = KernelAbstractions
const Tr = KerrSplat.Transfer
const Adapt = KerrSplat.Splats.Adapt                     # KerrSplat's own (its manifest version)

export M_SOLAR, D_CM, LUNIT, TUNIT, BANDS, NU, NB, target_flux, FTARGET, FRACERR, ΘUNIT
export thermal_synchrotron_k
export MOVING_SPLAT_PARAMS, NMOVINGPARAMS, MovingSplats, moving_weight, at_rest, MultiBand, OrderState, half_orbit_mino, half_orbit_times
export critical_curve, RingCamera, ring_camera, render_orders!, frame_flux, frame_parcels, LookbackTable
export REGIONS, X_START, births_span, model_parcels, SPIN, INCLINATION, TMAX, NACTIVE, TUNING_SEED, CUTOFF
export PRODUCTION_CAMERA, PRODUCTION_CAMERA_SAMPLES, TUNING_CAMERA, TUNING_CAMERA_SAMPLES, camera_setup
export zamo_γβ_rotation, keplerian_γβ, Population, plasma, disk_population, funnel_population, jet_population, quaternion_spherical

# ---- the source: GRAVITY Collaboration (2019), M = 4.152×10⁶ M☉ at D = 8.178 kpc
const M_SOLAR = 4.152e6
const D_CM = 8.178e3 * Tr.PC
const LUNIT = gravitational_radius(M_SOLAR)               # GM/c² [cm]
const TUNIT = LUNIT / Tr.CL                               # GM/c³ [s]
const ΘUNIT = Tr.KBOL / (Tr.ME * Tr.CL^2)                # Θe per kelvin

# ---- the target spectrum on ALMA's standard continuum frequencies for bands 3–9 (see scripts/test_sgra_disk20.jl and
# the README): 2.0 Jy at 98 GHz and 2.4 Jy at 213–229 GHz (EHT Collaboration 2022, Paper II), α = −0.26 above 233 GHz
# (Bower et al. 2019), a power law between
const BANDS = (3, 4, 5, 6, 7, 8, 9)
const NU = (97.5e9, 145.0e9, 203.0e9, 233.0e9, 343.5e9, 405.0e9, 679.0e9)
const NB = length(NU)
target_flux(ν) = ν <= 233e9 ? 2.4 * (ν / 233e9)^(log(2.4 / 2.0) / log(233 / 97.5)) : 2.4 * (ν / 233e9)^-0.26
const FTARGET = target_flux.(NU)
const FRACERR = (0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.20)

# ---- kinematics
"ZAMO-frame γβ along φ̂ of a fluid element rotating rigidly at angular velocity `Ω` at Boyer–Lindquist (r, θ), spin `a`."
function zamo_γβ_rotation(r, θ, Ω, a)
    s2 = sin(θ)^2; Σ = r^2 + a^2 * cos(θ)^2; Δ = r^2 - 2r + a^2
    A = (r^2 + a^2)^2 - a^2 * Δ * s2
    v = sqrt(A / Σ) * sqrt(s2) * (Ω - 2a * r / A) / sqrt(Σ * Δ / A)       # ϖ (Ω − ω) / α
    abs(v) < 1 || throw(DomainError(v, "rotation at Ω = $Ω is superluminal at r = $r, θ = $θ"))
    return v / sqrt(1 - v^2)
end
"ZAMO-frame γβ of a prograde Keplerian circular orbit at radius `r` in the equatorial plane of spin `a`."
keplerian_γβ(r, a) = zamo_γβ_rotation(r, π / 2, 1 / (r^1.5 + a), a)

# ---- splats that move radially
"""
    MOVING_SPLAT_PARAMS

KerrSplat's 21 polarized-splat rows followed by `:vr`, the radial speed of the pattern (M per M), and `:logwd`, the ln
of the envelope's width after its peak (`:logw` is the width before it). Splat i is born at its centre x₀ and
brightens there with the envelope exp(−½((t − t₀)/w)²) until t₀; after t₀ it fades with the width wd while its centre
moves out to f(t) x₀, with f = 1 + vr (t − t₀)/|x₀|. At every time its pattern turns by ω(t − t₀) about the spin
axis, as KerrSplat's does. While it moves its scales grow by f, its density falls as f⁻² and its field as f⁻¹ (a
disturbance carried out through a conical flow of constant speed, where n ∝ r⁻² and B ∝ r⁻¹); its temperature and
velocity stay as they are. With vr = 0 and wd = w it is KerrSplat's polarized splat to the bit.
"""
const MOVING_SPLAT_PARAMS = (POLARIZED_SPLAT_PARAMS..., :vr, :logwd)
const NMOVINGPARAMS = length(MOVING_SPLAT_PARAMS)
const FMIN = 0.05

"""
    moving_weight(p, i, t, x, y, z, R2 = SUPPORT_RADIUS2) -> (W, f)

The density weight W = G/f² of moving splat `i` at time `t` and quasi-Cartesian (x, y, z), G being the Gaussian ×
envelope of unit peak, and its expansion factor f. W is zero wherever it is below the cutoff e^(−R2/2) (the tests on
the envelope and on the support sphere are exact consequences of it: q² + τ² < R2 − 4 ln f with q² ≥ d²/(f σmax)²);
with KerrSplat's cutoff (the default) and f = 1 that is KerrSplat's weight and culling, so a splat at rest renders to
the bit as KerrSplat's.
"""
@inline function moving_weight(p, i, t, x, y, z, R2 = SUPPORT_RADIUS2)
    T = typeof(float(x))
    @inbounds begin
        Δt = t - p[11, i]
        τ2 = (Δt * exp(-(Δt < 0 ? p[12, i] : p[23, i])))^2
        τ2 > R2 && return zero(T), one(T)                          # the envelope alone is below the cutoff
        r0 = sqrt(p[1, i]^2 + p[2, i]^2 + p[3, i]^2)
        f = max(1 + p[22, i] * max(Δt, zero(Δt)) / r0, T(FMIN))
        room = R2 - 4 * log(f) - τ2                                 # what q² may still use
        room > 0 || return zero(T), f
        sφ, cφ = sincos_pair(p[21, i] * Δt)
        dx = x - f * (p[1, i] * cφ - p[2, i] * sφ)
        dy = y - f * (p[1, i] * sφ + p[2, i] * cφ)
        dz = z - f * p[3, i]
        smax = max(p[4, i], p[5, i], p[6, i])
        dx^2 + dy^2 + dz^2 > room * (f * f) * exp(2 * smax) && return zero(T), f
        dr = SVector(dx * cφ + dy * sφ, -dx * sφ + dy * cφ, dz) / f
        u = quaternion_rotation(p[7, i], p[8, i], p[9, i], p[10, i])' * dr
        q2 = (u[1] * exp(-p[4, i]))^2 + (u[2] * exp(-p[5, i]))^2 + (u[3] * exp(-p[6, i]))^2
        return exp(-(q2 + τ2) / 2) / (f * f), f
    end
end

"""
    MovingSplats(params, t_obs)

The moving splats as a `Transfer.RadiativeTransport` model (one band at a time, through KerrSplat's own consumer):
the reference that `MultiBand` is checked against.
"""
struct MovingSplats{P,V}
    params::P
    t_obs::V
end
Adapt.@adapt_structure MovingSplats
MovingSplats(params::AbstractMatrix{T}, t_obs::Real) where {T} = MovingSplats(params, fill!(similar(params, 1), T(t_obs)))
Tr.nelements(m::MovingSplats) = size(m.params, 2)
Tr.coefficient_type(m::MovingSplats, ν_obs) = promote_type(eltype(m.params), typeof(ν_obs))
@inline function Tr.element(m::MovingSplats, i, pix, s::GeodesicSample{T}, ν_obs) where {T}
    p = m.params
    x, y, z = quasi_cartesian_kerr_schild(Krang.metric(pix), s.r, s.θ, s.ϕ)
    W, f = moving_weight(p, i, @inbounds(m.t_obs[1]) - s.t, x, y, z)
    W > T(WEIGHT_CUTOFF) || return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
    @inbounds return splat_coefficients(pix, s, ν_obs, exp(p[13, i]) * W, p[14, i], p[15, i] - log(f),
                                        p[16, i], p[17, i], p[18, i], p[19, i], p[20, i])
end

"Moving splats from KerrSplat's polarized splats: at rest, with symmetric envelopes."
at_rest(p21::AbstractMatrix) = vcat(p21, zeros(1, size(p21, 2)), p21[12:12, :])

# ---- every band in one pass, split by lensing order
"""
    OrderState{NB,T}

Accumulator of the multi-band consumer: one `RadiativeState` per band, the Stokes vectors of every band at the end
of the ray's order-0 and order-≤1 parts (`s0`, `s1`, set when the order first exceeds 0 and 1), and the order `n` of
the last sample. The order of a sample at Mino time τ from the observer is ⌊τ/τ½⌋, τ½ being the Mino time of one
half-oscillation of the ray in θ (`half_orbit_mino`): the number of half-orbits about the hole the light has made.
A ray that is nearly straight covers one half-oscillation from infinity to infinity, so all of its light is order 0
wherever it comes from; the k-th equatorial crossing of any ray falls in ((k−1)τ½, kτ½), so for an equatorial source
order n is the n-th image of Gralla & Lupsasca (the (n+1)-th crossing). Truncating a ray at an order boundary drops
both the emission and the absorption beyond it, as KerrSplat's `nmax`.
"""
struct OrderState{NB,T}
    st::NTuple{NB,RadiativeState{T}}
    s0::NTuple{NB,SVector{4,T}}
    s1::NTuple{NB,SVector{4,T}}
    n::Int32
end
Base.zero(::Type{OrderState{NB,T}}) where {NB,T} =
    OrderState(ntuple(_ -> zero(RadiativeState{T}), Val(NB)), ntuple(_ -> zero(SVector{4,T}), Val(NB)),
               ntuple(_ -> zero(SVector{4,T}), Val(NB)), Int32(0))

"""
    half_orbit_mino(a, θo, α, β) -> τ½

The Mino time of one half-oscillation in θ (from θ₋ to θ₊) of the ray of screen coordinates (α, β) for spin `a` and
inclination `θo`: 2K(u₊/u₋)/(a√−u₋), with u± = Δ ± √(Δ² + η/a²), Δ = (1 − (η + λ²)/a²)/2 the roots in cos²θ of the
polar potential, and K by the arithmetic–geometric mean. Infinite for η ≤ 0 (rays that stay in one hemisphere).
"""
function half_orbit_mino(a, θo, α, β)
    λ = -α * sin(θo); η = (α^2 - a^2) * cos(θo)^2 + β^2
    η > 0 || return Inf
    Δ = (1 - (η + λ^2) / a^2) / 2
    up = Δ + sqrt(Δ^2 + η / a^2); um = Δ - sqrt(Δ^2 + η / a^2)
    x, y = 1.0, sqrt(1 - up / um)                       # K(m) = π / (2 AGM(1, √(1 − m))), m = u₊/u₋ < 0
    while abs(x - y) > 1e-15 * x
        x, y = (x + y) / 2, sqrt(x * y)
    end
    return 2 * (π / (2 * x)) / (a * sqrt(-um))
end

"""
    MultiBand(params, box, τhalf, t_obs, νs, L; cutoff)

Fused-march consumer for moving splats (`NMOVINGPARAMS × n`, on the backend) at observation time `t_obs` and the
frequencies `νs` (a tuple): per sample the geometry of every splat (position, weight, field, local frame) is formed
once and the coefficients and the transfer step once per band, each band composed exactly as KerrSplat's
`RadiativeTransport` composes it, into an `OrderState`. A splat counts where its density weight exceeds `cutoff`
(KerrSplat's `WEIGHT_CUTOFF` by default, with which the result is KerrSplat's). `box` (6 × n, from `frame_parcels`) bounds each splat's
support for this frame in emission time, radius and height; a sample outside it skips the splat before any of its
geometry is formed (the box is conservative, so the result is the same). `τhalf` holds each ray's τ½ in the cache's
sorted order, for the order of its samples (see `OrderState`).
"""
struct MultiBand{NB,P,B,V,T}
    params::P
    box::B
    τhalf::V
    t_obs::T
    νs::NTuple{NB,T}
    L::T
    cutoff::T
    R2::T
end
Adapt.@adapt_structure MultiBand
MultiBand(params::P, box::B, τhalf::V, t_obs::Real, νs::NTuple{NB,<:Real}, L::Real; cutoff = WEIGHT_CUTOFF) where {NB,P,B,V} =
    MultiBand{NB,P,B,V,Float64}(params, box, τhalf, t_obs, Float64.(νs), L, cutoff, -2 * log(cutoff))

@inline function (c::MultiBand{NB})(acc::OrderState{NB,T}, j, k, s::GeodesicSample, Δτ, pix) where {NB,T}
    met = Krang.metric(pix)
    (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || return acc
    # the order of this sample (k Δτ is its Mino time)
    n = Int32(min(floor(k * Δτ / @inbounds(c.τhalf[j])), T(1000)))
    zs = s.r * cos(s.θ)
    # (maps over the band tuples, not closures over them: the tuples are reassigned in the loop, and a closure that
    # captures a reassigned variable boxes it, which costs an allocation per sample)
    sts = acc.st
    s0 = (acc.n < 1 && n >= 1) ? map(st -> st.S, sts) : acc.s0
    s1 = (acc.n < 2 && n >= 2) ? map(st -> st.S, sts) : acc.s1
    # the coefficients of every splat present, every band
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    te = c.t_obs - s.t
    p = c.params; box = c.box
    js = map(_ -> zero(SVector{4,T}), c.νs); as = js; ρs = map(_ -> zero(SVector{3,T}), c.νs)
    on = map(_ -> false, c.νs)
    for i in 1:size(p, 2)
        @inbounds (te < box[1, i] || te > box[2, i] || s.r < box[3, i] || s.r > box[4, i] || zs < box[5, i] || zs > box[6, i]) && continue
        W, f = moving_weight(p, i, te, x, y, z, c.R2)
        W > c.cutoff || continue
        @inbounds begin
            ne = exp(p[13, i]) * W; Θe = exp(p[14, i]); Bmag = exp(p[15, i] - log(f))
            sθ, cθ = sincos_pair(p[16, i]); sϕ, cϕ = sincos_pair(p[17, i])
            ũ = SVector(p[18, i], p[19, i], p[20, i])
        end
        fr = local_frame(pix, s, ũ, SVector(Bmag * sθ * cϕ, Bmag * sθ * sϕ, Bmag * cθ))
        terms = band_coefficients(ne, Θe, Bmag, Tr.safe_acos(fr.cosθB), fr, c.νs)
        js = map((x, t) -> x + t[1], js, terms)
        as = map((x, t) -> x + t[2], as, terms)
        ρs = map((x, t) -> x + t[3], ρs, terms)
        on = map((x, t) -> x | t[4], on, terms)
    end
    new = band_steps(sts, on, js, as, ρs, c.νs, c.L, s.r * s.r + met.spin^2 * cos(s.θ)^2, Δτ)
    return OrderState(new, s0, s1, n)
end

"""
    thermal_synchrotron_k(ne, Θe, B, ν, θ, k0, k1, k2) -> StokesCoefficients

KerrSplat's `Transfer.thermal_synchrotron` with its default prescription (Dexter's emissivities, Shcherbakov's ρ_V),
transcribed line for line, with the modified Bessel functions K₀, K₁, K₂ of 1/Θe passed in: they depend on the
temperature alone, so the multi-band consumer forms them once per parcel and sample instead of once per band (a
fifth of the cost of each call). The tests check it against KerrSplat's to the bit.
"""
@inline function thermal_synchrotron_k(ne, Θe, B, ν, θ, k0, k1, k2)
    T = typeof(float(ne * Θe * B * ν * θ))
    S = _scalar(T)                        # constants in the scalar type behind T (see `_scalar`)
    sinθ, cosθ = sincos_pair(θ)
    # rotativities (Dexter 2016 appendix B; ipole maxwell_juettner_rho_Q / rho_V)
    ω0 = S(EE) * B / (S(ME) * S(CL))
    ωp2 = 4 * S(π) * ne * S(EE)^2 / S(ME)
    ω = 2 * S(π) * ν
    ω0_ω = ω0 / ω                       # the ratios ω₀/ω ~ 1e-4 and ω_p²/ω² ~ 1e-9 stay in Float32's range,
    ωp2_ω2 = ωp2 / (ω * ω)              # where (2πν)⁴ ~ 1e48 does not
    if !(sinθ > 0)                          # along the field only ρ_V survives (x = 0 in the fits below); before any use of
        f0 = k2 > 0 ? k0 / k2 : one(T)      # sin θ, which is −9e-8 at θ = Float32(π) and would send sqrt a negative argument
        ρV0 = ω / S(CL) * (ωp2_ω2 * ω0_ω * f0 * cosθ)
        return StokesCoefficients(zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), ρV0)
    end
    x = Θe * sqrt(sqrt(S(2)) * sinθ * (S(1e3) * ω0_ω))
    extraterm = (S(0.011) * exp(-x / S(47.2)) - S(2)^S(-1 / 3) / S(3)^S(23 / 6) * S(π) * S(1e4) * (x + S(1e-16))^S(-8 / 3)) *
                (S(0.5) + S(0.5) * tanh((log(x) - log(S(120))) / S(0.1)))
    jffunc = S(2.011) * exp(-x^S(1.035) / S(4.7)) - cos(x / 2) * exp(-x^S(1.2) / S(2.73)) - S(0.011) * exp(-x / S(47.2)) + extraterm
    kratio = k2 > 0 ? k1 / k2 : one(T)
    eps11m22 = jffunc * ωp2_ω2 * ω0_ω^2 * (kratio + 6 * Θe) * sinθ^2   # ω_p² ω₀²/ω⁴ …
    ρQ = ω / (2 * S(CL)) * eps11m22
    fit_factor = (k2 > 0 ? k0 / k2 : one(T)) * (1 - S(0.11) * log(1 + S(0.035) * x))
    ρV = ω / S(CL) * (ωp2_ω2 * ω0_ω * fit_factor * cosθ)              # … and ω_p² ω₀/ω³
    # emissivities (Dexter 2016; ipole maxwell_juettner_dexter_*), ipole's sign for Q
    νs = 3 * S(EE) * B * sinθ / (4 * S(π) * S(ME) * S(CL)) * Θe^2 + 1
    xs = ν / νs
    pref = ne * S(EE)^2 * ν / (2 * sqrt(S(3)) * S(CL) * Θe^2)
    jI = pref * dexter_II(xs)
    jQ = pref * dexter_IQ(xs)
    jV = 2 * ne * S(EE)^2 * ν * cosθ / sinθ / (3 * sqrt(S(3)) * S(CL) * Θe^3) * dexter_IV(xs)
    # absorptivities by Kirchhoff's law
    Bν = planck(ν, Θe)
    inv_B = Bν > 0 ? inv(Bν) : zero(T)
    return StokesCoefficients(jI, jQ, jV, jI * inv_B, jQ * inv_B, jV * inv_B, ρQ, ρV)
end

"The screen-basis coefficients of one splat at every band (a tuple of `screen_coefficients` results)."
@inline function band_coefficients(ne, Θe, Bmag, θB, fr, νs)
    k0, k1, k2 = besselk012(inv(Θe))
    return map(ν -> screen_coefficients(thermal_synchrotron_k(ne, Θe, Bmag, ν / fr.g, θB, k0, k1, k2), fr, ν), νs)
end

"Each band's state advanced over one sample, as `Transfer.transfer_sample` does; bands with nothing there are left as they are."
@inline function band_steps(sts, on, js, as, ρs, νs, L, Σ, Δτ)
    return map(sts, on, js, as, ρs, νs) do st, o, j4, α4, ρ3, ν
        o || return st
        O, E = transfer_step(j4, α4, ρ3, L / νhat(ν) * Σ * Δτ)
        advance(st, O, E)
    end
end

"""
    render_orders!(out, cache, params, box, τhalf, t_obs, νs, L; cutoff = WEIGHT_CUTOFF, workgroup = 8) -> out

The `OrderState` of every ray of a regenerated `Fused` cache (sorted order) at `t_obs` for the bands `νs`, `τhalf`
being each ray's τ½ in sorted order (`half_orbit_times`).
"""
function render_orders!(out, cache::GeodesicCache, params, box, τhalf, t_obs, νs, L; cutoff = WEIGHT_CUTOFF, workgroup::Integer = 8)
    fused_march!(MultiBand(params, box, τhalf, t_obs, νs, L; cutoff), out, cache; workgroup)
    return out
end

"Each ray's τ½ (`half_orbit_mino`) in the sorted order of a regenerated cache, on its backend."
function half_orbit_times(cache::GeodesicCache)
    αs = Array(cache.αs); βs = Array(cache.βs)
    v = [half_orbit_mino(cache.spin, cache.θo, αs[q], βs[q]) for q in cache.perm_host]
    d = KA.allocate(cache.backend, Float64, length(v)); copyto!(d, v)
    return d
end

"""
    frame_flux(out, w, νs) -> Array (4, 3, NB)

Total Stokes flux densities [Jy] of a rendered frame: `w` holds each ray's solid angle in sorted order [sr]; the
second index is the part of the light: all of it, order 0, and orders ≤ 1 (so order 1 is the third minus the
second, and orders ≥ 2 the first minus the third).
"""
function frame_flux(out::AbstractVector{OrderState{NBs,T}}, w::AbstractVector, νs) where {NBs,T}
    F = zeros(T, 4, 3, NBs)
    for (j, acc) in enumerate(Array(out))
        for b in 1:NBs
            S = acc.st[b].S
            S0 = acc.n >= 1 ? acc.s0[b] : S
            S1 = acc.n >= 2 ? acc.s1[b] : S
            @views F[:, 1, b] .+= w[j] .* S; F[:, 2, b] .+= w[j] .* S0; F[:, 3, b] .+= w[j] .* S1
        end
    end
    for b in 1:NBs
        F[:, :, b] .*= νhat(νs[b])^3 / Tr.JY
    end
    return F
end

"""
    LookbackTable(cache; margin = 2)

The range of the lookback time t̃ over the samples of a regenerated stored cache that lie inside each radius of a
grid (log-spaced from the horizon to 10⁴ M), widened by `margin` M: light that leaves a point within radius r reaches
the screen between t_obs − t̃hi(r) and t_obs − t̃lo(r)... in reverse: a frame at t_obs sees r ≤ R only at emission
times in [t_obs − t̃hi(R), t_obs − t̃lo(R)].
"""
struct LookbackTable
    r::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end
function LookbackTable(cache::GeodesicCache; margin = 2.0)
    S = Geodesics.host(cache.samples)
    rh = 1 + sqrt(1 - cache.spin^2)
    rg = exp.(range(log(rh), log(1e4); length = 400))
    lo = fill(Inf, length(rg)); hi = fill(-Inf, length(rg))
    for j in axes(S.t, 1), k in axes(S.t, 2)
        f = Geodesics.unpack_flags(S.flags[j, k])
        (f.ok && S.r[j, k] > rh * (1 + 1e-3)) || continue
        q = searchsortedfirst(rg, S.r[j, k])
        q <= length(rg) || continue
        lo[q] = min(lo[q], S.t[j, k]); hi[q] = max(hi[q], S.t[j, k])
    end
    for q in 2:length(rg)                      # inside the radius: running extrema
        lo[q] = min(lo[q], lo[q-1]); hi[q] = max(hi[q], hi[q-1])
    end
    return LookbackTable(rg, lo .- margin, hi .+ margin)
end
"The lookback range of the samples inside radius `R` (the next tabulated radius above it)."
function (tab::LookbackTable)(R)
    q = min(searchsortedfirst(tab.r, R), length(tab.r))
    return tab.lo[q], tab.hi[q]
end

"""
    frame_parcels(P, t_obs, tab::LookbackTable; cutoff = WEIGHT_CUTOFF) -> (Pa, box, cols)

The parcels a frame at `t_obs` can see and their boxes, rows (te_lo, te_hi, r_lo, r_hi, z_lo, z_hi): the emission
times at which the envelope is above the cutoff and light from the parcel's reach can arrive at t_obs, and the radii
and heights the support sphere (radius √(2 ln(1/cutoff)) f σmax about the centre f x₀) covers over those times. The
distance from the centre bounds both |r − f|x₀|| and |z − f z₀|, so a sample outside the box is outside the support.
For a moving parcel the window and the reach are narrowed together from those of its whole envelope until they
agree, every step holding the true window.
"""
function frame_parcels(P::AbstractMatrix, t_obs, tab::LookbackTable; cutoff = WEIGHT_CUTOFF)
    K = sqrt(-2 * log(cutoff)) * (1 + 1e-9)
    cols = Int[]; boxes = Vector{Float64}[]
    for i in axes(P, 2)
        t0 = P[11, i]; r0 = sqrt(P[1, i]^2 + P[2, i]^2 + P[3, i]^2); z0 = P[3, i]; vr = P[22, i]
        σ = exp(max(P[4, i], P[5, i], P[6, i]))
        elo, ehi = t0 - K * exp(P[12, i]), t0 + K * exp(P[23, i])          # the envelope
        fof(t) = max(1 + vr * max(t - t0, 0) / r0, FMIN)
        f1, f2 = minmax(fof(elo), fof(ehi)); lo, hi = elo, ehi
        for _ in 1:20          # from the whole envelope inward: every window holds the true one, so each is conservative
            t̃lo, t̃hi = tab(max(f1, f2, 1.0) * (r0 + K * σ))
            lo = max(elo, t_obs - t̃hi); hi = min(ehi, t_obs - t̃lo)
            lo <= hi || break
            g1, g2 = minmax(fof(lo), fof(hi))
            (g1 == f1 && g2 == f2) && break
            f1, f2 = g1, g2
        end
        lo <= hi || continue
        R = K * max(f1, f2, 1.0) * σ
        push!(cols, i)
        push!(boxes, [lo - 1e-9 * abs(lo), hi + 1e-9 * abs(hi), f1 * r0 - R, f2 * r0 + R, min(f1 * z0, f2 * z0) - R, max(f1 * z0, f2 * z0) + R])
    end
    return P[:, cols], isempty(boxes) ? zeros(6, 0) : reduce(hcat, boxes), cols
end

# ---- the three emitting regions as stochastic populations of parcels
"""
    Population

A stochastic population of moving splats in one region of the spacetime: the parameter matrix `P` with every row but
the plasma rows (13–15) filled, the radius `r` at which each parcel's profiles are evaluated (its birth radius), the
scatter `lnA` of each parcel's density about the profile, the region's `kind` (`:disk`, `:funnel`, `:jet`) and its
plasma β (of the ions' virial pressure for the disk, of the electrons for the funnel and the jet). `plasma(pop, x)`
fills the plasma rows from the profile parameters x = (ln n₀ [cm⁻³], ln T₀ [K], sT, sn): nₑ = n₀ A r^sn,
Tₑ = T₀ r^sT, and the field from β.
"""
struct Population
    kind::Symbol
    P::Matrix{Float64}
    r::Vector{Float64}
    lnA::Vector{Float64}
    β::Float64
end
Base.length(pop::Population) = size(pop.P, 2)

function plasma(pop::Population, x)
    P = copy(pop.P)
    for i in axes(P, 2)
        r = pop.r[i]
        ne = exp(x[1] + x[4] * log(r) + pop.lnA[i]); Te = exp(x[2] + x[3] * log(r))
        B = pop.kind == :disk ? sqrt(8π * ne * Tr.MP * Tr.CL^2 / (pop.β * 12 * r)) :     # equipartition with the ions' virial pressure
                                sqrt(8π * ne * Tr.KBOL * Te / pop.β)                     # the electrons' β
        P[13, i] = log(ne); P[14, i] = log(Te * ΘUNIT); P[15, i] = log(B)
    end
    return P
end

loguniform(rng, lo, hi) = lo * (hi / lo)^rand(rng)

"Quaternion of R_z(φ) R_y(θ): it takes the splat's axes (x̂, ŷ, ẑ) to (θ̂, φ̂, r̂) at the polar angle θ and azimuth φ."
quaternion_spherical(θ, φ) = (cos(φ / 2) * cos(θ / 2), -sin(φ / 2) * sin(θ / 2), cos(φ / 2) * sin(θ / 2), sin(φ / 2) * cos(θ / 2))

# births at uniform times over `tspan` until the summed envelopes fill `nactive` parcels on average (∫ envelope dt =
# √(π/2)(w + wd) per parcel)
function births(rng, tspan, nactive, draw)
    cols = Vector{Float64}[]; rs = Float64[]; lnAs = Float64[]
    budget = nactive * (tspan[2] - tspan[1]); used = 0.0
    while used < budget
        col, r = draw(tspan[1] + (tspan[2] - tspan[1]) * rand(rng))
        push!(cols, col); push!(rs, r); push!(lnAs, 0.5 * randn(rng))
        used += sqrt(π / 2) * (exp(col[12]) + exp(col[23]))
    end
    return reduce(hcat, cols), rs, lnAs
end

"""
    disk_population(rng, tspan, nactive; a, β = 3, rin = 2, rout = 14)

The equatorial flow: parcels at radii log-uniform from `rin` to `rout` and uniform azimuth, centred within 0.1 r of
the midplane, Gaussians of scales (0.12, 0.3, 0.2) r × [0.7, 1.4] (radial, along the orbit, vertical) moving on
prograde Keplerian circular orbits (pattern and fluid), the field pitched 45° from vertical toward the orbit, living
for a symmetric envelope of width 0.25–1 orbital periods (log-uniform). Born at uniform times over `tspan`, as many
as keep `nactive` alive on average.
"""
function disk_population(rng, tspan, nactive; a, β = 3.0, rin = 2.0, rout = 14.0)
    P, r, lnA = births(rng, tspan, nactive, function (t0)
        r = loguniform(rng, rin, rout); φ = 2π * rand(rng); κ = loguniform(rng, 0.7, 1.4)
        w = 2π * (r^1.5 + a) * loguniform(rng, 0.25, 1.0)
        col = [r * cos(φ), r * sin(φ), 0.1 * r * randn(rng),
               log(0.12 * r * κ), log(0.3 * r * κ), log(0.2 * r * κ), cos(φ / 2), 0.0, 0.0, sin(φ / 2),
               t0, log(w), 0.0, 0.0, 0.0, π / 4, π / 2,
               0.0, keplerian_γβ(r, a), 0.0, 1 / (r^1.5 + a), 0.0, log(w)]
        return col, r
    end)
    return Population(:disk, P, r, lnA, β)
end

"""
    funnel_population(rng, tspan, nactive; a, β = 0.1, rin = 1.6, rout = 8, ϑ = (12°, 38°), ΩF = 0.35 Ω_H)

The magnetosphere above and below the hole: parcels at radii log-uniform from `rin` to `rout`, at angles from the
nearer pole uniform in `ϑ`, in either hemisphere, Gaussians of scales (0.12, 0.25, 0.2) r × [0.7, 1.4] (polar,
azimuthal, radial) turning rigidly with the field lines at `ΩF` (pattern and fluid), the field radial and wound back
by ϖΩF (reversed in the south, as a split monopole's), living 10–80 M (log-uniform, symmetric envelope).
"""
function funnel_population(rng, tspan, nactive; a, β = 0.1, rin = 1.6, rout = 8.0, ϑ = (deg2rad(12), deg2rad(38)),
                           ΩF = 0.35 * a / (2 * (1 + sqrt(1 - a^2))))
    P, r, lnA = births(rng, tspan, nactive, function (t0)
        r = loguniform(rng, rin, rout); φ = 2π * rand(rng); κ = loguniform(rng, 0.7, 1.4)
        north = rand(rng) < 0.5
        θ = (ϑ[1] + (ϑ[2] - ϑ[1]) * rand(rng)); θ = north ? θ : π - θ
        w = loguniform(rng, 10.0, 80.0)
        ϖ = r * sin(θ); phB = -atan(ϖ * ΩF) + (north ? 0.0 : π)
        col = [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ),
               log(0.12 * r * κ), log(0.25 * r * κ), log(0.2 * r * κ), quaternion_spherical(θ, φ)...,
               t0, log(w), 0.0, 0.0, 0.0, π / 2, phB,
               0.0, zamo_γβ_rotation(r, θ, ΩF, a), 0.0, ΩF, 0.0, log(w)]
        return col, r
    end)
    return Population(:funnel, P, r, lnA, β)
end

"""
    jet_population(rng, tspan, nactive; a, β = 1, rin = 4, rout = 25, ϑ = (8°, 22°), γβ = (0.5, 1.5))

The outflow: blobs born at radii log-uniform from `rin` to `rout` at angles from the nearer pole uniform in `ϑ`, in
either jet, Gaussians of scales (0.1, 0.2, 0.15) r × [0.7, 1.4] (polar, azimuthal, radial). Each brightens where it is
born over a rise of 3–15 M, then moves straight out at the coordinate speed of its radial ZAMO γβ (uniform in `γβ`),
expanding conically (density ∝ r⁻², field ∝ r⁻¹) and fading with a width of 0.4–1.2 times the time it takes to
cross its birth radius (log-uniform), so that it is gone by the time it has moved out a few birth radii. The field is helical, wound 60°
toward −φ̂ (reversed in the counter-jet); the electrons are in equipartition with it (β = 1).
"""
function jet_population(rng, tspan, nactive; a, β = 1.0, rin = 4.0, rout = 25.0, ϑ = (deg2rad(8), deg2rad(22)), γβ = (0.5, 1.5))
    met = Krang.Kerr(a)
    P, r, lnA = births(rng, tspan, nactive, function (t0)
        r = loguniform(rng, rin, rout); φ = 2π * rand(rng); κ = loguniform(rng, 0.7, 1.4)
        north = rand(rng) < 0.5
        θ = (ϑ[1] + (ϑ[2] - ϑ[1]) * rand(rng)); θ = north ? θ : π - θ
        u = γβ[1] + (γβ[2] - γβ[1]) * rand(rng)
        vr = KerrSplat.Splats.coordinate_velocity(met, r, θ, SVector(u, 0.0, 0.0))[1]
        col = [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ),
               log(0.1 * r * κ), log(0.2 * r * κ), log(0.15 * r * κ), quaternion_spherical(θ, φ)...,
               t0, log(loguniform(rng, 3.0, 15.0)), 0.0, 0.0, 0.0, π / 2, -π / 3 + (north ? 0.0 : π),
               u, 0.0, 0.0, 0.0, vr, log(r / vr * loguniform(rng, 0.4, 1.2))]
        return col, r
    end)
    return Population(:jet, P, r, lnA, β)
end

# ---- the three models
const REGIONS = (:disk, :funnel, :jet)
const GENERATORS = (disk = disk_population, funnel = funnel_population, jet = jet_population)
"Profile parameters (ln n₀, ln T₀, sT, sn) each region's tuning starts from."
const X_START = (disk = [log(4.6e7), log(2.8e11), -0.18, -2.6], funnel = [log(1e7), log(1e11), 0.0, -2.0], jet = [log(1e7), log(2e11), 0.0, -2.0])
"The longest envelope width each region's parcels can have (M), which sets how far before the first frame births must start."
const WMAX = (disk = 2π * (14.0^1.5 + 1.0), funnel = 80.0, jet = 25.0 / 0.3 * 1.2)   # jet: r/vr × 1.2 at the slowest blob

"""
    births_span(kind, t1, t2) -> (tlo, thi)

The birth times a population needs for frames from `t1` to `t2` to see a stationary process: four of its longest
envelope widths and the lookback within 200 M beyond each end.
"""
births_span(kind, t1, t2) = (t1 - 4 * WMAX[kind] - 250.0, t2 + 4 * WMAX[kind] + 250.0)

"""
    model_parcels(nactive, x, (t1, t2); a, seed) -> (P, region)

The parcels of a model: for each region with `nactive[region] > 0`, a population born over `births_span` keeping
that many alive on average (its own random stream, from `seed` and the region), with its plasma from the profile
parameters `x[region]`; their columns side by side, and the region of each column.
"""
function model_parcels(nactive::NamedTuple, x::NamedTuple, span; a, seed::Integer)
    Ps = Matrix{Float64}[]; region = Symbol[]
    for (k, kind) in enumerate(REGIONS)
        n = get(nactive, kind, 0.0)
        n > 0 || continue
        pop = GENERATORS[kind](Random.Xoshiro(seed * 10 + k), births_span(kind, span...), n; a)
        push!(Ps, plasma(pop, x[kind])); append!(region, fill(kind, length(pop)))
    end
    return reduce(hcat, Ps), region
end

# ---- the camera
"""
    critical_curve(a, θo; n = 4000) -> (αc, ψs, ρs)

The critical curve of spin `a` seen at inclination `θo`, as the polar radius ρ(ψ) about the midpoint αc of its
horizontal extent, sampled at `n` angles (Bardeen's parametrization by the radius of the bound photon orbit; Krang's
screen convention α = −λ/sin θo).
"""
function critical_curve(a, θo; n::Integer = 4000)
    rm = 2 * (1 + cos(2 / 3 * acos(-a))); rp = 2 * (1 + cos(2 / 3 * acos(a)))
    pts = Tuple{Float64,Float64}[]
    for x in range(0, 1; length = 20001)
        r = rm + (rp - rm) * (1 - cos(π * x)) / 2               # clustered at the ends, where β → 0
        Δ = r^2 - 2r + a^2
        λ = a + r / a * (r - 2Δ / (r - 1)); η = r^3 / a^2 * (4Δ / (r - 1)^2 - r)
        β2 = η + a^2 * cos(θo)^2 - λ^2 * cot(θo)^2
        β2 >= 0 || continue
        α = -λ / sin(θo); β = sqrt(β2)
        push!(pts, (α, β)); push!(pts, (α, -β))
    end
    αc = (minimum(first, pts) + maximum(first, pts)) / 2
    ψp = [atan(β, α - αc) for (α, β) in pts]; ρp = [hypot(α - αc, β) for (α, β) in pts]
    o = sortperm(ψp); ψp = ψp[o]; ρp = ρp[o]
    ψs = collect(range(-π, π; length = n + 1))[1:n]
    ψext = vcat(ψp[end] - 2π, ψp, ψp[1] + 2π); ρext = vcat(ρp[end], ρp, ρp[1])
    ρs = map(ψs) do ψ
        k = searchsortedlast(ψext, ψ)
        w = (ψ - ψext[k]) / (ψext[k+1] - ψext[k])
        (1 - w) * ρext[k] + w * ρext[k+1]
    end
    return αc, ψs, ρs
end

"""
    RingCamera

A camera on annuli about the centre of the critical curve, in the scaled radius s = ρ/ρc(ψ) (s = 1 is the critical
curve): annulus k spans s ∈ [edges[k], edges[k+1]] with `nψ[k]` equal cells in ψ, and each point sits at its cell's
centre with the cell's area (M²) as its weight.
"""
struct RingCamera
    camera::Geodesics.Camera{Float64,Vector{Float64}}
    area::Vector{Float64}
    s::Vector{Float64}
    ψ::Vector{Float64}
    αc::Float64
    edges::Vector{Float64}
    nψ::Vector{Int}
end

"""
    ring_camera(a, θo; ρmax, inner = 0.04, ring = (0.92, 1.12, 0.002), core = nothing, mid = 1.6, outer = 0.04, arc = 0.06, nψring = 720)

The annuli: uniform steps of `inner` (in s) from the centre to the ring band, steps of `ring[3]` across the band
s ∈ [ring[1], ring[2]] where the n ≥ 1 images lie, steps growing geometrically to `mid`, then steps of a fixed
fraction `outer` of s out to ρmax. With `core = (hmin, κ)` the steps across the band shrink toward the critical
curve as clamp(κ|s − 1|, hmin, ring[3]), which resolves the n ≥ 2 images (each thinner than the last by e^γ). Each
annulus off the band has as many cells in ψ as make them about square (ρ Δψ ≈ Δρ), within [16, `nψring`], so that
they stay near a size `arc` (M) or the radial step, whichever is larger; every annulus in the band has `nψring`.
"""
function ring_camera(a, θo; ρmax, inner = 0.04, ring = (0.92, 1.12, 0.002), core = nothing, mid = 1.6, outer = 0.04, arc = 0.06, nψring = 720)
    αc, ψc, ρc = critical_curve(a, θo)
    ρcmean = sum(ρc) / length(ρc)
    edges = collect(0.0:inner:ring[1])
    edges[end] < ring[1] && push!(edges, ring[1])
    if core === nothing
        append!(edges, (ring[1]+ring[3]):ring[3]:ring[2])
    else
        step(e) = clamp(core[2] * abs(e - 1), core[1], ring[3])
        up = [1.0]; while up[end] < ring[2] - 1e-12; push!(up, up[end] + step(up[end])); end
        dn = [1.0]; while dn[end] > ring[1] + 1e-12; push!(dn, dn[end] - step(dn[end])); end
        band = vcat(reverse(dn[2:end-1]), up[1:end-1])
        append!(edges, band[band .> ring[1] + 1e-12])
    end
    edges[end] < ring[2] - 1e-12 && push!(edges, ring[2])
    h = ring[3]
    while edges[end] < mid
        h = min(h * 1.15, outer * edges[end]); push!(edges, edges[end] + h)
    end
    smax = ρmax / minimum(ρc)
    while edges[end] < smax
        push!(edges, edges[end] * (1 + outer))
    end
    ρcfun(ψ) = (x = (ψ + π) / (2π) * length(ψc); k = floor(Int, x); w = x - k; (1 - w) * ρc[mod(k, length(ρc)) + 1] + w * ρc[mod(k + 1, length(ρc)) + 1])
    αs = Float64[]; βs = Float64[]; area = Float64[]; ss = Float64[]; ψs = Float64[]; nψ = Int[]
    for k in 1:length(edges)-1
        s1, s2 = edges[k], edges[k+1]; sm = (s1 + s2) / 2
        inband = s1 >= ring[1] - 1e-12 && s2 <= ring[2] + 1e-12
        cell = max(arc, (s2 - s1) * ρcmean)
        m = inband ? nψring : clamp(round(Int, 2π * sm * ρcmean / cell), 16, nψring)
        push!(nψ, m)
        for q in 1:m
            ψ = -π + (q - 0.5) * 2π / m
            ρ = ρcfun(ψ)
            push!(αs, αc + ρ * sm * cos(ψ)); push!(βs, ρ * sm * sin(ψ))
            push!(area, (s2^2 - s1^2) / 2 * ρ^2 * 2π / m)
            push!(ss, sm); push!(ψs, ψ)
        end
    end
    return RingCamera(Geodesics.Camera(αs, βs), area, ss, ψs, αc, edges, nψ)
end



# ---- the run: the spacetime, a day of Sgr A*, the cameras
const SPIN = 0.94
const INCLINATION = 30.0                                   # degrees
"A day at Sgr A* in GM/c³ (rounded up)."
const TMAX = ceil(86400 / TUNIT)
"Mean number of parcels alive in a population of each region alone, tuned to the whole target spectrum."
const NACTIVE = (disk = 40.0, funnel = 40.0, jet = 40.0)
const TUNING_SEED = 20261029
"""
The density weight (relative to a parcel's peak at birth) below which a parcel is left out of a sample: 1e-4, where
KerrSplat uses 1e-6. The parcels are then Gaussians truncated at 4.3σ; the flux changes by 0.2–0.4% (absorbed by the
tuning) and a frame costs half as much (see the README).
"""
const CUTOFF = 1e-4
"""
The production camera: 26,714 rays on annuli about the critical curve, the band s ∈ [0.93, 1.55] (where the order-1
light of the disk and nearly all of the funnel's lies) in steps shrinking from 0.006 to 0.001 in s at the critical
curve, 120 cells in ψ, out to 80 M; 120 samples per ray. Against a reference of 1.08 million rays (steps of 0.0005,
720 cells, 240 samples) its total, order-1 and order-≥2 flux densities agree to 0.1%, 0.1% (0.4% for the jet's,
which is 1% of its light) and the four digits printed (see the README).
"""
const PRODUCTION_CAMERA = (ρmax = 80.0, inner = 0.04, ring = (0.93, 1.55, 0.006), core = (0.001, 0.1), outer = 0.04, arc = 0.1, nψring = 120)
const PRODUCTION_CAMERA_SAMPLES = 120
"A coarser camera for the tuning, which needs the spectrum (to a percent), not the ring."
const TUNING_CAMERA = (ρmax = 80.0, inner = 0.06, ring = (0.93, 1.55, 0.012), outer = 0.06, arc = 0.2, nψring = 60)
const TUNING_CAMERA_SAMPLES = 100

"""
    camera_setup(backend, rc::RingCamera, a, θo, Val(N)) -> (cache, tab, w, τhalf)

A fused geodesic cache of the ring camera with `N` samples per ray, the lookback table of its rays (from a stored
cache of the same rays, margin 5 M), each ray's solid angle [sr] and each ray's τ½, in the cache's sorted order.
"""
function camera_setup(backend, rc::RingCamera, a, θo, ::Val{N}) where {N}
    sc = GeodesicCache(KA.CPU(), rc.camera, Val(N); store_samples = true)
    regenerate!(sc, a, θo; marcher = Recurrence(64))
    tab = LookbackTable(sc; margin = 5.0)
    sc = nothing; GC.gc()
    cache = GeodesicCache(backend, rc.camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    return cache, tab, rc.area[cache.perm_host] .* (LUNIT / D_CM)^2, half_orbit_times(cache)
end

end
