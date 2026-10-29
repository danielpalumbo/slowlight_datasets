# slowlight_datasets

Slow-light light curves of black-hole accretion models, rendered with
[KerrSplat](https://github.com/danielpalumbo/kerrsplat): Gaussian splats of thermal plasma in the Kerr
spacetime, ray traced along Krang's analytic geodesics with full Stokes synchrotron transfer. Every sample of every
ray sees the plasma at its own emission time t_obs − t̃ (t̃ the photon's lookback time to that sample), so orbiting
material appears where it was when the light left it, and lensed sub-images arrive late. The light curves here are
those finite-light-speed (slow-light) movies integrated over the screen.

## Datasets

| directory | model | bands | script |
|---|---|---|---|
| `datasets/test_sgra_disk20/` | 20-splat Keplerian disk, inner 20 M of Sgr A*, tuned to its mean spectrum | ALMA 3–9 | `scripts/test_sgra_disk20.jl` |
| `datasets/sgra_disk_dominated/` | a day of Sgr A*, most of the light from the equatorial disk | ALMA 3–9 | `scripts/sgra_regions.jl --model disk` |
| `datasets/sgra_funnel_dominated/` | a day of Sgr A*, most of the light from the funnel above and below the hole | ALMA 3–9 | `scripts/sgra_regions.jl --model funnel` |
| `datasets/sgra_jet_dominated/` | a day of Sgr A*, most of the light from the jet and counter-jet | ALMA 3–9 | `scripts/sgra_regions.jl --model jet` |

Each dataset directory holds:

- `lightcurves.csv`: one row per (band, time): `band, nu_GHz, t_M, t_s, I_Jy, Q_Jy, U_Jy, V_Jy`. Time is the
  observer's, in GM/c³ and in seconds; the Stokes parameters are the total flux densities, integrated over the
  screen (interstellar scattering, which blurs Sgr A*'s image, conserves them), with KerrSplat's conventions (ipole's
  Stokes basis, screen north = +β, east = −α).
- `lightcurves_by_order.csv` (the region datasets): the same rows split by lensing order, `I_n0_Jy … V_n0_Jy`,
  `I_n1_Jy … V_n1_Jy`, `I_n2plus_Jy … V_n2plus_Jy`, the light that reached the screen directly, after one
  half-orbit about the hole, and after more (see [lensing orders](#lensing-orders)); the three sum to
  `lightcurves.csv`. This is the truth for a light-echo search: hold it back from the light curves if it is a blind
  test.
- `spectrum.csv`: per band, the target flux density and its fractional weight, the light curve's mean, standard
  deviation and range; for the test dataset the spectrum of its tuning snapshot, for the region datasets the share
  of the day's light in orders 1 and ≥ 2 and the share each region gives.
- `splats.csv` (the test dataset): the model, one row per splat, one column per KerrSplat polarized-splat parameter
  (`POLARIZED_SPLAT_PARAMS`: centre, log scales, orientation quaternion, t₀, log envelope, ln nₑ [cm⁻³], ln Θₑ,
  ln B [G], field angles, ZAMO γβ, pattern rate [rad/M]). Its transpose is the parameter matrix of
  `KerrSplat.Splats.polarized_image`, which with the camera, spin and inclination of `metadata.toml` reproduces any
  frame.
- `parcels.csv` (the region datasets): every parcel of the day, its region and its 23 parameters
  (`MOVING_SPLAT_PARAMS` of `scripts/SlowLight.jl`: KerrSplat's 21, then the radial speed `vr` and the ln width of
  the envelope after its peak `logwd`).
- `metadata.toml`: the source (mass, distance, GM/c² and GM/c³), spin and inclination, camera, time grid, tuned
  normalizations, and the KerrSplat commit that rendered it.
- `overview.png`: images of the model, its spectrum against the target, and the light curves.

## Setup

Julia 1.10. The environment pins KerrSplat to a commit on GitHub and Krang.jl to commit `f36f43a` (KerrSplat needs
Krang's `main`, not the registered release); `Manifest.toml` carries the versions KerrSplat is tested with.

    JULIA_PKG_USE_CLI_GIT=true julia --project=. -e 'import Pkg; Pkg.instantiate()'

`LocalPreferences.toml` pins the CUDA 12.8 runtime (needed on GeForce cards with driver 570; change or delete it on
other machines). The scripts use a CUDA GPU when one is functional and fall back to the CPU backend of
KernelAbstractions otherwise (`--cpu` forces it); run with threads on the CPU (`julia -t auto`).

## The test dataset: a 20-splat disk for Sgr A*

    julia -t auto --project=. scripts/test_sgra_disk20.jl

Sgr A* at M = 4.152 × 10⁶ M☉ and D = 8.178 kpc (GRAVITY Collaboration 2019), so M = GM/c³ = 20.5 s and 1 M on the
sky is 5.0 μas; spin 0.94 and inclination 30° (the EHT's favoured region, EHT Collaboration 2022 Paper V).

**The disk.** Twenty one-zone thermal parcels on a golden-angle spiral, their centres log-uniform in radius from 2 M
to 16 M, each a Gaussian of scales 0.25 r (radial), 0.5 r (along the orbit) and 0.3 r (vertical). Each carries
power-law profiles at its centre, in the manner of the semi-analytic RIAF of Broderick & Loeb (2006): peak density
n₀ r^sn, electron temperature T₀ r^sT, and a field in equipartition with the ions at plasma β
(B²/8π = n mₚc²/(12 β r)), pitched 45° between toroidal and vertical. Every parcel moves on a prograde Keplerian
circular orbit: its pattern rotates at Ω = 1/(r^{3/2} + a) and its fluid has the matching ZAMO velocity, so the
emission pattern and the Doppler factor move together. The parcels do not shear, flare or fade; the variability is
their orbital motion seen through beaming, lensing and light-travel delays.

**The tuning.** The four profile parameters (n₀, T₀, sT, sn) are fitted by Levenberg–Marquardt to the logarithm of the
total flux density at t = 0 at ALMA's standard continuum frequencies for bands 3–9, with β held at 3 (Bower et al.
2019 find β ≃ 1–10 once the X-ray limit is imposed). Two simpler tunings were tried and rejected. With Broderick &
Loeb's slopes (sn = −1.1, sT = −0.84) and (n₀, T₀, β) free, β runs off to 10⁻⁶: the spectrum alone cannot tell a
dense, weakly magnetized plasma from a tenuous one in kilogauss fields. With those slopes and β held anywhere in
1–30, the disk is too cool inside to reach band 9 (1.1–1.2 Jy against 1.8, χ² 6–10). Freeing the slopes instead
gives a nearly isothermal disk, Θₑ ≈ 30–40, with the density falling steeply (n ∝ r^−2.6). That puts nₑ, Tₑ and B
at a few M where Bower et al.'s one-zone fit puts them (nₑ ≈ 2–5 × 10⁶ cm⁻³, Tₑ ≈ 1–3 × 10¹¹ K, B ≈ 10–50 G) and
makes the submillimetre emission compact. The target spectrum:

| band | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| ν [GHz] | 97.5 | 145 | 203 | 233 | 343.5 | 405 | 679 |
| target [Jy] | 2.00 | 2.17 | 2.33 | 2.40 | 2.17 | 2.08 | 1.82 |
| weight | 10% | 10% | 10% | 10% | 10% | 15% | 20% |

The anchors are all from 2017: 2.0 ± 0.2 Jy at 98 GHz and 2.4 ± 0.2 Jy at 213–229 GHz during the EHT campaign
(EHT Collaboration 2022, Paper II, Table 2), and the simultaneous 233 → 678 GHz spectral index α = −0.26 ± 0.02 of
Bower et al. (2019, ApJL 881, L2). Between them a power law (α = +0.21 below 233 GHz, −0.26 above). Sgr A*'s
submillimetre spectrum is variable and its peak frequency is debated (the historical "submillimetre bump" near
3 Jy at 345 GHz, e.g. Subroweit et al. 2017, against the falling spectrum ALMA measured in 2017), so the target is a
rough mean; edit `target_flux` and `FRACERR` in the script to change it.

**The result** (`datasets/test_sgra_disk20/`, 128² pixels over 48 M, 200 samples per ray, 0–500 M every 2 M, i.e.
2.8 hours every 41 s). The tuned profiles are n₀ = 4.6 × 10⁷ cm⁻³ with sn = −2.60 and T₀ = 2.8 × 10¹¹ K with
sT = −0.18. That gives nₑ = 7.6 × 10⁶ cm⁻³, Tₑ = 2.5 × 10¹¹ K and B = 63 G at 2 M, and nₑ = 7.0 × 10⁵ cm⁻³,
Tₑ = 2.1 × 10¹¹ K and B = 12 G at 5 M. The snapshot it was tuned on matches the target within 4% in every band
(χ² = 0.29 for seven bands). The light curves' means sit within 6% of the target (1.89, 2.16, 2.26, 2.26, 2.17, 2.09
and 1.75 Jy in bands 3–9), and their rms variability grows with frequency, from 2.4–2.8% in bands 3–6 to 3.6%, 4.1%
and 5.6% in bands 7–9, as the inner orbits take over the emission. The outer 4 M of the field hold 0.25% of the
band-3 flux.

**Run time.** 25 minutes on a 32-thread CPU: 1.5 minutes of tuning, then 1757 frames at 0.78 s each. The script
starts the tuning from this dataset's parameters; `--no-tune` skips it and renders exactly this model. A quick look at reduced resolution
takes about two minutes:

    julia -t auto --project=. scripts/test_sgra_disk20.jl --res 64 --tmax 100 --dt 4 --out scratch/quick

## The region datasets: a day of Sgr A* from the disk, the funnel or the jet

    julia -t auto --project=. scripts/tune_regions.jl          # the three regions' plasma, against the target spectrum
    scripts/run_regions.sh                                       # the three days and their figures

Three realizations of Sgr A* (the source, spin 0.94 and inclination 30° of the test dataset) in each of which one
region of the spacetime gives most of the light, each a day long (0 to 4224 M, 24.0 h, every 1 M = 20.5 s) in full
Stokes on ALMA bands 3–9. They are meant as test data for light-echo searches: the light that has gone half an orbit
around the hole (the n = 1 ring) arrives later than the direct light by a lapse that depends on where it was emitted,
and the datasets carry that light separately as truth.

**The regions.** Each region is a stochastic population of short-lived thermal parcels (`scripts/SlowLight.jl`):

- *disk*: the equatorial flow, parcels at radii log-uniform from 2 to 14 M within 0.1 r of the midplane, Gaussians of
  scales (0.12, 0.3, 0.2) r (radial, along the orbit, vertical; each scaled by a factor log-uniform in [0.7, 1.4]) on
  prograde Keplerian orbits, the field pitched 45° between vertical and toroidal and in equipartition with the ions'
  virial pressure at β = 3 (as in the test dataset), each living for 0.25–1 orbital periods;
- *funnel*: the magnetosphere above and below the hole, parcels at radii log-uniform from 1.6 to 8 M and 12°–38°
  from either pole, scales (0.12, 0.25, 0.2) r (polar, azimuthal, radial), turning rigidly with the field lines at
  Ω_F = 0.35 Ω_H (pattern and fluid), the field radial and wound back by ϖΩ_F (reversed in the south, as a split
  monopole's), the electrons at β_e = 0.1, each living 10–80 M;
- *jet*: blobs born at radii log-uniform from 4 to 25 M, 8°–22° from either pole, scales (0.1, 0.2, 0.15) r, which
  brighten where they are born over 3–15 M and then move straight out at the coordinate speed of a radial γβ uniform
  in [0.5, 1.5], expanding conically (scales ∝ r, density ∝ r⁻², field ∝ r⁻¹) and fading over 0.4–1.2 times the time
  it takes them to cross their birth radius; the field helical (wound 60° from radial), the electrons in
  equipartition with it (β_e = 1).

Parcels are born at uniform times through the day (and four of their longest lifetimes before and after it), as
many as keep a set number alive on average; each has the profiles of its region at its birth radius, nₑ = n₀ A r^sn
and Tₑ = T₀ r^sT, with a scatter A log-normal of width 0.5. The jet's moving blobs are not KerrSplat splats: they
are KerrSplat's polarized splat with two more rows (a radial speed and a separate width for the envelope after its
peak), rendered by the consumer in `scripts/SlowLight.jl`, which reduces to KerrSplat's splat at rest (checked to
the bit in `test/test_slowlight.jl`).

**The mixture.** A population of each region alone, keeping 40 parcels alive, is tuned to the whole target spectrum
of the test dataset (`scripts/tune_regions.jl`: Levenberg–Marquardt on (ln n₀, ln T₀, sT, sn) against the log mean
flux densities of 16 snapshots spread through a day, weak priors of width 1.5 holding the slopes near their starting
values). A model then keeps 32 parcels of its dominant region alive and 4 of each of the others, so that on average
the dominant region gives 80% of the light and the others 10% each; the parcels of every region have the same
plasma in all three models (the minor regions are rarer, not fainter), and each model is its own random draw.

| region | n₀ [cm⁻³] | sn | T₀ [K] | sT | χ² (7 bands + 2 priors) | snapshot rms (B3 / B6 / B9) |
|---|---|---|---|---|---|---|
| disk | 1.2 × 10⁸ | −2.82 | 2.9 × 10¹¹ | −0.26 | 0.31 | 21 / 28 / 35% |
| funnel | 1.0 × 10⁶ | −0.71 | 1.3 × 10¹¹ | +0.08 | 4.32 | 24 / 28 / 31% |
| jet | 5.5 × 10⁷ | −2.40 | 3.2 × 10¹¹ | −0.37 | 0.41 | 24 / 35 / 45% |

The funnel cannot quite take the target's shape: alone it is 15% low at band 3 and 7–8% high at bands 7 and 8.

<a id="lensing-orders"></a>**Lensing orders.** Each sample of a ray at Mino time τ from the observer has order
⌊τ/τ½⌋, τ½ being the Mino time of one half-oscillation of the ray in θ, 2K(u₊/u₋)/(a√−u₋) (u± the roots in cos²θ of
the polar potential): the number of half-orbits about the hole the light has made since it left the sample. A nearly
straight ray covers exactly one half-oscillation from infinity to infinity, so all of its light is order 0 wherever
it comes from; the k-th equatorial crossing of any ray falls between (k−1)τ½ and kτ½, so for an equatorial source
order n is the n-th image of Gralla & Lupsasca (2020), and the boundaries lie far from the midplane however thick
the disk. The order-n light is that of the ray truncated after order n less that truncated after n − 1, emission
and absorption both, as KerrSplat's `nmax` (for a thin disk the two agree to 1e-16 in `test/test_slowlight.jl`).
Order 1 is mostly the n = 1 ring, between s = ρ/ρ_c = 0.93 and 1.55 (ρ_c the critical curve's radius, 4.84–4.98 M
about its centre at α = 1.10 M): all but 0.03–0.3% of the disk's order-1 light at band 6 lies there, and 93–94% of
the funnel's (the rest, down to s ≈ 0.83, is near-horizon funnel light seen against the shadow). For the jet it is
57–77%: the rest is the weakly lensed secondary images of distant material behind the hole, out to s ≈ 2.3–3 (for
an equatorial source these would be Gralla & Lupsasca's n = 1 images too), a small part of a small order-1 share.

**The camera.** A uniform grid fine enough for the n = 1 ring over the jet's field would need millions of rays, so
the camera is a set of annuli about the critical curve in the scaled radius s, each ray carrying its cell's area:
steps of 0.04 in s inside s = 0.93, a band to s = 1.55 in steps shrinking from 0.006 to 0.001 at the critical curve
(the n ≥ 2 images thin by e^γ each), 120 cells in azimuth there, then steps growing to 4% of s out to 80 M; 26,714
rays and 120 samples per ray. Its band-6 flux densities for a population of each region at two times, against a
reference of 1.08 million rays (steps of 0.0005 in s across the band, 720 cells, 240 samples per ray):

| region (share of the light in order 1, orders ≥ 2) | total | order 1 | orders ≥ 2 |
|---|---|---|---|
| disk (5–6%, 0.6%) | < 0.05% | < 0.03% | within 10⁻⁴ Jy (2%) |
| funnel (24–26%, 0.5–0.8%) | < 0.09% | < 0.09% | within 10⁻⁴ Jy (0.6%) |
| jet (1.1–1.4%, 0.1%) | < 0.04% | < 0.5% | within 10⁻⁴ Jy (15%) |

Along the way (`test/camera/`, at bands 3, 6 and 9): steps of 0.002 in s with 720 cells give the same order-1 light
to 0.2%; uniform steps of 0.004–0.006 get order 1 to 0.2% but miss the n ≥ 2 rings by up to 80%; 120 samples per ray
agree with 240 to 0.1% in the total and 0.2% in order 1. The parcels are truncated where their density falls below
10⁻⁴ of its peak, where KerrSplat uses 10⁻⁶: that halves the cost, and moves the flux by 0.1–0.4% (absorbed by the
tuning).

**The datasets** (rendered on a 32-thread CPU, 12.3 hours in all; 4225 frames each):

| model | shares disk / funnel / jet (B6) | mean / target (B3 / B6 / B9) | rms (B3 / B6 / B9) | order 1 (B3 / B6 / B9) | orders ≥ 2 (B6) | parcels | time |
|---|---|---|---|---|---|---|---|
| disk-dominated | 74 / 19 / 8% | 0.92 / 0.83 / 0.79 | 21 / 27 / 35% | 6.0 / 8.2 / 9.0% | 0.7% | 2579 | 3.5 h |
| funnel-dominated | 11 / 78 / 12% | 0.93 / 0.97 / 0.98 | 23 / 28 / 32% | 13.6 / 19.1 / 20.6% | 0.9% | 2970 | 5.2 h |
| jet-dominated | 11 / 9 / 81% | 1.05 / 0.91 / 0.83 | 27 / 37 / 51% | 4.4 / 5.0 / 4.9% | 0.3% | 6596 | 3.5 h |

A region's share is its flux rendered alone over the whole model's, averaged over 16 times through the day; the
three sum to 1.01–1.10 rather than 1 because the regions absorb some of one another's light (most at band 3). Each
model is one random draw, and with only 4 parcels of each minor region alive the shares land away from the nominal
80/10/10 (the disk model's funnel gives 19%), and the means away from the tuning's: the disk model sits 8–21% below
the target (beyond the 10% weights from band 5 up), the funnel model within 7% of it, the jet model from 5% above to
17% below. The variability, 21–51% rms and growing with frequency, is well above Sgr A*'s typical 10–20% at 230 GHz:
forty short-lived parcels make a flaring source.

The order-1 light is not the dominant region's alone. By the order-1 shares of each region alone (the camera table
above), about half of the disk model's order-1 light is the funnel's and three quarters of the jet model's is the
funnel's and the disk's; the funnel model's is nearly all the funnel's. At band 6 the order-1 light curve correlates
with the direct (order-0) light at r = 0.44 in the disk model and 0.66 in the funnel model, peaking 5–6 M
(2 minutes) later, and hardly at all in the jet model (r = 0.09).

The seeds are fixed (101, 202 and 303 for the three models, `TUNING_SEED` for the tuning), so
`scripts/run_regions.sh` redraws the same parcels; a model interrupted resumes from its checkpoint in `scratch/`.

**Checks.** `julia -t auto --project=. test/test_slowlight.jl`: the moving splats at rest are KerrSplat's splats to
the bit; the seven-band consumer equals KerrSplat's `RadiativeTransport` band by band to 1e-15; the coefficients with
precomputed Bessel functions are KerrSplat's to the bit; the per-frame parcel boxes cull nothing that would count;
the order split of a thin disk equals KerrSplat's `nmax` truncation; τ½ matches the spacing of the marched rays'
polar turning points.

## License

Apache License 2.0 (`LICENSE`), as KerrSplat.
