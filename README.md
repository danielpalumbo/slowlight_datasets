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

Each dataset directory holds:

- `lightcurves.csv`: one row per (band, time): `band, nu_GHz, t_M, t_s, I_Jy, Q_Jy, U_Jy, V_Jy`. Time is the
  observer's, in GM/c³ and in seconds; the Stokes parameters are the total flux densities, integrated over the
  screen (interstellar scattering, which blurs Sgr A*'s image, conserves them), with KerrSplat's conventions (ipole's
  Stokes basis, screen north = +β, east = −α).
- `spectrum.csv`: per band, the target flux density and its fractional weight, the light curve's mean, standard
  deviation and range, and the spectrum of the tuning snapshot.
- `splats.csv`: the model, one row per splat, one column per KerrSplat polarized-splat parameter
  (`POLARIZED_SPLAT_PARAMS`: centre, log scales, orientation quaternion, t₀, log envelope, ln nₑ [cm⁻³], ln Θₑ,
  ln B [G], field angles, ZAMO γβ, pattern rate [rad/M]). Its transpose is the parameter matrix of
  `KerrSplat.Splats.polarized_image`, which with the camera, spin and inclination of `metadata.toml` reproduces any
  frame.
- `metadata.toml`: the source (mass, distance, GM/c² and GM/c³), spin and inclination, camera, time grid, tuned
  normalizations, and the KerrSplat commit that rendered it.
- `overview.png`: the model from above, a band-6 image, the spectrum against its target, and the light curves.

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

## License

Apache License 2.0 (`LICENSE`), as KerrSplat.
