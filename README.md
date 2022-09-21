<p align="center">
<img src="https://user-images.githubusercontent.com/16495490/170366679-17b0711e-4a74-41ee-82df-30b61e82b211.gif" width="600">
</p>


# <img src="https://user-images.githubusercontent.com/81261009/191581307-54e75f6f-6a18-40da-bc73-d71893398ee0.png" width="60"> WaveSpin: Scattering Discriminative Invariants

Joint Time-Frequency Scattering, Wavelet Scattering: features for classification, regression, and synthesis of audio, biomedical, and other signals. Friendly overviews:

 - [Wavelet Scattering](https://dsp.stackexchange.com/a/78513/50076)
 - [Joint Time-Frequency Scattering](https://dsp.stackexchange.com/a/78623/50076)

## Features

 - Joint Time-Frequency Scattering: the most accurate, flexible, and fast implementation
 - Wavelet (time) Scattering: the fastest, most accurate and parameter-efficient implementation
 - Differentiability, GPU acceleration: NumPy, PyTorch, and TensorFlow support
 - Visualizations, introspection and debug tools

## Installation

`pip install wavespin`. Or, for latest version (most likely stable):

`pip install git+https://github.com/OverLordGoldDragon/wavespin`

## Why scattering?

 - **Time-shift invariance**: a delayed word is the same word, a shifted cat is still a cat. Or, we don't care about words, only whole sentences: then words and letters are noise.
 - **Time-warp stability**: want a translator to work with people who may speak at different speeds, or a system to recognize digits with different handwritings.
 - **Information preservation**: unlike MFCC and alike, scattering recovers information lost to averaging, building invariants while preserving discriminability.

## Why JTFS?

### FDTS-sensitivity

(time) Scattering can't tell apart a chirp from an impulse, since it's insensitive to _frequency-dependent time shifts_, or more generally, _relative frequency shifts_. This discards a wide variety of informative time-frequency behavior - that JTFS preserves:

<img src="https://i.stack.imgur.com/HOdap.gif" width="600">

This need not be an "explicit" FDTS phenomenon; anything whose frequency rises or falls over time is considered "FDTS" to JTFS, i.e. any curvature in time-frequency. Ordinarily, rises and falls overlap under time averaging.

### Frequency transposition invariance

Same as time-shift invariance, but along frequency. Useful in musical instrument classification, retrieving timbre while ignoring changes in pitch, intensity, and performer's expressive technique. Optional.

<img src="https://user-images.githubusercontent.com/16495490/135682253-d10b74a8-4384-4eb8-8c7b-f363bee9b419.gif" width="580">

## Examples

### 1. Chirp-sine-pulse | JTFS, 2D viz

<img src="https://user-images.githubusercontent.com/16495490/163857080-9ae52cad-9202-4fb8-a1f5-a7d008f19073.png" alt="signal" width="800">
<img src="https://user-images.githubusercontent.com/16495490/163851994-b35772b0-5f73-4eef-8417-26ad02bbb65c.png" alt="scale-rate" width="750">

The signal produces vertical, horizontal, and diagonal time-frequency geometries, all intersecting at one point, making it impossible to untangle with traditional methods. JTFS's pairs,

 - **spin up** ("phase countour" down): captures rise down, minimal here (straying from its ideal of zero due to imperfections in said geometries)
 - **spin down**: captures rise up
 - **temporal lowpass** (w/ frequential bandpass): captures horizontal geometry
 - **frequential lowpass** (w/ temporal bandpass): captures vertical geometry
 - **joint lowpass**: geometry-agnostic averaging of the entire scalogram

There's almost no overlap in energies of any one geometry with another. This demonstrates sparsity (selective activation) of JTFS, and how it enables some highly nonlinear filtering.

### 2. Trumpet | JTFS, 4D viz

<img src="https://raw.githubusercontent.com/OverLordGoldDragon/StackExchangeAnswers/main/SignalProcessing/Q78644%20-%20Joint%20Time-Frequency%20Scattering%20explanation/jtfs3d_trumpet.gif" width="600">

Exhibits "spin assymetry", where spin up (`xi1_fr > 0`) has more energy than spin down (`xi1_fr < 0`), and vice versa, at different time instances.

### 3. Reconstruction, gradient-based

Inverting scalogram of exponential chirp:

<img src="https://user-images.githubusercontent.com/16495490/133305417-621b1353-9f92-48e6-8691-baec84649b7b.gif" width="710">

### 4. FDTS localization, -1.23 dB SNR

Discriminability and localization persists through severe noise and moderate averaging (6.3% time, 7.1% freq). Greater averaging compromises localization but not discriminability.

<img src="https://user-images.githubusercontent.com/16495490/184383354-b3a0657f-8c04-4099-ab95-1063f7a83930.gif" width="500">

### 5. Increased frequency resolution, Time Scattering

<img src="https://i.stack.imgur.com/KjmYK.png" width="500">

<sup><i>Fig 6, [Deep Scattering Spectrum](https://arxiv.org/abs/1304.6763)</i></sup>

 - First plot is scalogram, second is first-order scattering, third is second-order.
 - Left signal at $\xi_1$ is a $600 \text{ Hz}$ and $675 \text{ Hz}$ chord (played simultaneously), right same frequencies but appregio (played in quick succession)
 - Scalogram shows the chords blended together, and $S_1$ makes chords and appregios look the same per time averaging
 - $S_2$ resolves the blended frequencies

## Visualizations

### Filterbank, 1D, freq-domain

<img src="https://user-images.githubusercontent.com/16495490/170584651-ed471194-f129-4ea6-83e4-4b4e7de6b5e5.png" width="650">

### Filterbank, 2D heatmap, time-domain

<img src="https://user-images.githubusercontent.com/16495490/170584832-2b01c27d-2df8-40c8-9982-18c090cc9876.png" width="700">

### Coefficient energies

"Chirp-sine-pulse" example

<img src="https://user-images.githubusercontent.com/16495490/170585113-300fc6c3-696c-45f2-9acc-ec473ce46bad.png" width="700">

### JTFS pairs, 4D

<img src="https://user-images.githubusercontent.com/16495490/170585443-2c8d11f6-f999-458d-b2eb-e0409a83aa14.gif" width="600">

### Other

For an extended demo, see `examples/visuals_tour.py`, also [Visual Articles](https://github.com/OverLordGoldDragon/wavespin#tutorials--visual-articles).

## Toolkit

 - **`pack_coeffs_jtfs`**: enables 3D and 4D convolutions, on top of the existing 1D and 2D.
 - **`normalize`**: standardizes data for ML pipelines intelligently, using `sparse_mean` to prioritize coefficient geometry over raw energy.
 - **`est_energy_conservation`**, **`coeff_distance`**, ...: transform and coefficient introspection utilities.
 - **`validate_filterbank`**: automated filterbank validation, see article. 

`validate_filterbank` sample output:

```
== REDUNDANCY ======================================================================
Found filters with duplicate peak frequencies! Showing up to 20 filters:
psi_fs[63], peak_idx=8
psi_fs[64], peak_idx=8
psi_fs[66], peak_idx=6 ...

== DECAY (boundary effects) ========================================================
Found filter(s) with incomplete decay (will incur boundary effects), with 
following ratios of amplitude max to edge (less is worse; threshold is 1000.0):
psi_fs[52]: 580.6
psi_fs[53]: 107.7
psi_fs[54]: 51.9 ...

== ALIASING ========================================================================
Found Fourier peaks that are spaced neither exponentially nor linearly, 
suggesting possible aliasing.
psi_fs[n], n=[20 18 17]

== FREQUENCY-BANDWIDTH TILING ======================================================
Found non-CQT wavelets in upper quarters of frequencies - i.e., 
`(center freq) / bandwidth` isn't constant: 
psi_fs[32], Q=1.9328358208955223
psi_fs[34], Q=2.2686567164179103
psi_fs[35], Q=2.462686567164179 ...
```

## Tutorials / Visual Articles

 1. Wavelet scattering: [part 1](https://dsp.stackexchange.com/a/78513/50076) -- [part 2](https://dsp.stackexchange.com/a/78515/50076)
 2. JTFS: [part 1](https://dsp.stackexchange.com/a/78623/50076) -- [part 2](https://dsp.stackexchange.com/a/78625/50076)
 3. Why's it called "spin"?
 4. Validating a wavelet filterbank

## Q & A

### 1. Are the speedups at expense of accuracy?
 
No loss of precision is involved. Main speedups are from vectorization, Smart Paths, and intelligent padding and striding (JTFS). To contrary, accuracy is _increased_ by excluding coefficients correctly, whereas other implementations may lose information or (with normalization) amplify noise.

### 2. Why are WaveSpin's implementations "the most accurate"?

WaveSpin's design is _discrete-first_, and _information-oriented_. Continuous-time-only approaches are inherently limited as they're premised on infinite information. The following are largely unique to WaveSpin:

 1. **Padding & filter decay**: convolution pad amounts and filter sampling lengths are always sufficient to minimize boundary effects and spectrotemporal distortion. Users can choose to prioritize compute speed instead.
 2. **Optimal frequential tiling**: wavelets are parametrized such that complete _and sufficient_ tiling is achieved, in that no frequency is significantly attenuated relative to others.
 3. **Energy conservation**: transforms are designed and tested for energy conservation and non-expansiveness. Filterbanks are normalized for optimal energy overlap (Littlewood-Paley sum), and unpad aliasing and stride are accounted for. This provides guarantees on information loss.
 4. **Strict analyticity**: without it, Morlet's pseudo-analyticity extracts A.M. and F.M. where none exist, notably for very high and low frequencies. This is undesired for applications with said frequencies (e.g. seizure EEG), and bad news for JTFS's FDTS-discriminability for all frequencies.
 5. **Promises delivered**: aliasing error bounds, sampling/padding sufficiency, and other guarantees require accurate measures of bandwidth, spatial support, and so on; only discrete measures are universally applicable, and are more accurate and reliable than e.g. discretized integration.
 6. **Documentation**: at expense of more reading, docs don't oversimplify, are detailed, and reference material explaining and justifying theoretical and implemented apsects.

### 3. Planned support for 2D, 3D scattering?
 
No.
 
### 4. Planned support for other wavelets?
 
[Generalized Morse Wavelets](https://overlordgolddragon.github.io/generalized-morse-wavelets/) are superior to Morlets in every way for scattering, but the difference isn't that great so it's low priority. There's no plans to support arbitrary wavelets.

## How to cite

Short form:

> John Muradeli, WaveSpin, 2022. GitHub repository, https://github.com/OverLordGoldDragon/wavespin/. DOI: 10.5281/zenodo.5080508

BibTeX:

```bibtex
@article{OverLordGoldDragon2022wavespin,
  title={WaveSpin},
  author={John Muradeli},
  journal={GitHub. Note: https://github.com/OverLordGoldDragon/wavespin/},
  year={2022},
  doi={10.5281/zenodo.5080508},
}
```

## References

WaveSpin originated as a fork of [Kymatio](https://github.com/kymatio/kymatio/) [2] (who [scrapped my work](https://www.reddit.com/r/antiwork/comments/utrd42/1800_hours_of_unpaid_work_scrapped_last_minute/)). The library is showcased in [1] for audio classification and synthesis. JTFS was introduced in [3], and Wavelet Scattering in [4].

 1. J. Muradeli, C. Vahidi, C. Wang, H. Han, V. Lostanlen, M. Lagrange, G. Fazekas (2022). [Differentiable Time-Frequency Scattering in WaveSpin](https://arxiv.org/abs/2204.08269).
 2. M. Andreux, T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. Andén, E. Belilovsky, J. Bruna, V. Lostanlen, M. J. Hirn, E. Oyallon, S. Zhang, C. Cella, M. Eickenberg (2019). [Kymatio: Scattering Transforms in Python](https://arxiv.org/abs/1812.11214).
 3. J. Anden, V. Lostanlen, S. Mallat (2015). [Joint time-frequency scattering for audio classification](https://ieeexplore.ieee.org/abstract/document/7324385).
 4. S. Mallat (2012). [Group Invariant Scattering](https://arxiv.org/abs/1101.2286).

## License

WaveSpin is MIT licensed, as found in the LICENSE file. Some source functions may be under other authorship/licenses; see NOTICE.txt.
