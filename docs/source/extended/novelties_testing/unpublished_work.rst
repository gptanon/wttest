Unpublished Work
================

Below are compressed abstracts of my unpublished work, most of which is publicly implemented in WaveSpin. 

Format: Title, Abstract paragraph (no breaks), maybe Comments (in parentheses), Bullets to relevant code / docs.


.. _uw0:

Smart Scattering Paths: Speed and Memory via Sparsity
-----------------------------------------------------

We present a novel algorithm for maximizing the energy-efficiency of scattering coefficients, with empirical guarantees for 
user-controllable thresholding. We accomplish this by studying the Fourier properties of the complex modulus nonlinearity and 
interdependencies of first- and second-order scattering filterbanks, and data mine with machine learning and stochastic and 
real-world data to attain precise energy thresholding. We also show that every existing approach is necessarily suboptimal, and is 
either too conservative or too liberal in coefficient exclusion.

(See README for more info.)

- In `Scattering Docs <../../scattering_docs.html>`__: `smart_paths`


.. _uw1:

Fast Time-Frequency Analysis: Min-Maxing Compute Graphs
-------------------------------------------------------

We present methodology to speed up CWT, STFT, and scattering transforms by **x2-10** relative to standard optimized implementations, 
along a novel subroutine reusable in other domains.

(For obvious reasons, not much more can be stated (I'm selling it). If your application demands fast time-frequency analysis, consider hiring me.)


.. _uw2:

Beyond 4D: Exploiting Full JTFS Structure
-----------------------------------------

We extend the success of 3D and 4D JTFS by exploiting the final degree of freedom. 
The adjustment provides a notable classification improvement in all tested configurations.
We visualize attention heatmaps to explain the improvements in an interpretable manner.

(To avoid "scooping", I don't elaborate further.)


.. _uw3:

JTFS in 3D and 4D: Exploiting Conv-Net Synergies and Quefrential Interdependencies
----------------------------------------------------------------------------------

Prior works utilize 1D and 2D convolutions upon JTFS coefficients for classification. 
We observe major improvements by convolving in 3D and 4D, which exploits cross-wavelet 
dependencies in log-frequency and log-quefrency dimensions.
We additionally develop a framework for choosing hyperparameters for convolutional 
neural networks that are trained with these features, and interpret these synergies 
in an information-theoretic manner.

- In `wavespin.toolkit`:  `pack_coeffs_jtfs <../../wavespin.html#wavespin.modules._toolkit.postprocessing.pack_coeffs_jtfs>`__


.. _uw4:

Maximizing Coefficient Informativeness and Accuracy in JTFS
-----------------------------------------------------------

JTFS extends the sparsities of second-order time scattering with two additional dimensions, 
creating new problems for quality of coefficients. We study these problems and implement 
solutions, with careful adjustments in stride, padding, unpadding, and scaling of coefficients, 
and parameters of filterbanks. We additionally propose a new frequential averaging scheme to 
allow greater subsampling without imposing greater invariance, improving information preservation.

- In `Scattering Docs <../../scattering_docs.html>`__ (major): `aligned`, `out_3D`, `F_kind`, `sampling_filters_fr`, `pad_mode_fr`
- In `Scattering Docs <../../scattering_docs.html>`__ (minor): `max_noncqt_fr`, `N_fr_p2up`, `N_frs_min_global`



.. _uw5:

Validating and Optimizing Real-World Scattering Via Theory
----------------------------------------------------------

Real-world scattering is discrete and finite, which often behaves considerably differently from the 
continuous domain in which it was developed. This invalidates some theoretical guarantees, and requires 
adjustments in others. We study these differences and develop numeric tests on a wide range of signal profiles,
including real-world and stochastic. Adjustments are proposed to maximize agreement between practice and theory, 
which improves frequential averaging and frequency-dependent time-shift discriminability in JTFS.

- `tests/scattering1d <https://github.com/OverLordGoldDragon/wavespin/tree/master/tests>`__, especially `test_jtfs.py`, `test_correctness.py`, `test_measures.py`


.. _uw6:

Discrete Measures for Wavelet Transforms, with Scattering Applications
----------------------------------------------------------------------

Real-world wavelets are discrete and finite, which often behave considerably differently from the 
continuous domain in which they're developed. Typically, continuous-time formulas are applied directly 
to these discretizations; this works well most of the time, but fails drastically where it often matters. 
We develop discrete measures and efficient numeric algorithms to compute them, and show that they remain
accurate in all settings, including few-sample and non-bandlimited regimes. We apply these measures to improve 
determination of padding and subsampling in the wavelet scattering transform.

- `wavespin.utils.measures <../../wavespin.utils.html#module-wavespin.utils.measures>`__


.. _uw7:

Debugging and Postprocessing with Scattering Tools and Visualizations
---------------------------------------------------------------------

We present novel visualizations and utilities for JTFS and time scattering coefficients. Included are 
static and animated illustrations of output coefficients and network filterbanks. Major debugging aid 
is achieved by visualizing all relevant structures with quick function calls. 
Included also are methods for normalizing coefficients and filterbanks for machine learning, and 
making relevant comparisons of coefficient distances on signals of interest.

- `wavespin.visuals <../../wavespin.html#module-wavespin.visuals>`__
- `wavespin.toolkit <../../wavespin.html#module-wavespin.toolkit>`__
