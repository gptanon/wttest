How's it work?
**************

If you're from...
-----------------

... machine learning
^^^^^^^^^^^^^^^^^^^^

Scattering is a feature extractor that resembles CNNs. Like CNNs, there's convolutions followed by nonlinearities and downsampling, in multiple layers. Unlike CNNs, 
the kernels aren't learned but fixed. Scattering is advantageous when data is limited, as it provides a strong inductive prior for real-world tasks. It's also useful
in synthesis, as the entire network is differentiable, and in interpretability, as the transform is bio-inspired, sparse, and mathematically grounded.


... signal processing
^^^^^^^^^^^^^^^^^^^^^

Scattering is time-frequency analysis adapted to real-world irregularities. It alternates continuous wavelet transforms, modulus nonlinearities, averaging, and subsampling. 
While traditional signal processing aptly handles many real tasks, it falls off in absence of sufficient mathematical definition or linearization ability. 

The Fourier modulus, for example, is identical for vastly different image textures, and is unstable to deformations. Mean, kurtosis, and other aggregate measures, while 
much more invariant, lose too much information. In classification, this is the problem of invariance vs discriminability: we desire representations that don't change within 
the same class, but do for different classes. Scattering achieves this by design, with sparse and warp-stable filters, and modulus averaging, that enjoy strong invariance, 
stability, and invertibility.


... machine learning and signal processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scattering extracts discriminative invariants. It is invertible within a global shift, hence retaining ability to discriminate any classes not differing by said shift, and 
achieving shift-invariance. It is stable to spatial warp deformations, hence linearizing and bounding their distances. It differs from CWT and STFT, which are unstable to shifts 
or frequency transposition, and from MFCC which doesn't recover information lost to averaging. Besides selective activation, its sparsity also enables compression, hence 
dimensionality reduction.


... non-technical
^^^^^^^^^^^^^^^^^

We looked at ears and nerves and made formulas, then made number crunchers that use those formulas to tell if a sound is cats or dogs.


More on synthesis, other applications
-------------------------------------

Sparsity, or selective activation, is a key. Note that theoretically, JTFS is less invertible than time scattering, per additional averaging along frequency. Despite this, 
in practice, it is *more* invertible. A simple explanation is coefficient distances: JTFS is sensitive to relative frequency shifts, or slopes in time-frequency geometry. 
If low loss is to be achieved, a gradient descent optimizer has no choice but to accurately replicate the scalogram, and scalograms are perfectly invertible within a 
global phase shift, which is a strong inversion. In contrast, while time scattering will vary between a chirp and an impulse, it will do so in drastically lesser norms, 
making low loss achievable with either signal or anything in-between.

Selective activation's use is also self-explanatory, as we want to "select" parts of a signal to modify; with scattering we get the added benefit of stabilities.

Generative modeling and regression benefit also. Ultimately, scattering is useful wherever its properties are: discriminability-shift-invariance tradeoff, 
warp-stability, AM-FM sparsity.
