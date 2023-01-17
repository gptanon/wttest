## Lying with benchmarks

CWT results could look like this:



This feeds the fair filterbank (so _worse_ is possible) to `pywt.cwt` with its default `method='conv'`.

## Benchmarking fairly

Below describes the steps taken to ensure a fair comparison. "Fair" meaning, the test cases are realistic, and transforms are matched in their specifications. 
This ends up being generous for `scipy` and (especially) `pywt`, as it's easy to specify inputs that take drastically longer to compute, and hard to know how not to.

  1. Wavelet temporal supports and quality factors are matched. This is key, as `scipy` and `pywt` sample wavelets in time, and don't always FFT-convolve.
  2. Boundary effects are matched. Specifically, it'd be possible to use very large-scale wavelets for all libraries, but `pywt` would suffer unfairly here as its it'd pad extra to ensure proper decay.
  3. Zero-padding is used. One could instead pad manually (as is often needed), but that'd double compute times for `pywt` and `scipy` - the benchmark then measures a difference in features, rather than really a difference in performance.
  4. Smaller `hop_size` for shorter `x`, matching practical expectations on minimizing downsampling losses
  5. Warmup is accounted for, taking a few dummy iterations before benchmarking to ensure each library's caching and related optimizations are executed.

Below describes what's _not_ fair performance-wise. It's done because doing otherwise is overly generous and impractical.

  6. `float32` precision is used. `pywt` and `scipy` don't support these, but `float32` is more than enough in most cases, so that's their limitation.
  7. Only WaveSpin supports `hop_size`, so it's technically meaningless to measure this for others. Yet, a x200-upsampled input is most often both untenable and unnecessary, so we'll be downsampling in practice anyway.
  8. (Scattering) Paths are different. It'd be possible to make a 100% apples-to-apples comparison, but the difference wouldn't be great (for the specific configuration tested), and the distinction unimportant as WaveSpin's paths are more correct, and matching the paths isn't something a user could do.
