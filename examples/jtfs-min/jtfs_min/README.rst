JTFS-Minimal
============

This is a minimalistic implementation of Joint Time-Frequency Scattering, useful for learning. 
It implements JTFS with most arguments fixed. Meaningfully, this implementation differs as follows:

- Only spinned coefficients are computed
- Frequential filterbank structure mimics `Scattering1D`'s
- Most of compute-relevant quantities are computed at runtime rather than build time
- Frequential padding is always maxed (though not necessarily sufficient), so `n2`-dependence is dropped
- Frequential averaging is always done. Simple enough to understand unaveraged case from code, and to obtain approx. unaveraged coeffs via `F=1`
- `out_3D=True; aligned=True; pad_mode=pad_mode_fr="zero"`
- No energy correction
- Otherwise, the implementation is complete and can be studied as a mostly correct JTFS.

Fixed arguments:

:: 

    out_type="array"; pad_mode="zero"; smart_paths="primitive"; vectorized=False; r_psi=sqrt(.5);
    max_pad_factor=1; analytic=False; paths_exclude=None; precision="double"; average_fr=True;
    aligned=True; out_3D=True; sampling_filters_fr="resample"; analytic_fr=False; F_kind="average";
    max_pad_factor_fr=1; pad_mode_fr="zero"; normalize_fr="l1"; r_psi_fr=sqrt(.5); max_noncqt_fr=None;
    out_exclude=None; frontend="numpy"; J2=J1;
