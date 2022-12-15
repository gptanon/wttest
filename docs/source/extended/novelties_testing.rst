Novelties, Testing
==================

JTFS Novelties
--------------

1. `smart_paths`: path optimization, accounting for intricacies of JTFS
2. `aligned == False`: eliminate oversampling in frequential scattering -- `disc. <https://github.com/kymatio/kymatio/discussions/716>`__
3. `sampling_psi_fr == 'recalibrate'`: maximize coefficient informativeness by dedicating a filterbank to every frequential scale; `disc. <https://github.com/kymatio/kymatio/discussions/721>`__
4. `out_3D == True`: more info-rich if packing into 3D/4D, and enforces per-`n2` alignment
5. `F_kind == 'decimate'`: improved information preservation; enables unaveraged and unaliased subsampling
6. `pad_mode == 'reflect'`: make it work without compromising FDTS discriminability, via `backend.conj_reflections()`; `disc. <https://github.com/kymatio/kymatio/discussions/752#discussioncomment-864234>`__
7. `pad_mode_fr == 'conj-reflect-zeros'`: reflect-like padding across frequency; `disc. <https://github.com/kymatio/kymatio/discussions/752#discussioncomment-864234>`__
8. `max_pad_factor`: `None` to eliminate filter distortion and a class of boundary effects in temporal scattering
9. `max_pad_factor_fr`: frequential scattering variant, also supporting per-`n2` control
10. `energy_norm_filterbank()`: precise, individual filter rescaling for meeting the Littlewood-Paley sum bound
11. `pack_coeffs_jtfs()`: correctly pack coeffs into a 4D/3D tensor
12. `out_exclude, paths_exclude`: compute only what's needed
13. `meta` (coeffs): exhaustive, and added: `stride, slope, is_cqt`
14. `meta` (filters): added: `support, width, scale, bw, bw_idxs, is_cqt`
15. `max_noncqt_fr`, `N_fr_p2up`, `N_frs_min_global`: advanced quality controls for niche cases

For more info on any parameter, "Ctrl + F" in `Scattering Docs <https://wavespon.readthedocs.io/en/latest/scattering_docs.html>`_.


Smart Scattering Paths
----------------------
See in `Github README <https://github.com/gptanon/wttest#smart-scattering-paths>`_.


Testing
-------

Extensive and painstaking tests for agreement with theory and edge case handling.

1. Agreement with theory:

 - FDTS discriminability: sensitivity to time-frequency slopes while retaining time-shift invariance
 - Reconstruction: that both time scattering and JTFS work, and that JTFS beats time scattering for same `T`
 - Littlewood-Paley sum: full and even tiling of frequency axis
 - Energy conservation: coefficient completeness and attenuation fixes
 - Frequency transposition invariance: ensuring it works like time averaging, and if it doesn't, that it's explained or remedied
 - Alignment: `n1` paths per `(n2, n1_fr)`, enabling the true 4D JTFS structure
 - Time-warp stability: not included in CI testing but confirmed in example and paper # TODO

2. Library completeness:

 - Coverage with high %
 - GPU and differentiability for every applicable backend
 - Backends: almost all features supported with all backends
 - Performance benchmarks (as example scripts)
