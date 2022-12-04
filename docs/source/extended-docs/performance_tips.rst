Performance Tips
****************

Configs to improve speed / memory.

JTFS
----

  - `paths_exclude = {'j2': 1}`
  - `sampling_filters_fr = 'exclude'`
  - `F = 'global'`
  - lower `T`
  - `pad_mode_fr='zero'`
  - `pad_mode='zero'`
  - `max_pad_factor=0` -- can be done safely if `J <= log2(len(x)) - 3`, and especially if `len(x)` isn't a power of 2 (which we necessarily pad to a power of 2 even with `=0`)
  - `max_pad_factor_fr=1` -- `0` really isn't recommended, *except* if it's like with `max_pad_factor`, now `J->J_fr` and `len(x) -> N_frs_max`
  - `out_3D=False` 
  - `precision = 'single'` (already default for non-numpy)


Time Scattering
---------------
Just `vectorized=True`.
