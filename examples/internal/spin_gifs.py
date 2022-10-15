# -*- coding: utf-8 -*-
"""
Spinned visuals
===============
Generates spinned visuals seen in README and docs.

3+1D visuals. We slightly modify the built-in functions so that the tensor
being animated isn't square in dimensions - namely, the spatial dimension
is upsampled but not temporal, so that we don't require 60FPS to render the
same duration, else it causes GIF renderers to slow down.
"""
import os
import wavespin.visuals as v

#%% Commons
pair_presets = {0: ('up',),
                1: ('up', 'dn'),
                2: ('up', 'phi_f', 'dn', 'phi_t', 'phi', 'phi_t_dn')}

def maker(pairs):
    # reuse these from `viz_spin_2d` except higher `N`
    N, xi0, sigma0 = 256, 4., 1.35
    # use lower `N_time` so we do 30 FPS (see GIF note in docs)
    N_time = int(128 * (24 / 30))
    pair_waves = {pair: v.animated.make_jtfs_pair(N, pair, xi0, sigma0, N_time)
                  for pair in pairs}
    return pair_waves

#%% One spin
name = 'viz_spin_up.gif'
temp_name = '_' + name
pair_waves = maker(pair_presets[0])
v.viz_spin_2d(pair_waves=pair_waves, verbose=0, savepath=temp_name, is_time=1,
              anim_kw={'linewidth': 3})

# Output size on author's machine was 1152x576. Let's crop a little:
if os.path.isfile(name):
    os.remove(name)
cmd = (f"convert {temp_name} -coalesce -repage 0x0 -crop 620x530+266+36 "
       f"+repage -layers optimize {name}")
os.system(cmd)
os.remove('_' + name)

#%% Both spins
name = 'viz_spin_both.gif'
temp_name = '_' + name
pair_waves = maker(pair_presets[1])
v.viz_spin_2d(pair_waves=pair_waves, verbose=0, savepath=temp_name, is_time=1,
              fps=24, anim_kw={'linewidth': 3})

# Output size on author's machine was 1152x576. Let's crop a little:
if os.path.isfile(name):
    os.remove(name)
cmd = (f"convert {temp_name} -coalesce -repage 0x0 -crop 1122x430+15+55 "
       f"+repage -layers optimize {name}")
os.system(cmd)
os.remove('_' + name)

#%% All pairs
name = 'viz_spin_all.gif'
temp_name = '_' + name
pair_waves = maker(pair_presets[2])
v.viz_spin_2d(pair_waves=pair_waves, verbose=0, savepath=temp_name, is_time=1)

# Output size on author's machine was 1152x576. Let's crop a little:
if os.path.isfile(name):
    os.remove(name)
cmd = (f"convert {temp_name} -coalesce -repage 0x0 -crop 950x550+101+13 "
       f"+repage -layers optimize {name}")
os.system(cmd)
os.remove('_' + name)
