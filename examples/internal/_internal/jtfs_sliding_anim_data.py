# -*- coding: utf-8 -*-
import os
import os.path as op
import numpy as np
import librosa
import scipy.io.wavfile as siw
from scipy.signal import decimate
from pathlib import Path

CURRENT_DIR = str(Path(__file__).parent.resolve())
SAVEDIR = str(Path(Path(CURRENT_DIR).parent, 'data').resolve())
# in seconds
TARGET_DURATION = 4.4

if not op.isdir(SAVEDIR):
    os.mkdir(SAVEDIR)

#%% Helpers ##################################################################
def convert_to_target_duration(x, sr, T):
    T_now = len(x) / sr
    speedup_factor = T_now / T
    return librosa.effects.time_stretch(x, rate=speedup_factor)


def validate_conversion(name, sr, T):
    # validate, ensure within 1% error of target duration
    sr_load, x_load = siw.read(op.join(SAVEDIR, name.replace('make_', '')))
    assert sr == sr_load, (sr, sr_load)
    assert abs(len(x_load) / sr - TARGET_DURATION) / TARGET_DURATION < .01

def yt_dl(url, savename):
    # https://github.com/ytdl-org/youtube-dl/
    ext = Path(savename).suffix[1:]
    savename_fmt = savename.replace(ext, r'%(ext)s')
    template = op.join(SAVEDIR, savename_fmt)
    cmd = (fr'youtube-dl --output "{template}" --extract-audio --audio-format '
           fr'"{ext}" --audio-quality 0 {url}')
    out = os.system(cmd)
    if out:
        raise RuntimeError(f"System command exited with status {out} "
                           f"for command `{cmd}`")

def save_data(x, name):
    name = name.replace('.wav', '.npy')
    np.save(op.join(SAVEDIR, name), x)

#%% Trumpet ##################################################################
x, sr = librosa.load(librosa.ex('trumpet'))
# drop silent part
x = x[:80000]

name = 'make_librosa_trumpet_slow.wav'
siw.write(op.join(SAVEDIR, name), sr, x)
# .83 at https://audiotrimmer.com/audio-speed-changer/
# save as same name minus `make_`

#%%
# 4.362 in this case
validate_conversion(name, sr, TARGET_DURATION)

# save data to transform
nm = name.replace('make_', '')
x = siw.read(op.join(SAVEDIR, name.replace('make_', '')))[1]
x = x.astype('float64')
x /= x.std()
save_data(x, nm)

# pad to make first half of animated slide empty, to help real-time synch
# NOTE: hard-coded for animation parameters, slide size 75, JTFS out time len 523
xp = np.pad(x, [int(75/375/1.5 * len(x)), 0])
nm = 'pre_pad_' + nm
xpd = decimate(xp, 2, ftype='fir')
save_data(xpd, nm)

#%% Shepard Tone #############################################################
# note, download may take some minutes
name = 'make_shepard_tone_fast.wav'
yt_dl(r"https://youtu.be/BzNzgsAE4F0", name)
#%%
sr, x = siw.read(op.join(SAVEDIR, name))
# select interesting part
x = x[2**21:2**22 + 2**21, 0]
x = x.astype('float64')
x /= x.std()

# for this, VideoPad's "Speed Change" "Effects" was actually used, with target
# duration 4.369. librosa sounds similar but VP was preferred.
# https://www.nchsoftware.com/videopad/
xs = convert_to_target_duration(x, sr, TARGET_DURATION)
nm = name.replace('make_', '')
siw.write(op.join(SAVEDIR, nm), sr, xs)
validate_conversion(name, sr, TARGET_DURATION)

# save data to transform
xd = decimate(x, 4, ftype='fir')
nm = nm.replace('_fast', '')
save_data(xd, nm)

#%% Brain Waves ##############################################################
# requires access to proprietary data, request from
# https://www.epilepsyecosystem.org/
x = np.load(op.join(CURRENT_DIR, 'Pat3Train_7_1.npy'))
sr = 400
x /= x.std()
x = x[3, 110000:-100000]

name = 'make_brain_waves_fast.wav'
siw.write(op.join(SAVEDIR, name), sr, x)
# for this, VideoPad's "Speed Change" "Effects" was used, by 1700%
# https://www.nchsoftware.com/videopad/

# save data to transform
nm = name.replace('_fast', '')
save_data(x, nm)
