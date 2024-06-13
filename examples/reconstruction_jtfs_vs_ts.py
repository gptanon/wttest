# -*- coding: utf-8 -*-
print("TODO DO OMIT LOW j1 compare ts vs jtfs then oversampling_fr=99 and viz")
print("TODO max_noncqt_fr? & other differences")
print("TODO MSE")
print("TODO try only up & dn, big lr diff on chirp rev case?")
print("TODO hear chirp case? w/ and w/o S1")
print("TODO `restrict` in C? pgo?")
import numpy as np
import torch
import librosa
from scipy.io.wavfile import write
from scipy.fft import fft, ifft, ifftshift

from wavespin import Scattering1D, TimeFrequencyScattering1D, gauss_1d
from wavespin.visuals import plot, plotscat, imshow, viz_jtfs_2d
from wavespin.toolkit import echirp, rel_l2, jtfs_to_numpy
from wavespin.utils.gen_utils import npy
# 1/0
#%%
def do_jtfs(sc, x, div_vec):
    out = sc(x)
    for pair in out:
        out[pair] = out[pair][0] / div_vec[pair]
    # out = torch.vstack(list(out.values()))
    out = torch.vstack([c.reshape(-1, c.shape[-1]) for c in out.values()])
    return out

def save_as_mp3(x, sr, name):
    xint = (x * (np.iinfo(np.int16).max / abs(x).max())
            ).astype(np.int16)
    write(f"{name}.wav", sr, xint)

#%%
def viz_data(data, sc, use_jtfs):
    losses, losses_recon, xn = [data[k] for k in "losses losses_recon xn".split()]
    title = "JTFS" if use_jtfs else "TS"
    ckw = dict(abs=1, show=1, title=title)

    plotscat(losses, **ckw, ylims=(0, None))
    plotscat(losses_recon, **ckw, ylims=(0, None))
    imshow(npy(sc.cwt(xn))[:, ::10], **ckw, w=1.3, h=1.2, norm_scaling=1)
    print(min(losses), title)

def run_reconstruction(sc, div_vec, n_iters, device, use_jtfs, fn=None):
    N = sc.N
    # lr_max = {
    #     0: .5,
    #     1: 1,
    # }[int(use_jtfs)]
    lr_max = {
        0: 2,
        1: .25,  # 2
    }[int(use_jtfs)
      # ] * 100000 * 2
      ] * 100000 * 8
    # * 200

    n_cos = int(1*n_iters)
    n_cos += n_cos % 2  # ensure even
    lrs = (1 + np.sin(2*np.pi * .25 * np.linspace(-1+1e-2, 1, n_cos//2, 1))
           ) / 2 * lr_max
    lrs = np.hstack([lrs, lrs[::-1]])

    y = torch.from_numpy(xfloat).to(device)
    yn = y.detach().cpu

    if use_jtfs:
        Sy = sc(y)
        # mxdiv = torch.max(torch.stack([
        #     torch.abs(c).max() for c in Sy.values()]))
        for pair in Sy:
            if 1:#pair != 'S0':
                div_vec[pair] = torch.abs(Sy[pair][0]).max()
                #dim=1).values.reshape(-1, 1)
            else:
                div_vec[pair] = 1
            # div_vec[pair] = mxdiv
            if pair.endswith('_dn'):
                div_vec[pair] = div_vec['psi_t * psi_f_up']  # arbitrary up vs dn
            Sy[pair] = Sy[pair][0] / div_vec[pair]
        # Sy = torch.vstack(list(Sy.values()))
        if fn is not None:
            Sy = fn(Sy)
        Sy = torch.vstack([c.reshape(-1, c.shape[-1]) for c in Sy.values()])
        # Sy = torch.log10(1 + 10 * Sy)
    else:
        Sy = sc(y)[1:]
        for slc in (o0_slc, o1_slc, o2_slc)[1:]:
            div_vec[slc] = torch.abs(Sy[slc]).max()
        Sy /= div_vec

    torch.manual_seed(0)
    x = torch.randn(N, device=device, dtype=tdtype)
    x /= torch.max(torch.abs(x))
    x.requires_grad = True
    opt = torch.optim.SGD([x], lr=lrs[0], momentum=.9, nesterov=0)
    # opt = torch.optim.RMSprop([x], lr=lrs[0])
    # opt = torch.optim.NAdam([x], lr=lrs[0])
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.L1Loss()

    losses, losses_recon = [], []
    x_hist, g_hist = [], []
    for i in range(n_iters):
        opt.zero_grad()

        if use_jtfs:
            Sx = do_jtfs(sc, x, div_vec)
        else:
            Sx = sc(x)[1:]
            Sx = Sx / div_vec
        # Sx = torch.log10(1 + 10 * Sx)

        loss = loss_fn(Sx, Sy)
        loss.backward()
        opt.step()

        # loop metrics
        losses.append(float(loss.detach().cpu().numpy()))
        xn, yn = [g.detach().cpu().numpy() for g in (x, y)]
        losses_recon.append(float(rel_l2(yn, xn)))

        # debug
        x_hist.append(xn)
        g_hist.append(opt.param_groups[0]['params'][0].grad)

        # "progbar"
        print(end='.', flush=True)

        # LP scheduling
        # if abs(i - n_iters) <= 4:
        #     continue
        if i != n_iters - 1:
            for g in opt.param_groups:
                if i <= n_cos - 2:
                    g['lr'] = lrs[i + 1]
                else:
                    if i >= 1:
                        _f = 1.1
                        factor = _f if losses[-1] < losses[-2] else 1/(_f*1.05)
                        g['lr'] *= factor

    data = dict(
        losses=losses, losses_recon=losses_recon, xn=xn,
        x_hist=x_hist, g_hist=g_hist
    )
    return data

#%%
nm = ('trumpet', 'fishin', 'dog', 'echirp', 'grid')[-1]
precision = ('single', 'double')[0]

dtype = ('float32' if precision == 'single' else 'float64')
tdtype = getattr(torch, dtype)

if nm in ('trumpet', 'fishin'):
    data, sr = librosa.load(librosa.ex(nm))
    # data = data[:80000]
    data = data[8*sr:18*sr]
elif nm == 'dog':
    data, sr = librosa.load(r"C:\Users\overl\Downloads\dog-barking-70772.mp3")
    data = data[10000:100000]
elif nm == 'echirp':
    N = 32768
    w = ifftshift(ifft(gauss_1d(N, .13/(N/16))).real)
    # xfloat = np.pad(echirp(N//16, fmin=64, fmax=N/64), int(15*N/32)) * w
    data = echirp(N, fmin=64) * w
elif nm == 'grid':
    sr = 22050
    M = 2500
    x = np.zeros(2**14)
    w = ifftshift(ifft(gauss_1d(M, .13/(M/8))).real)
    offset = 2000
    offset = 8192
    x[offset:offset+M] = np.cos(2*np.pi*np.arange(M)/M*150) * w
    # x[offset+5000:offset+5000+M] += np.cos(2*np.pi*np.arange(M)/M*750) * w
    # x[offset+5000:offset+5000+M] += np.cos(2*np.pi*np.arange(M)/M*30) * w

    # x += np.sum([np.roll(x, int(1.*M)*i) for i in range(1, 3)], axis=0)
    data = x

xfloat = data.copy()
save_as_mp3(xfloat, sr, nm)
xfloat = (xfloat / np.mean(abs(xfloat))).astype(dtype)

#%%
use_jtfs = 1
use_ts = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pinclude = ('u','d','pt','pf','p')[:]

Q = 20
r_psi = .95
N = len(xfloat)
J = int(np.log2(N) - 5)  # -5
J = (J, J + 2)
T = 8192//4


ckw_ts = dict(J=J, Q=Q, r_psi=r_psi, max_pad_factor=1, frontend='torch',
              pad_mode='zero', T=T,
              precision=precision)
ckw_jtfs  = dict(J_fr=5, Q_fr=1, average_fr=0, max_pad_factor_fr=0,
                 pad_mode_fr='zero', sampling_filters_fr='exclude',
                 # F=64,
                 # paths_exclude={'j2': -1},
                 **ckw_ts)
out_exclude = ['S0', 'S1', 'psi_t * psi_f_up', 'psi_t * psi_f_dn',
               'phi_t * phi_f', 'phi_t * psi_f', 'psi_t * phi_f']
for pi in pinclude:
    out_exclude.pop(out_exclude.index(
        {'u':  'psi_t * psi_f_up',
         'd':  'psi_t * psi_f_dn',
         'pt': 'phi_t * psi_f',
         'pf': 'psi_t * phi_f',
         'p':  'phi_t * phi_f',
         }[pi]
    ))
out_exclude = tuple(out_exclude)
ckw_jtfs['out_exclude'] = out_exclude

if use_jtfs:
    jtfs = TimeFrequencyScattering1D(N, **ckw_jtfs, out_type='dict:array',
                                     smart_paths=.01,
                                     do_energy_correction=True,
                                     ).to_device(device)
    # div_vec_jtfs = {
    #     pair: torch.ones(len(mn), device=device).reshape(-1, 1)
    #     for pair, mn in sc.meta()['n'].items()}
    div_vec_jtfs = {}

#%%
if use_ts:
    ts = Scattering1D(N, **ckw_ts, out_type='array'
                      ).to_device(device)

    n_n1s, n_paths_total = len(ts.psi1_f), len(ts.meta()['n'])
    o0_slc = slice(0, 1)
    o1_slc = slice(0, 0 + n_n1s)
    o2_slc = slice(0 + n_n1s, n_paths_total)
    div_vec_ts = torch.ones(n_paths_total - 1, device=device, dtype=tdtype
                            ).reshape(-1, 1)

if use_jtfs and use_ts:
    assert ts.paths_include_n2n1 == jtfs.paths_include_n2n1

#%%
def jtfs_mod(out):
    out['psi_t * psi_f_up'], out['psi_t * psi_f_dn'] = (
        out['psi_t * psi_f_dn'], out['psi_t * psi_f_up'])
    return out

n_iters = 100

if use_jtfs:
    data_jtfs = run_reconstruction(jtfs, div_vec_jtfs, n_iters, device, use_jtfs=1,
                                   fn=None)
if use_ts:
    data_ts = run_reconstruction(ts, div_vec_ts, n_iters, device, use_jtfs=0)

viz_data(data_jtfs, jtfs, use_jtfs=1)
viz_data(data_ts,   ts,   use_jtfs=0)
imshow(npy(jtfs.cwt(xfloat)), abs=1, w=1.3, h=1.2, title="orig")

#%%
def make_savename(pinclude, use_jtfs):
    savename = "{}_{}_{}_{}".format(nm, T, r_psi, "jtfs" if use_jtfs else "ts")
    if use_jtfs:
        savename += '_' + '_'.join(pinclude)
    return savename
1/0
if use_jtfs:
    save_as_mp3(data_jtfs['xn'], sr, make_savename(pinclude, 1))

if use_ts:
    save_as_mp3(data_ts['xn'], sr, make_savename(pinclude, 0))

#%%
if 1:
    # xn = np.load('xn.npy')
    xn = data_jtfs['xn']
    div_vec_jtfs_n = {k: npy(v) for k, v in div_vec_jtfs.items()}

    # pe = jtfs.paths_exclude
    # pe['n2'].extend([0, 1, 2, 3, 4, 5])
    jtfs.update(out_exclude=())
    # jtfs.update(paths_exclude=pe)

    # out0 = jtfs_to_numpy(jtfs(xfloat))
    out1 = jtfs_to_numpy(jtfs(xn))

    # out0 = {k: v / div_vec_jtfs_n[k] for k, v in out0.items()}
    out1 = {k: v / div_vec_jtfs_n[k] for k, v in out1.items()}

#%%
if 1:
    ckw = dict(w=1.5, h=1.5, viz_filterbank=0, axis_labels=0, equalize_pairs=1,
               plot_cfg=dict(coeff_color_max_mult=.5))

    viz_jtfs_2d(jtfs, out0, **ckw)
    # viz_jtfs_2d(jtfs, out1, **ckw)

