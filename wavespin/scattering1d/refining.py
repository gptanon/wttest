# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2022- John Muradeli
#
# Distributed under the terms of the MIT License
# (see wavespin/__init__.py for details)
# -----------------------------------------------------------------------------
import math
import numpy as np
import warnings
from ..utils.measures import (compute_bandwidth, compute_filter_redundancy,
                              r_psi_to_redundancy)
from ..utils.gen_utils import npy
from .filter_bank import j2_j1_cond


def smart_paths_exclude(psi1_f, psi2_f, e_loss=.01, level=1, e_th_direct=None):
    """Intelligently excludes second-order temporal paths that are likely to be
    uninformative and contribute minimal energy to scattering coefficients.

    The explanation of the algorithm remains proprietary for the time being.
    In a nutshell, it's a rigorously developed and correct version of the
    `j2 > j1` criterion used by some libraries.

    Parameters
    ----------
    psi1_f : list[dict]
        `sc.psi1_f`, where `sc` is `Scattering1D` or `TimeFrequencyScattering1D`.

    psi2_f : list[dict]
        `sc.psi2_f`, where `sc` is `Scattering1D` or `TimeFrequencyScattering1D`.

    e_loss : float[>0, <1]
        Energy loss threshold. No more than this fraction of energy will be lost
        relative to the full transform. See `smart_paths` in
        `wavespin.scattering1d.frontend.base_frontend.Scattering1D`.

    level : int
        Conservativeness of the algorithm; higher means greater guarantee, but
        less paths excluded. See `smart_paths` in
        `wavespin.scattering1d.frontend.base_frontend.Scattering1D` or
        `help(Scattering1D)`.

    e_th_direct : float [>0, <1] / None
        `e_loss` is used to set `e_th`, an internal parameter used by the
        algorithm to exclude wavelets. `e_th_direct` bypasses `e_loss` to set
        `e_th` directly, which can be useful for debugging.

    Returns
    -------
    paths_exclude_smart : dict['n2, n1': list[int]]
        Second-order temporal paths to exclude, e.g.
        `{'n2, n1': [(0, 0), (1, 3)]}`.

    level=0 details
    ---------------
    Surveyed datasets are:

        - Seizure iEEG of three patients, 17280 hours and 93GB at float32.
          16 channels, 10 minute segments at 400Hz, 103680 segments total
          (channels counted as segments).
          https://www.kaggle.com/c/melbourne-university-seizure-prediction
        - Musical instruments, 18 hours and 10.6GB at float32.
          1 channel, 2.972 second segments at 44100Hz, 21572 segmets total.
          https://zenodo.org/record/3464194
    """
    # compute bw's energy threshold based on specified loss
    if e_th_direct is None:
        e_th = _e_th_from_e_loss(psi1_f, psi2_f, e_loss, level)
    else:
        e_th = e_th_direct

    # .5 since higher numbers were never tested
    assert e_th <= .5, e_th
    # filter length, used later
    N_psi = len(psi1_f[0][0])

    # compute psi1 bandwidths
    bw1_all = []
    for n1, p1 in enumerate(psi1_f):
        # take left of peak if we're in high freqs, else right of peak.
        # note this excludes the peak itself, which would be the DC bin of
        # the AM spectrum, so now we do lp_sum[bw] instead of lp_sum[bw - 1]
        r = e_th
        R = r / (1 - r)

        ckw = dict(pf=npy(p1[0]),
                   center_input=1, criterion_amplitude=np.sqrt(R))
        bw1 = compute_bandwidth(**ckw)
        # bound at N_psi//2 in case of `analytic=False`; don't need more
        bw1 = min(bw1, N_psi // 2)
        bw1_all.append(bw1)
    # replace bws from max bw wavelet up to Nyquist with the max bw
    mx_idx = np.argmax(bw1_all)
    bw1_all[:mx_idx] = [np.max(bw1_all)] * mx_idx

    # `3` since lower freqs are uncertain within decay for insufficient padding;
    # will compensate manually later by skipping everything past the first skip.
    # main purpose here is to avoid premature first skip.
    # also exclude last (lowest freq) from assertion per low freq bw inflation -
    # then adjusted
    assert np.diff(bw1_all[:-1]).max() <= 3, (
        "higher freq doesn't always correspond to higher bandwidth, but need "
        "this to not skip sparsely (e.g. n1=3, 5, 6 instead of n1=3, 4, 5, 6). "
        "got bw1_all\n%s" % ", ".join(['%s' % bw1 for bw1 in bw1_all]))

    # force `bw1_all[i] <= bw1_all[i - 1]`
    # handles near DC (the 'uncertainty'), increasing bw1 to the running maximum.
    # also handles near Nyquist but that's already done (had to be before assert),
    # and excludes last wavelet (see above comment)
    bw_last = bw1_all.pop(-1)
    bw1_max_current = -1
    for i, bw1 in enumerate(bw1_all[::-1]):
        bw1_max_current = max(bw1, bw1_max_current)
        if bw1 < bw1_max_current:
            bw1_all[len(bw1_all) - i - 1] = bw1_max_current
    # don't allow it to inflate all other non-CQT's bws
    bw_last = min(bw_last, bw1_all[-1])
    # reinsert
    bw1_all.append(bw_last)

    # non-CQT redundancy adjustment, but maintain non-CQT <= CQT
    min_cqt_bw = min(bw1 for n1, bw1 in enumerate(bw1_all)
                     if psi1_f[n1]['is_cqt'])
    noncqt_bw_r_adj = 1.1
    for i, p in enumerate(psi1_f):
        if not p['is_cqt']:
            bw1_all[i] = min(int(np.floor(bw1_all[i] * noncqt_bw_r_adj)),
                             min_cqt_bw)

    # final check, ensure bw is non-increasing, non-zero, and <=N_psi//2
    assert np.diff(bw1_all).max() <= 0, bw1_all
    assert np.min(bw1_all) > 0, bw1_all
    assert np.max(bw1_all) <= N_psi // 2, (bw1_all, N_psi // 2)

    # second-order LP sum, LP threshold and condition, loop vars #############
    def compute_lp_slice(p):
        p = npy(p)
        p = np.abs(p)**2
        p[1:N//2] += p[N//2 + 1:][::-1]
        p[N//2 + 1:] = 0
        return p[:N//2 + 1]

    # compute *realized* second-order LP sum
    N = len(psi1_f[0][0])
    lp_sum2 = np.sum([compute_lp_slice(p2[0]) for p2 in psi2_f
                      if p2['j'] != 0], axis=0)

    # compute total second-order LP sum
    # don't threshold higher than the filterbank can satisfy
    # between first and last peaks -- *.9999 to avoid rare float == in cond_fn
    first_active_peak = psi2_f[-1]['peak_idx'][0]
    last_active_peak = [p2['peak_idx'][0] for p2 in psi2_f if p2['j'] != 0][0]
    lp_th = min(1,
                np.min(lp_sum2[first_active_peak:last_active_peak + 1])*.9999)

    lp_sum2_full = np.sum([compute_lp_slice(p2[0]) for p2 in psi2_f], axis=0)
    # adjust near DC, boundary discontinuity: below thresholding assumes the LP
    # minimum to be >= the minimum between first and last `psi2` peaks, which
    # isn't always true. It's possible that e.g. closest `psi2` to DC has
    # peak at 2, and bin 1 is below `lp_th` - then it's possible to get *fewer*
    # exclusions with lesser `bw1` (greater `e_th`), which shouldn't happen.
    # So, extend LP2 it according to the LP value at the last `psi2` peak -
    # this still accounts for the next-to-last `psi2` in thresholding.
    peak2_near_dc = psi2_f[-1]['peak_idx'][0]
    lp_sum2_full[1:peak2_near_dc] = lp_sum2_full[peak2_near_dc]

    # define `cond`
    def cond_fn(bw1, j1, j2, lp2, lp2_next, lp_th):
        return (
            j2_j1_cond(j2, j1) and
            (
                # filterbank up to the current psi2 tiles sufficiently,
                # but not without this psi2: keep
                (lp2[bw1] > lp_th and lp2_next[bw1] <= lp_th) or
                # filterbank up to the current psi2 does not tile
                # sufficiently, and it'll tile even less sufficiently
                # if we exclue this psi2: keep
                lp2[bw1] <= lp_th
            )
        )

    # validate `cond`: lesser `bw1` must produce more exclusions; ignore j2, j1
    lp2 = lp_sum2_full
    for n2, p2 in enumerate(psi2_f):
        lp2_next = lp2 - compute_lp_slice(p2[0])
        exclusions = np.array([int(not cond_fn(bw1, 1, 1, lp2, lp2_next, lp_th))
                               for bw1 in range(1, max(bw1_all) + 1)])
        lp2 = lp2_next
        if 0 not in exclusions:
            continue
        first_inclusion = np.where(exclusions == 0)[0][0]
        n_exclusions = np.sum(exclusions == 1)
        assert first_inclusion == n_exclusions, (first_inclusion, n_exclusions,
                                                 n2, exclusions)

    # main loop ##############################################################
    paths_include_all = []
    paths_exclude_smart = {'n2, n1': []}
    for n1, p1 in enumerate(psi1_f):
        j1 = p1['j']
        bw1 = bw1_all[n1]
        lp2 = lp_sum2_full

        n2_traversed = []
        paths_include = []
        for n2, p2 in enumerate(psi2_f):
            lp2_next = lp2 - compute_lp_slice(p2[0])
            j2 = p2['j']
            n2_traversed.append(n2)

            cond = cond_fn(bw1, j1, j2, lp2, lp2_next, lp_th)
            ## debug code --------------------------------------------------- #
            # from wavespin.visuals import plot, plotscat
            # # if j2 >= j1:
            # if (n2 == 12 and n1 == 85):
            #     _len = N//2**j2
            #     plot_fn = plot if _len > 128 else plotscat
            #     plot_fn(lp2[:_len], auto_xlims=0,
            #             title="{}, {} | cond={}".format(n2, n1, cond))
            #     plot_fn(lp2_next[:_len], show=1,
            #             vlines=(bw1, {'color': 'k', 'linewidth': 1}))
            #     break
            ## -------------------------------------------------------------- #

            if cond:
                # include this `psi2` and all ones with lower `xi2`, as
                # the spectrum of `|conv(x, psi1)|` extends all the way *up to*
                # this `psi2`.
                paths_include_all.extend([(_n2, n1) for _n2 in
                                          range(n2, len(psi2_f))])

                paths_include.extend([(_n2, n1) for _n2 in
                                      range(n2, len(psi2_f))])
                # already included all remaining `psi2`
                break
            lp2 = lp2_next
        paths_exclude = [(n2, n1) for n2 in range(len(psi2_f))
                         if (n2 in n2_traversed and
                             j2_j1_cond(psi2_f[n2]['j'], j1) and
                             n2 not in [n2n1[0] for n2n1 in paths_include])]
        paths_exclude_smart['n2, n1'].extend(paths_exclude)

    # sort for nicer visual: (2,0), (3,2), (2,1) -> (2,0), (2,1), (3,2)
    paths_exclude_smart['n2, n1'].sort(key=lambda x: int("%s%s" % x))

    # ensure `n1`s increment by 1 (no skips); utilizes the sort
    n2s = [n2n1[0] for n2n1 in paths_exclude_smart['n2, n1']]
    for n2 in n2s:
        n1s = [n2n1[1] for n2n1 in paths_exclude_smart['n2, n1']
               if n2n1[0] == n2]
        assert np.all(np.diff(n1s) == 1), "\nn2={}\nn1s={}\nbw1_all={}".format(
            n2, n1s, bw1_all)

    return paths_exclude_smart


def primitive_paths_exclude(psi1_f, psi2_f):
    """Include only j2 > j1."""
    pe = []
    for n2, p2 in enumerate(psi2_f):
        j2 = p2['j']
        for n1, p1 in enumerate(psi1_f):
            j1 = p1['j']
            if j2 == j1:  # not <= because < is already always excluded
                pe.append((n2, n1))

    paths_exclude_primitive = {'n2, n1': pe}
    return paths_exclude_primitive


def _e_th_from_e_loss(psi1_f, psi2_f, e_loss, level):
    # compute CQT redundancy #################################################
    # choose between first and last to avoid bell trimming by `analytic=True`
    # favor one toward lower freq in case of low Q

    # reference redundancy at which regression coefficients were obtained
    r_psi_ref = np.sqrt(.5)
    r_ref = r_psi_to_redundancy(r_psi_ref)

    # avoid 'trimmed bell' from `analytic=True`
    decayed_idxs = [i for i, p in enumerate(psi1_f)
                    if (p['is_cqt'] and
                        p['bw_idxs'][0][1] < len(p[0])//2)]
    # user-facing text
    # `analytic=False` may be a "solution" but a bad one, don't list it
    solutions = ("Solutions include: "
                 "lowering `Q`, raising `J`, raising `max_pad_factor`, "
                 "lowering `r_psi` (if it was raised from the default "
                 "value), or disabling via `smart_paths=0`.")
    if len(decayed_idxs) >= 2:
        # pick from middle, best estimate for "average" r
        test_idx = int(np.ceil(
            (decayed_idxs[0] + decayed_idxs[-1]) / 2
        ))
        p0, p1 = psi1_f[test_idx][0], psi1_f[test_idx - 1][0]
        r = compute_filter_redundancy(p0, p1)
        if r < .9*r_ref:
            warnings.warn("Detected low-redundancy filterbank, possibly "
                          "due to a distortion; Smart Paths doesn't "
                          "account for this regime. %s" % solutions)
    else:
        common = ("\nSmart Paths doesn't account for highly distorted "
                  "or non-CQT-majority filterbanks, and these are bad "
                  "for scattering anyway. %s" % solutions)
        if sum(p['is_cqt'] for p in psi1_f) >= 2:
            raise Exception("Couldn't find at least two first-order CQT "
                            "filters decayed before Nyquist; %s" % common)
        else:
            raise Exception("First-order filterbank doesn't have at least "
                            "two CQT filters; %s" % common)

    # regress `e_loss`, `r` onto `e_th` ######################################
    # randn, adv
    # e_loss = 0.1077*e_th - 0.004510*r + 0.0007170
    # e_loss = 0.4256*e_th - 0.02423*r  + 0.006062
    # e_th = 9.285*(e_loss + 0.004511*r - 0.0007170)
    # e_th = 2.350*(e_loss + 0.02423*r  - 0.006060)
    a_b_opts = {
        0: [(1, 1), (1, 1), (1, 1)],
        1: [(.5772, -0.0006734), (3.688, 0.01112),  (2.389, 0.07581)],
        2: [(1, 1),              (2.197, -0.01337), (1.634, -0.02292)],
        3: [(1, 1), (1, 1), (1, 1)]
    }
    if e_loss < .01:
        idx = 0
    elif e_loss < .05:
        idx = 1
    elif e_loss < .15:
        idx = 2
    a, b = a_b_opts[level][idx]
    e_th = a*e_loss + b

    assert e_th <= .5, e_th

    return e_th


#### Energy renormalization ##################################################
def energy_norm_filterbank_tm(psi1_f, psi2_f, phi_f, J, log2_T, normalize):
    """Energy-renormalize temporal filterbank; used by `base_frontend`.
    See `help(wavespin.scattering1d.refining.energy_norm_filterbank)`.
    """
    # in case of `trim_tm` for JTFS
    if phi_f is not None:
        phi = phi_f[0][0] if isinstance(phi_f[0], list) else phi_f[0]
    else:
        phi = None
    kw = dict(phi_f=phi, log2_T=log2_T, passes=3)

    # first order
    if 'energy' in normalize[0]:
        psi1_f0 = [p[0] for p in psi1_f]
        energy_norm_filterbank(psi1_f0, J=J[0], **kw)

    # second order
    if 'energy' in normalize[1]:
        psi2_f0 = [p[0] for p in psi2_f]
        scaling_factors2 = energy_norm_filterbank(psi2_f0, J=J[1], **kw)

        # apply unsubsampled scaling factors on subsampled
        for n2 in range(len(psi2_f)):
            for k in psi2_f[n2]:
                if isinstance(k, int) and k != 0:
                    psi2_f[n2][k] *= scaling_factors2[0][n2]


def energy_norm_filterbank_fr(psi1_f_fr_up, psi1_f_fr_dn, phi_f_fr,
                              J_fr, log2_F, sampling_psi_fr):
    """Energy-renormalize frequential filterbank; used by `base_frontend`.
    See `help(wavespin.scattering1d.refining.energy_norm_filterbank)`.
    """
    psi_id_max = max(psi_id for psi_id in psi1_f_fr_up
                     if isinstance(psi_id, int))
    psi_id_break = None
    for psi_id in range(psi_id_max + 1):
        psi_fs_up = psi1_f_fr_up[psi_id]
        psi_fs_dn = psi1_f_fr_dn[psi_id]

        if len(psi_fs_up) <= 3:  # possible with `sampling_psi_fr = 'exclude'`
            if psi_id == 0:
                raise Exception("largest scale filterbank must have >=4 filters")
            psi_id_break = psi_id
            break
        phi_f = None  # not worth the hassle to account for
        passes = 10  # can afford to do more without performance hit
        is_recalibrate = bool(sampling_psi_fr == 'recalibrate')
        scaling_factors = energy_norm_filterbank(psi_fs_up, psi_fs_dn, phi_f,
                                                 J_fr, log2_F,
                                                 is_recalibrate=is_recalibrate,
                                                 passes=passes)

    # we stopped normalizing when there were <= 3 filters ('exclude'),
    # but must still normalize the rest, so reuse factors from when we last had >3
    if psi_id_break is not None:
        for psi_id in range(psi_id_break, psi_id_max + 1):
            if psi_id not in psi1_f_fr_dn:
                continue
            for n1_fr in range(len(psi1_f_fr_dn[psi_id])):
                psi1_f_fr_up[psi_id][n1_fr] *= scaling_factors[0][n1_fr]
                psi1_f_fr_dn[psi_id][n1_fr] *= scaling_factors[1][n1_fr]


def energy_norm_filterbank(psi_fs0, psi_fs1=None, phi_f=None, J=None, log2_T=None,
                           is_recalibrate=False, warn=False, passes=3,
                           scaling_factors=None):
    """Rescale wavelets such that their frequency-domain energy sum
    (Littlewood-Paley sum) peaks at 2 for an analytic-only filterbank
    (e.g. time scattering for real inputs) or 1 for analytic + anti-analytic.
    This makes the filterbank energy non-expansive.

    Parameters
    ----------
    psi_fs0 : list[np.ndarray]
        Analytic filters (also spin up for frequential).

    psi_fs1 : list[np.ndarray] / None
        Anti-analytic filters (spin down). If None, filterbank is treated as
        analytic-only, and LP peaks are scaled to 2 instead of 1.

    phi_f : np.ndarray / None
        Lowpass filter. If `log2_T < J`, will exclude from computation as
        it will excessively attenuate low frequency bandpasses.

    J, log2_T : int, int
        See `phi_f`. For JTFS frequential scattering these are `J_fr, log2_F`.

    is_recalibrate : bool (default False)
        Used with `warn=True`, setting the threshold for number of peak
        duplicates to `3` if `True`, else `2`, since `'recalibrate'` with high
        `psi_id` and insufficient padding greatly increases duplication odds.

    warn : bool (default False)
        Whether to warn about found duplicate peaks degrading energy norm.
        Defaults to False since we already warn of degradation in frontend.

    passes : int
        Number of times to call this function recursively; see Algorithm.

    scaling_factors : None / dict[float]
        Used internally if `passes > 1`.

    Returns
    -------
    scaling_factors : None / dict[float]
        Used internally if `passes > 1`.


    Algorithm
    ---------
    Wavelets are scaled by maximum of *neighborhood* LP sum - precisely, LP sum
    spanning from previous to next peak location relative to wavelet of interest:
    `max(lp_sum[peak_idx[n + 1]:peak_idx[n - 1]])`. This reliably accounts for
    discretization artifacts, including the non-CQT portion.

    "Nyquist correction" is done for the highest frequency wavelet; since it
    has no "preceding" wavelet, it's its own right bound (analytic; left for
    anti-), which overestimates necessary rescaling and makes resulting LP sum
    peak above target for the *next* wavelet.

    Multiple "passes" are done to improve overall accuracy, as not all edge
    case behavior is accounted for in one go (which is possible but complicated);
    the computational burden is minimal.


    Nyquist correction note
    -----------------------
    Current implementation accounts for the entire frequency axis, namely
    also including the region between highest wavelet's peak frequency and
    Nyquist. Doing otherwise, with `analytic=False`, can result in significant
    overshoot in that region, especially with small `Q` (and undershoot with
    `analytic=True`). Yet, current implementation also attenuates the
    next-highest center frequency with `analytic=True`, which is of greater
    importance for low `Q`, but attenuates only slightly. The implementation
    favors ridding of over/undershoots, as for JTFS the Nyquist frequential
    wavelet bears non-trivial energy, and the temporal wavelet isn't used anyway.

    The unused implementation is kept for reference, via `ncv==2`.

    Ideally we'd weight frequency bins to prioritize attaining a tight frame at
    likelier frequencies (for second orders this means lower frequencies), also
    ignore unused wavelets (e.g. j2==0), but that's beyond the scope of this
    implementation.
    """
    from ..toolkit import compute_lp_sum

    def norm_filter(psi_fs, peak_idxs, lp_sum, n, s_idx=0):
        if ncv == 1 and (passes == 1 and n == 0):
            # don't tweak Nyquist again
            return

        # higher freq idx
        if n - 1 in peak_idxs:
            # midpoint
            pi0, pi1 = peak_idxs[n], peak_idxs[n - 1]
            if pi1 == pi0:
                # handle duplicate peaks
                lookback = 2
                while n - lookback in peak_idxs:
                    pi1 = peak_idxs[n - lookback]
                    if pi1 != pi0:
                        break
                    lookback += 1
            midpt = (pi0 + pi1) / 2
            # round closer to Nyquist
            a = (math.ceil(midpt) if s_idx == 0 else
                 math.floor(midpt))
        else:
            a = peak_idxs[n]

        # lower freq idx
        if n + 1 in peak_idxs:
            if n == 0 and nyquist_correction:
                b = a + 1 if s_idx == 1 else a - 1
            else:
                b = peak_idxs[n + 1]
        else:
            b = (None if s_idx == 1 else
                 1)  # exclude dc

        # peak duplicate
        if a == b:
            if s_idx == 1:
                b += 1
            else:
                b -= 1
        start, end = (a, b) if s_idx == 1 else (b, a)

        # include endpoint
        if end is not None:
            end += 1

        # if we're at endpoints, don't base estimate on single point
        if start is None:  # left endpoint
            end = max(end, 2)
        elif end is None:  # right endpoint
            start = min(start, len(lp_sum) - 1)
        elif end - start == 1:
            if start == 0:
                end += 1
            elif end == len(lp_sum) - 1:
                start -= 1

        # check `start` & `end` #####
        err_common_txt = ("Something likely went wrong before this function's "
                          "call (is Q too low or r_psi too great?)")
        Nhalf = len(psi_fs[n])//2
        # `end - 1` since that's [start:end] indexes in Python
        if ((s_idx == 0 and ((start is not None and start > Nhalf ) or
                             (end is None or end - 1 > Nhalf))) or
            (s_idx == 1 and ((start is None or start < Nhalf) or
                             (end is not None and end - 1 < Nhalf)))):
            raise Exception(("Found {} frequency peak in an {} filterbank! "
                             "(start, end, N//2 = {}, {}, {})\n{}"
                             ).format(
                                 "negative" if s_idx==0 else "positive",
                                 "analytic" if s_idx==0 else "anti-analytic",
                                 start, end, Nhalf, err_common_txt))

        # compute `lp_max` ####
        lp_max = lp_sum[start:end].max()

        # check `lp_max` ####
        if lp_max < 1e-7:
            warnings.warn(("Found very low LP sum peak while normalizing - "
                           "something's likely wrong with the filterbank."))
        elif lp_max == 0:  # no-cov
            raise Exception("LP sum peak is zero! " + err_common_txt)

        # normalize ####
        factor = np.sqrt(peak_target / lp_max)
        psi_fs[n] *= factor
        if n not in scaling_factors[s_idx]:
            scaling_factors[s_idx][n] = 1
        scaling_factors[s_idx][n] *= factor

    def correct_nyquist(psi_fs_all, peak_idxs, lp_sum):
        def _do_correction(start, end, s_idx=0):
            lp_max = lp_sum[start:end].max()
            factor = np.sqrt(peak_target / lp_max)

            if ncv == 1:
                ns = (0,)
            else:
                # first (Nyquist-nearest) psi rescaling may drive LP sum above
                # bound for second psi, since peak was taken only at itself,
                # so rescale both
                ns = (0, 1)

            for n in ns:
                psi_fs[n] *= factor
                scaling_factors[s_idx][n] *= factor

        if analytic_only:
            psi_fs = psi_fs_all
            # include endpoint
            if ncv == 1:
                start, end = peak_idxs[2], len(psi_fs[0]) // 2 + 1
            else:
                start, end = peak_idxs[2], peak_idxs[0] + 1
            _do_correction(start, end)
        else:
            for s_idx, psi_fs in enumerate(psi_fs_all):
                a = peak_idxs[s_idx][2]
                b = (len(psi_fs[0]) // 2 if ncv == 1 else
                     peak_idxs[s_idx][0])
                start, end = (a, b) if s_idx == 0 else (b, a)
                # include endpoint
                end += 1
                _do_correction(start, end, s_idx)

    # nyquist correction version
    # 1: account for region between highest freq peak and nyquist
    # 2: don't
    # currently unused, but worth keeping for clarity and future revisions
    ncv = 1

    # run input checks #######################################################
    assert len(psi_fs0) >= 4, (
        "must have at least 4 filters in filterbank (got %s) " % len(psi_fs0)
        + "try increasing J or Q")
    if psi_fs1 is not None:
        assert len(psi_fs0) == len(psi_fs1), (
            "analytic & anti-analytic filterbanks "
            "must have same number of filters")

    # determine whether to do Nyquist correction ##############################
    # we wish to check the CQT region but at the same time away from
    # Nyquist in case of `analytic=True` which does trimming that
    # `compute_filter_redundancy` doesn't account for
    mid = max(len(psi_fs0)//2, 1)
    r = compute_filter_redundancy(psi_fs0[mid - 1], psi_fs0[mid])
    # Do correction only if the filterbank is below this redundancy threshold
    # (empirically determined), since otherwise the direct estimate's accurate
    r_th = .35
    # skip on last pass
    nyquist_correction = bool(r < r_th and passes != 1)

    # execute ################################################################
    # as opposed to `analytic_and_anti_analytic`
    analytic_only = bool(psi_fs1 is None)
    peak_target = 2 if analytic_only else 1

    # store rescaling factors
    if scaling_factors is None:  # else means passes>1
        scaling_factors = {0: {}, 1: {}}

    # compute peak indices
    peak_idxs = {}
    if analytic_only:
        psi_fs_all = psi_fs0
        for n, psi_f in enumerate(psi_fs0):
            peak_idxs[n] = np.argmax(psi_f)
    else:
        psi_fs_all = (psi_fs0, psi_fs1)
        for s_idx, psi_fs in enumerate(psi_fs_all):
            peak_idxs[s_idx] = {}
            for n, psi_f in enumerate(psi_fs):
                peak_idxs[s_idx][n] = np.argmax(psi_f)

    # warn if there are 3 or more shared peaks
    pidxs_either = list((peak_idxs if analytic_only else peak_idxs[0]).values())
    th = 3 if is_recalibrate else 2  # at least one overlap likely in recalibrate
    if (any(pidxs_either.count(idx) >= th for idx in pidxs_either)
            and warn):  # no-cov
        pad_varname = "max_pad_factor" if analytic_only else "max_pad_factor_fr"
        warnings.warn(f"Found >={th} wavelets with same peak freq, most likely "
                      f"per too small `{pad_varname}`; energy norm may be poor")

    # ensure LP sum peaks at 2 (analytic-only) or 1 (analytic + anti-analytic)
    def get_lp_sum():
        if analytic_only:
            return compute_lp_sum(psi_fs0, phi_f, J, log2_T,
                                  fold_antianalytic=True)
        else:
            return (compute_lp_sum(psi_fs0, phi_f, J, log2_T) +
                    compute_lp_sum(psi_fs1))

    lp_sum = get_lp_sum()
    assert len(lp_sum) % 2 == 0, "expected even-length wavelets"
    if analytic_only:  # time scattering
        for n in range(len(psi_fs0)):
            norm_filter(psi_fs0, peak_idxs, lp_sum, n)
    else:  # frequential scattering
        for s_idx, psi_fs in enumerate(psi_fs_all):
            for n in range(len(psi_fs)):
                norm_filter(psi_fs, peak_idxs[s_idx], lp_sum, n, s_idx)

    if nyquist_correction:
        lp_sum = get_lp_sum()  # compute against latest
        correct_nyquist(psi_fs_all, peak_idxs, lp_sum)

    if passes == 1:
        return scaling_factors
    return energy_norm_filterbank(psi_fs0, psi_fs1, phi_f, J, log2_T,
                                  is_recalibrate, warn, passes - 1,
                                  scaling_factors)
