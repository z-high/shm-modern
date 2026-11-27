"""Phase onset picking (Baer-Kradolfer + AIC, array cross-correlation) and waveform QC."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time

def correlate_traces(
    ref_tr: Trace,
    target_tr: Trace,
    t_ref_lo: float,
    t_ref_hi: float,
    max_lag_sec: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute zero-mean normalized cross-correlation between a reference window [t_ref_lo, t_ref_hi]
    on ref_tr and target_tr matching mt_corr in shcorr.c.

    Returns (lags_sec, corr_coeffs, best_lag_sec, max_corr_coeff).
    """
    dt = ref_tr.delta if ref_tr.delta > 0 else 0.05
    i0 = max(0, int(round(min(t_ref_lo, t_ref_hi) / dt)))
    i1 = min(ref_tr.length, int(round(max(t_ref_lo, t_ref_hi) / dt)) + 1)
    if i1 <= i0 + 2:
        i0, i1 = 0, min(ref_tr.length, 100)

    x = ref_tr.data[i0:i1] - np.mean(ref_tr.data[i0:i1])
    norm_x = float(np.linalg.norm(x))
    if norm_x <= 1e-30:
        return np.array([0.0]), np.array([0.0]), 0.0, 0.0

    win_len = x.size
    max_lag_samps = max(1, int(round(max_lag_sec / dt)))
    lags_samps = np.arange(-max_lag_samps, max_lag_samps + 1)
    lags_sec = lags_samps * dt
    corrs = np.zeros(lags_samps.size, dtype=np.float64)

    # Center sample on target_tr corresponding to i0 on ref_tr (accounting for start_epoch difference)
    epoch_i0 = ref_tr.start_epoch + ref_tr.display_offset + i0 * dt
    j0_center = int(round((epoch_i0 - (target_tr.start_epoch + target_tr.display_offset)) / dt))

    has_overlap = False
    for idx, lag in enumerate(lags_samps):
        j_start = j0_center + int(lag)
        j_end = j_start + win_len
        if j_start < 0 or j_end > target_tr.length:
            continue
        y = target_tr.data[j_start:j_end] - np.mean(target_tr.data[j_start:j_end])
        norm_y = float(np.linalg.norm(y))
        if norm_y > 1e-30:
            corrs[idx] = float(np.dot(x, y)) / (norm_x * norm_y)
            has_overlap = True

    if not has_overlap or not np.any(corrs != 0.0):
        return lags_sec, corrs, 0.0, 0.0

    best_idx = int(np.argmax(corrs))
    best_lag = float(lags_sec[best_idx])
    best_val = float(corrs[best_idx])

    # Parabolic sub-sample refinement around best_idx
    if 0 < best_idx < corrs.size - 1:
        y_m, y_0, y_p = corrs[best_idx - 1], corrs[best_idx], corrs[best_idx + 1]
        denom = y_m - 2.0 * y_0 + y_p
        if denom < -1e-12:
            delta_s = 0.5 * (y_m - y_p) / denom
            best_lag += delta_s * dt

    return lags_sec, corrs, best_lag, best_val


def corrpick_array(
    traces: List[Trace],
    ref_trace_idx: int = 0,
    phase_name: str = "P",
    win_pre_sec: float = 1.0,
    win_post_sec: float = 3.0,
    max_lag_sec: float = 2.5,
    min_corr: float = 0.5,
) -> List[PhasePick]:
    """Automatically pick phase_name across all array traces by cross-correlating against ref_trace_idx."""
    if not traces or not (0 <= ref_trace_idx < len(traces)):
        return []
    ref_tr = traces[ref_trace_idx]
    ref_pk = ref_tr.get_pick(phase_name)
    if ref_pk is None or not ref_pk.onset:
        return []

    ref_rel_t = ref_pk.onset_epoch - ref_tr.start_epoch
    t_lo = max(0.0, ref_rel_t - win_pre_sec)
    t_hi = min(ref_tr.duration, ref_rel_t + win_post_sec)

    created: List[PhasePick] = []
    for idx, tr in enumerate(traces):
        if idx == ref_trace_idx:
            created.append(ref_pk)
            continue
        _lags, _corrs, best_lag, max_c = correlate_traces(ref_tr, tr, t_lo, t_hi, max_lag_sec=max_lag_sec)
        if max_c >= min_corr:
            new_pk = PhasePick(
                name=phase_name,
                onset=format_sh_time(ref_pk.onset_epoch + best_lag),
                spec=ref_pk.spec,
                sign=ref_pk.sign,
                source="a",
                filter_name=tr.filter_name,
                station=tr.station,
                comp=tr.comp,
                chan1=tr.chan1,
                chan2=tr.chan2,
            )
            if max_c >= 0.90:
                new_pk.set_quality(1)
            elif max_c >= 0.80:
                new_pk.set_quality(2)
            elif max_c >= 0.65:
                new_pk.set_quality(3)
            elif max_c >= 0.50:
                new_pk.set_quality(4)
            else:
                new_pk.set_quality(6)
            tr.add_or_replace_pick(new_pk)
            created.append(new_pk)
    return created


def pick_onset_baer_kradolfer(
    tr: Trace,
    t_lo: Optional[float] = None,
    t_hi: Optional[float] = None,
    thresh_sigma: float = 3.5,
) -> Optional[PhasePick]:
    """Automatic phase onset picker combining the Baer & Kradolfer (1987) characteristic
    function with causal STA/LTA triggering and Maeda (1985) AIC onset refinement.
    """
    dt = tr.delta if tr.delta > 0 else 0.05
    if t_lo is not None and t_hi is not None:
        i0 = max(0, int(round(min(t_lo, t_hi) / dt)))
        i1 = min(tr.length, int(round(max(t_lo, t_hi) / dt)) + 1)
    else:
        i0, i1 = 0, tr.length

    seg = np.asarray(tr.data[i0:i1], dtype=np.float64)
    if seg.size < 10:
        return None
    if not np.all(np.isfinite(seg)):
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

    y = seg - float(np.mean(seg))
    peak_abs = float(np.max(np.abs(y)))
    if peak_abs <= 1e-30:
        return None

    dy = np.diff(y, prepend=y[0])
    y2_sum = float(np.sum(y * y))
    dy2_sum = float(np.sum(dy * dy))
    r_ratio = dy2_sum / y2_sum if y2_sum > 1e-30 else 1.0
    cf = (y * y) + (dy * dy) / max(1e-12, r_ratio)

    n = cf.size
    # Check if trace is essentially noise-free synthetic (pre-signal quiet)
    n_init = max(5, min(50, n // 4))
    rms_init = math.sqrt(float(np.mean(y[:n_init] ** 2)))
    is_quiet_synthetic = rms_init < 1e-4 * peak_abs

    if is_quiet_synthetic or n < 40:
        mu0 = float(np.mean(cf[:n_init]))
        sig0 = max(1e-12, float(np.std(cf[:n_init])))
        cf_thr = max(mu0 + thresh_sigma * sig0, 1e-4 * float(np.max(cf)))
        hits = np.where(cf[n_init:] >= cf_thr)[0]
        onset_idx = int(hits[0] + n_init) if hits.size > 0 else int(np.argmax(cf))
    else:
        # Causal STA/LTA on Baer-Kradolfer characteristic function cf
        n_sta = max(5, min(n // 8, int(round(1.5 / dt))))
        n_lta = max(n_sta * 4, min(n // 3, int(round(20.0 / dt))))
        if n_lta + n_sta >= n:
            n_sta = max(3, n // 10)
            n_lta = max(n_sta * 3, n // 3)

        csum = np.concatenate(([0.0], np.cumsum(cf)))
        # sta[j] = mean(cf[j + n_lta - n_sta : j + n_lta]), lta[j] = mean(cf[j : j + n_lta - n_sta])
        n_bg = max(1, n_lta - n_sta)
        sta = (csum[n_lta:] - csum[n_lta - n_sta : -n_sta]) / float(n_sta)
        lta = (csum[n_lta - n_sta : -n_sta] - csum[: -n_lta]) / float(n_bg)
        # Floor LTA slightly to avoid division by tiny quiet pauses
        lta_floor = max(1e-30, 1e-4 * float(np.median(cf[cf > 0])) if np.any(cf > 0) else 1e-30)
        amp_ratio = np.sqrt(sta / np.maximum(lta, lta_floor))

        peak_ratio = float(np.max(amp_ratio))
        # Sustained trigger: require amplitude ratio above threshold AND a meaningful fraction of peak
        trig_thr = max(2.2, 0.85 * float(thresh_sigma))
        eff_thr = max(trig_thr, 0.30 * peak_ratio) if peak_ratio > 6.0 else trig_thr
        # Require at least min_hold consecutive samples above threshold to reject 1-sample glitches
        min_hold = max(2, min(n_sta // 2, int(round(0.4 / dt))))
        above = (amp_ratio >= eff_thr).astype(np.int32)
        if min_hold > 1 and above.size >= min_hold:
            run_sum = np.convolve(above, np.ones(min_hold, dtype=np.int32), mode="valid")
            sustained = np.where(run_sum >= min_hold)[0]
        else:
            sustained = np.where(above > 0)[0]

        if sustained.size > 0:
            trig_idx = int(sustained[0]) + n_lta - n_sta
        else:
            trig_idx = int(np.argmax(amp_ratio)) + n_lta - n_sta

        # Maeda (1985) vectorized O(N) AIC onset refinement in a window around trig_idx
        pre_pad = min(trig_idx, max(n_sta * 2, int(round(10.0 / dt))))
        post_pad = min(n - trig_idx, max(n_sta * 2, int(round(6.0 / dt))))
        w0 = max(0, trig_idx - pre_pad)
        w1 = min(n, trig_idx + post_pad)
        wseg = y[w0:w1]
        wn = wseg.size
        margin = max(6, min(wn // 6, int(round(0.5 / dt))))
        if wn >= 2 * margin + 4:
            cs1 = np.cumsum(wseg)
            cs2 = np.cumsum(wseg * wseg)
            ks = np.arange(margin, wn - margin, dtype=np.int64)
            # Variance of wseg[:k]
            m1 = cs1[ks - 1] / ks
            v1 = np.maximum(cs2[ks - 1] / ks - m1 * m1, 1e-30)
            # Variance of wseg[k:]
            rem = wn - ks
            s1_r = cs1[-1] - cs1[ks - 1]
            s2_r = cs2[-1] - cs2[ks - 1]
            m2 = s1_r / rem
            v2 = np.maximum(s2_r / rem - m2 * m2, 1e-30)
            aic = ks * np.log(v1) + (rem - 1.0) * np.log(v2)
            onset_idx = w0 + int(ks[int(np.argmin(aic))])
        else:
            onset_idx = trig_idx

    onset_idx = max(0, min(n - 1, onset_idx))

    # Determine first-motion polarity sign (+1 compression 'c', -1 dilatation 'd')
    look = y[onset_idx : min(n, onset_idx + max(4, int(round(0.3 / dt))))]
    ext_idx = int(np.argmax(np.abs(look))) if look.size > 0 else 0
    sign_val = 1 if (look.size > 0 and look[ext_idx] >= 0) else -1

    # Amplitude SNR: post-onset signal RMS vs pre-onset noise RMS
    pre_len = max(5, min(onset_idx, int(round(20.0 / dt))))
    post_len = max(5, min(n - onset_idx, int(round(5.0 / dt))))
    if pre_len > 0 and post_len > 0:
        pre_rms = math.sqrt(float(np.mean(y[onset_idx - pre_len : onset_idx] ** 2)))
        post_rms = math.sqrt(float(np.mean(y[onset_idx : onset_idx + post_len] ** 2)))
        snr = post_rms / max(pre_rms, 1e-12 * peak_abs)
    else:
        snr = 1.0
    snr = float(min(999.0, max(0.0, snr)))

    onset_ep = tr.start_epoch + (i0 + onset_idx) * dt
    pk = PhasePick(
        name="P",
        onset=format_sh_time(onset_ep),
        spec="i" if snr >= 5.0 else "e",
        sign=sign_val,
        source="a",
        snr=snr,
        filter_name=tr.filter_name,
        station=tr.station,
        comp=tr.comp,
        chan1=tr.chan1,
        chan2=tr.chan2,
    )
    # SHM quality convention: 1 = best (weight 9), 2 = weight 4, 3 = weight 3, 4 = weight 2, 6 = weight 1
    if snr >= 12.0:
        pk.set_quality(1)
    elif snr >= 6.0:
        pk.set_quality(2)
    elif snr >= 3.5:
        pk.set_quality(3)
    elif snr >= 2.2:
        pk.set_quality(4)
    else:
        pk.set_quality(6)
    return pk


def assess_trace_quality(traces: Sequence[Trace]) -> List[Dict[str, Any]]:
    """Evaluate waveform health across traces to flag dead/flat, non-finite, or chronically spiking sensors.

    Returns a list of dicts (one per trace) with keys:
      station, channel, ok (bool), flags (list[str]), pre_rms, peak_abs, spike_window_frac
    """
    reports: List[Dict[str, Any]] = []
    for tr in traces:
        flags: List[str] = []
        data = np.asarray(tr.data, dtype=np.float64)
        if data.size < 10:
            reports.append({
                "station": tr.station,
                "channel": tr.channel3,
                "ok": False,
                "flags": ["EMPTY"],
                "pre_rms": 0.0,
                "peak_abs": 0.0,
                "spike_window_frac": 0.0,
            })
            continue

        if not np.all(np.isfinite(data)):
            flags.append("NON_FINITE")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        y = data - float(np.mean(data))
        peak_abs = float(np.max(np.abs(y)))
        std_all = float(np.std(y))
        if peak_abs <= 1e-12 or std_all <= 1e-12:
            flags.append("FLAT")

        # Chronic periodic/impulsive spiking detector: split trace into ~10s blocks and check
        # how many blocks contain sharp high-crest spikes (peak / robust MAD > 8.0) across the trace.
        dt = tr.delta if tr.delta > 0 else 0.05
        win_n = max(40, int(round(10.0 / dt)))
        n_wins = y.size // win_n
        spike_frac = 0.0
        pre_rms = std_all
        if n_wins >= 6 and "FLAT" not in flags:
            # First-difference emphasizes sharp instrumental spikes over smooth seismic waves
            dy = np.diff(y, prepend=y[0])
            spiky_wins = 0
            win_rms = []
            for wi in range(n_wins):
                seg_w = dy[wi * win_n : (wi + 1) * win_n]
                med_w = float(np.median(seg_w))
                mad_w = float(np.median(np.abs(seg_w - med_w)))
                pk_w = float(np.max(np.abs(seg_w - med_w)))
                win_rms.append(float(np.std(y[wi * win_n : (wi + 1) * win_n])))
                if mad_w > 1e-12 and (pk_w / (1.4826 * mad_w)) >= 8.5:
                    spiky_wins += 1
            spike_frac = spiky_wins / float(n_wins)
            pre_rms = float(np.median(win_rms[: max(1, n_wins // 4)]))
            # If >= 60% of 10-second windows across the entire recording have sharp impulse spikes,
            # the channel is chronically corrupted by periodic glitches (e.g. timing/digitizer spikes).
            if spike_frac >= 0.60:
                flags.append("SPIKING")

        reports.append({
            "station": tr.station,
            "channel": tr.channel3,
            "ok": len(flags) == 0,
            "flags": flags,
            "pre_rms": round(pre_rms, 3),
            "peak_abs": round(peak_abs, 3),
            "spike_window_frac": round(spike_frac, 3),
        })
    return reports

