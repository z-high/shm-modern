"""Waveform filtering, signal processing, and 3C rotation/polarization SH commands."""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from shm_app.core.models import Trace
from shm_app.seismology.traveltime import sphere_diff
from shm_app.processing.filters import autocut_trace, filter_trace
from shm_app.processing.polarization import (
    apply_polarization_filter,
    compute_mdir,
    rotate_2d,
    rotate_3d_lqt,
    rotate_STS2_zne_to_uvw,
)
from shm_app.processing.trace_ops import (
    attenuate_trace,
    create_synthetic_trace,
    cut_trace,
    decimate_trace,
    demean_trace,
    despike_trace,
    detrend_trace,
    differentiate_trace,
    envelope_trace,
    hilbert_trace,
    instantaneous_phase_trace,
    integrate_trace,
    mend_trace,
    phase_weighted_stack,
    resample_trace,
    spectral_division_deconvolve,
    spiking_filter_levinson,
    sum_traces,
)


class _ProcessingCommandsMixin:
    """Filtering, time-series manipulation, synthetic generation, and 3C rotation commands."""

    def apply_filter_to_traces(
        self,
        filter_expr: str,
        indices: Optional[Sequence[int]] = None,
        mode: str = "AUTO",
        use_lpc: bool = False,
        demean_first: bool = False,
        visible_only: bool = False,
        autocut: bool = False,
        zero_phase: bool = False,
    ) -> int:
        target_indices = list(range(len(self.traces))) if indices is None else list(indices)
        count = 0
        for i in target_indices:
            if not (0 <= i < len(self.traces)):
                continue
            tr = self.traces[i]
            if visible_only and tr.hidden:
                continue
            if demean_first:
                tr = demean_trace(tr)
            self.traces[i] = getattr(sys.modules.get("shm_app.interpreter.sh_engine"), "filter_trace", filter_trace)(
                tr,
                filter_expr,
                mode=mode,
                use_lpc=use_lpc,
                config=self.config,
                autocut=autocut,
                zero_phase=zero_phase,
            )
            count += 1
        return count

    def invert_trace(self, trace_idx: int) -> bool:
        if not (0 <= trace_idx < len(self.traces)):
            return False
        tr = self.traces[trace_idx]
        tr.data = -tr.data
        return True

    def sort_traces(self, by: str = "distance") -> None:
        if by == "distance":
            p = self.current_params
            if p.lat != 0 or p.lon != 0:
                for tr in self.traces:
                    st = self.config.get_station(tr.station)
                    d, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = d
            self.traces.sort(key=lambda t: t.distance)
        else:
            self.traces.sort(key=lambda t: (t.station, t.channel3))


    def _cmd_filter(self, args: List[str], quals: Dict[str, str]) -> str:
        rem = list(args)
        mode_override = ""
        if len(rem) >= 2 and rem[0].upper() == "REC":
            mode_override = "REC"
            rem.pop(0)
        f_expr = rem[0] if rem else (self.loaded_filters[0] if self.loaded_filters else "WWSSN_SP")
        list_spec = quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        mode = (
            "REC"
            if ("REC" in quals or mode_override == "REC" or quals.get("MODE", "").upper() == "REC")
            else ("FFT" if "FFT" in quals else "AUTO")
        )
        use_lpc = "LPC" in quals
        autocut = "AUTOCUT" in quals
        zero_phase = ("ZEROPHASE" in quals or "ZERO" in quals)

        count = self.apply_filter_to_traces(
            f_expr,
            indices=idxs,
            mode=mode,
            use_lpc=use_lpc,
            autocut=autocut,
            zero_phase=zero_phase,
        )
        ac_note = " (with autocut)" if autocut else ""
        msg = f"Applied filter '{f_expr}'{ac_note} to {count} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_butter(self, args: List[str], quals: Dict[str, str]) -> str:
        if len(args) < 2:
            raise ValueError("Usage: BUTTER <LP|HP|BP|BS> <f_lo> [f_hi] [order]")
        ftype = args[0].upper()
        f1 = float(args[1])
        if ftype in ("LP", "HP"):
            order = int(args[2]) if len(args) > 2 else 3
            expr = f"BUTTER {ftype} {f1} {order}"
        else:
            f2 = float(args[2]) if len(args) > 2 else 5.0
            order = int(args[3]) if len(args) > 3 else 3
            expr = f"BUTTER {ftype} {f1} {f2} {order}"

        idxs = self.parse_trace_list(quals.get("LIST", "ALL"))
        autocut = "AUTOCUT" in quals
        zero_phase = ("ZEROPHASE" in quals or "ZERO" in quals)
        for i in idxs:
            self.traces[i] = getattr(sys.modules.get("shm_app.interpreter.sh_engine"), "filter_trace", filter_trace)(
                self.traces[i],
                expr,
                config=self.config,
                autocut=autocut,
                zero_phase=zero_phase,
            )
        ac_note = " (with autocut)" if autocut else ""
        msg = f"Applied Butterworth filter ({expr}){ac_note} to {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    @staticmethod
    def _looks_like_color(token: str) -> bool:
        t = token.strip().lower()
        if t.startswith("#") and len(t) in (4, 7, 9):
            return True
        return t in {
            "red", "blue", "green", "orange", "purple", "cyan", "magenta",
            "yellow", "teal", "navy", "crimson", "gold", "black", "gray",
            "grey", "lime", "brown", "pink", "violet", "indigo",
        }

    def _cmd_addfil(self, args: List[str], quals: Dict[str, str]) -> str:
        if "LIST" in quals:
            idxs = self.parse_trace_list(quals["LIST"])
        else:
            idxs = [i for i, t in enumerate(self.traces) if t.selected and not t.hidden]
        if not idxs:
            raise ValueError("ADDFIL requires selected traces (SEL <list>) or /LIST=<list>")

        rem = list(args)
        color = quals.get("COLOR", quals.get("COLOUR", ""))
        if not color and len(rem) >= 2 and self._looks_like_color(rem[-1]):
            color = rem.pop()

        f_expr = quals.get("FILTER", " ".join(rem).strip())
        clear_mode = ("CLEAR" in quals) or ("OFF" in quals) or (f_expr.upper() in ("NONE", "CLEAR", "OFF"))
        if not clear_mode and not f_expr:
            f_expr = self.loaded_filters[0] if self.loaded_filters else "WWSSN_SP"

        if clear_mode:
            for i in idxs:
                self.traces[i].overlay_filter_name = "NONE"
                self.traces[i].overlay_filter_color = ""
            msg = f"Cleared overlay filter on {len(idxs)} selected trace(s)."
            self.emit(msg)
            return msg

        eff_color = color or "#e11d48"
        mode = (
            "REC"
            if ("REC" in quals or quals.get("MODE", "").upper() == "REC")
            else ("FFT" if "FFT" in quals else "AUTO")
        )
        for i in idxs:
            tr = self.traces[i]
            prev_global = tr.filter_name
            filtered = getattr(sys.modules.get("shm_app.interpreter.sh_engine"), "filter_trace", filter_trace)(tr, f_expr, mode=mode, config=self.config)
            filtered.filter_name = prev_global
            filtered.overlay_filter_name = f_expr
            filtered.overlay_filter_color = eff_color
            filtered.filter_history = list(tr.filter_history) + [f_expr]
            self.traces[i] = filtered

        msg = f"Assigned overlay filter '{f_expr}' (color={eff_color}) to {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_autocut(self, args: List[str], quals: Dict[str, str]) -> str:
        settle_sec: Optional[float] = float(quals["SETTLE"]) if "SETTLE" in quals else None
        list_spec: Optional[str] = quals.get("LIST")

        rem = list(args)
        if rem:
            if len(rem) == 1:
                tok = rem[0]
                if tok.upper() in ("ALL", "SEL", "SELECTED", "*") or "," in tok or "-" in tok:
                    if list_spec is None:
                        list_spec = tok
                elif tok.isdigit() and 1 <= int(tok) <= len(self.traces):
                    if list_spec is None:
                        list_spec = tok
                else:
                    try:
                        settle_sec = float(tok)
                    except ValueError:
                        if list_spec is None:
                            list_spec = tok
            else:
                if list_spec is None:
                    list_spec = rem[0]
                if settle_sec is None:
                    settle_sec = float(rem[1])

        if list_spec is None:
            list_spec = "SEL"
        idxs = self.parse_trace_list(list_spec)
        if not idxs:
            msg = "No traces available for AUTOCUT."
            self.emit(msg)
            return msg

        zero_phase = ("ZEROPHASE" in quals or "ZERO" in quals)
        trimmed_samples = 0
        for i in idxs:
            before_len = self.traces[i].length
            self.traces[i] = autocut_trace(
                self.traces[i],
                settle_sec=settle_sec,
                zero_phase=zero_phase,
            )
            trimmed_samples += max(0, before_len - self.traces[i].length)

        msg = f"Autocut trimmed {trimmed_samples} sample(s) across {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_demean(self, args: List[str], quals: Dict[str, str]) -> str:
        if len(args) == 2 or len(args) > 3:
            raise ValueError("Usage: DEMEAN [trlist] [<lo_sec> <hi_sec>]")
        spec = args[0] if args else "ALL"
        lo = float(args[1]) if len(args) == 3 else None
        hi = float(args[2]) if len(args) == 3 else None
        idxs = self.parse_trace_list(spec)
        for i in idxs:
            self.traces[i] = demean_trace(self.traces[i], lo, hi)
        msg = f"Demeaned {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_trend(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = detrend_trace(self.traces[i])
        msg = f"Detrended {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_despike(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        crit = float(args[1]) if len(args) > 1 else 5.0
        total_spikes = 0
        for i in idxs:
            self.traces[i], scnt, _dcnt = despike_trace(self.traces[i], critmul=crit)
            total_spikes += scnt
        msg = f"Despiked {len(idxs)} trace(s) ({total_spikes} spikes removed)."
        self.emit(msg)
        return msg

    def _cmd_resample(self, args: List[str]) -> str:
        if not args:
            raise ValueError("RESAMPLE requires <new_dt> or <trlist> <new_dt>")
        if len(args) == 1:
            idxs = self.parse_trace_list("ALL")
            new_dt = float(args[0])
        else:
            idxs = self.parse_trace_list(args[0])
            new_dt = float(args[1])
        for i in idxs:
            self.traces[i] = resample_trace(self.traces[i], new_dt)
        msg = f"Resampled {len(idxs)} trace(s) to dt={new_dt} s."
        self.emit(msg)
        return msg

    def _cmd_decimate(self, args: List[str]) -> str:
        if not args:
            raise ValueError("DECIMATE requires <factor> or <trlist> <factor>")
        if len(args) == 1:
            idxs = self.parse_trace_list("ALL")
            fac = int(args[0])
        else:
            idxs = self.parse_trace_list(args[0])
            fac = int(args[1])
        for i in idxs:
            self.traces[i] = decimate_trace(self.traces[i], fac)
        msg = f"Decimated {len(idxs)} trace(s) by factor {fac}."
        self.emit(msg)
        return msg

    def _cmd_cut(self, args: List[str], quals: Dict[str, str]) -> str:
        if len(args) == 2:
            idxs = self.parse_trace_list("ALL")
            lo, hi = float(args[0]), float(args[1])
        elif len(args) >= 3:
            idxs = self.parse_trace_list(args[0])
            lo, hi = float(args[1]), float(args[2])
        else:
            raise ValueError("Usage: CUT [trlist] <lo_sec> <hi_sec>")
        for i in idxs:
            self.traces[i] = cut_trace(self.traces[i], lo, hi)
        msg = f"Cut {len(idxs)} trace(s) to [{lo:.2f}, {hi:.2f}] s."
        self.emit(msg)
        return msg

    def _cmd_shift(self, args: List[str]) -> str:
        if len(args) == 1:
            idxs = self.parse_trace_list("ALL")
            sec = float(args[0])
        elif len(args) >= 2:
            idxs = self.parse_trace_list(args[0])
            sec = float(args[1])
        else:
            return ""
        for i in idxs:
            self.traces[i].display_offset += sec
        msg = f"Shifted {len(idxs)} trace(s) by {sec:+.3f} s."
        self.emit(msg)
        return msg

    def _cmd_int(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = integrate_trace(self.traces[i])
        msg = f"Integrated {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_derive(self, args: List[str], quals: Dict[str, str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        mode = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1
        for i in idxs:
            self.traces[i] = differentiate_trace(self.traces[i], mode=mode)
        msg = f"Differentiated {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_hilbert(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = hilbert_trace(self.traces[i])
        msg = f"Computed Hilbert transform on {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_envelope(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = envelope_trace(self.traces[i])
        msg = f"Computed analytic envelope on {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_sphase(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = instantaneous_phase_trace(self.traces[i])
        msg = f"Computed instantaneous phase on {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_mult(self, args: List[str], factor_override: Optional[float] = None) -> str:
        if factor_override is not None:
            idxs = self.parse_trace_list(args[0] if args else "ALL")
            fac = factor_override
        elif len(args) == 1:
            idxs = self.parse_trace_list("ALL")
            fac = float(args[0])
        elif len(args) >= 2:
            idxs = self.parse_trace_list(args[0])
            fac = float(args[1])
        else:
            return ""
        for i in idxs:
            self.traces[i].data = self.traces[i].data * fac
        msg = f"Multiplied {len(idxs)} trace(s) by {fac:g}."
        self.emit(msg)
        return msg

    def _cmd_sum(self, args: List[str], quals: Dict[str, str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        trs = [self.traces[i] for i in idxs]
        s_tr = sum_traces(trs, normalize_weights=("NONORM" not in quals))
        if s_tr is not None:
            self.traces.append(s_tr)
            msg = f"Appended SUM trace from {len(trs)} trace(s)."
            self.emit(msg)
            return msg
        msg = "No traces to sum."
        self.emit(msg)
        return msg

    def _cmd_pws(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        power = float(args[1]) if len(args) > 1 else 2.0
        trs = [self.traces[i] for i in idxs]
        p_tr = phase_weighted_stack(trs, power=power)
        if p_tr is not None:
            self.traces.append(p_tr)
            msg = f"Appended Phase-Weighted Stack (PWS, power={power}) from {len(trs)} trace(s)."
            self.emit(msg)
            return msg
        msg = "No traces for PWS."
        self.emit(msg)
        return msg

    def _cmd_mend(self, args: List[str]) -> str:
        if len(args) < 2:
            raise ValueError("Usage: MEND <idx_lo> <idx_hi> [trlist]")
        i0, i1 = int(args[0]), int(args[1])
        idxs = self.parse_trace_list(args[2] if len(args) > 2 else "ALL")
        for i in idxs:
            self.traces[i] = mend_trace(self.traces[i], i0, i1)
        msg = f"Mended samples [{i0}..{i1}] on {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_attenuate(self, args: List[str]) -> str:
        t_star = float(args[0]) if args else 1.0
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        for i in idxs:
            self.traces[i] = attenuate_trace(self.traces[i], t_star=t_star)
        msg = f"Applied attenuation t*={t_star} s to {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_spiking(self, args: List[str]) -> str:
        flen = int(args[0]) if args else 80
        prew = float(args[1]) if len(args) > 1 else 0.01
        if not self.traces:
            msg = "No trace available for spiking filter."
            self.emit(msg)
            return msg
        tr = self.traces[0]
        f = spiking_filter_levinson(tr.data, filter_len=flen, prewhitening=prew)
        out = tr.copy(copy_picks=False)
        out.data = np.convolve(tr.data, f, mode="same")
        out.station = "SPIK"
        self.traces.append(out)
        msg = f"Computed spiking deconvolution trace (filter_len={flen})."
        self.emit(msg)
        return msg

    def _cmd_deconv(self, args: List[str]) -> str:
        if len(self.traces) < 2:
            raise ValueError("DECONV requires at least 2 traces (numerator and denominator)")
        wl = float(args[0]) if args else 0.01
        res = spectral_division_deconvolve(self.traces[0], self.traces[1], water_level=wl)
        res.station = "DECV"
        self.traces.append(res)
        msg = "Appended water-level deconvolved trace DECV."
        self.emit(msg)
        return msg

    def _cmd_create(self, args: List[str]) -> str:
        kind = args[0] if args else "KUEPPER"
        dur = float(args[1]) if len(args) > 1 else 60.0
        dt = float(args[2]) if len(args) > 2 else 0.05
        pos = float(args[3]) if len(args) > 3 else 15.0
        ampl = float(args[4]) if len(args) > 4 else 100.0
        width = float(args[5]) if len(args) > 5 else 2.0
        tr = create_synthetic_trace(kind, length_sec=dur, dt=dt, pos_sec=pos, ampl=ampl, width=width)
        self.traces.append(tr)
        msg = f"Created synthetic {kind.upper()} trace ({tr.length} samples)."
        self.emit(msg)
        return msg

    def _cmd_rot2(self, args: List[str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        vis = self.visible_traces()
        tr_n: Optional[Trace] = None
        tr_e: Optional[Trace] = None
        seen_sta: List[str] = []
        for t in vis:
            st = t.station.strip().upper()
            if st and st not in seen_sta:
                seen_sta.append(st)
        for st in seen_sta:
            tn = next((t for t in vis if t.station.strip().upper() == st and t.comp == "N"), None)
            te = next((t for t in vis if t.station.strip().upper() == st and t.comp == "E"), None)
            if tn is not None and te is not None:
                tr_n, tr_e = tn, te
                break
        if tr_n is None or tr_e is None:
            tr_n = next((t for t in vis if t.comp == "N"), None)
            tr_e = next((t for t in vis if t.comp == "E"), None)
        if tr_n is None or tr_e is None:
            raise ValueError("Need horizontal (N, E) traces for 2D rotation")
        r_tr, t_tr = rotate_2d(tr_n, tr_e, az)
        self.traces.extend([r_tr, t_tr])
        msg = f"Rotated ({tr_n.station} N,E) by {az:.1f} deg -> appended R, T traces."
        self.emit(msg)
        return msg

    def _cmd_rot3(self, args: List[str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        tr_z, tr_n, tr_e = self._find_zne_triplet()
        if len(args) > 1:
            inc = float(args[1])
        elif self.current_params.incidence > 0:
            inc = self.current_params.incidence
        elif tr_z.inci > 0:
            inc = tr_z.inci
        else:
            inc = 20.0
        l_tr, q_tr, t_tr = rotate_3d_lqt(tr_z, tr_n, tr_e, az, inc)
        self.traces.extend([l_tr, q_tr, t_tr])
        msg = f"Rotated 3D ({tr_z.station} Z,N,E) by az={az:.1f}, inci={inc:.1f} -> appended L, Q, T traces."
        self.emit(msg)
        return msg

    def _cmd_uvw(self, args: List[str], quals: Dict[str, str]) -> str:
        inv = "INV" in quals
        if inv:
            vis = self.visible_traces()
            tu = next((t for t in vis if t.comp == "U"), None)
            tv = next((t for t in vis if t.comp == "V"), None)
            tw = next((t for t in vis if t.comp == "W"), None)
            if tu is not None and tv is not None and tw is not None:
                tr_1, tr_2, tr_3 = tu, tv, tw
            else:
                tr_1, tr_2, tr_3 = self._find_zne_triplet()
        else:
            tr_1, tr_2, tr_3 = self._find_zne_triplet()
        t1, t2, t3 = rotate_STS2_zne_to_uvw(tr_1, tr_2, tr_3, inverse=inv)
        self.traces.extend([t1, t2, t3])
        msg = f"Transformed STS-2 {'UVW->ZNE' if inv else 'ZNE->UVW'}."
        self.emit(msg)
        return msg

    def _cmd_mdir(self, args: List[str]) -> str:
        if len(args) == 1 or len(args) > 2:
            raise ValueError("Usage: MDIR [<lo_sec> <hi_sec>]")
        tr_z, tr_n, tr_e = self._find_zne_triplet()
        lo = float(args[0]) if len(args) == 2 else (self.time_window[0] if self.time_window else None)
        hi = float(args[1]) if len(args) == 2 else (self.time_window[1] if self.time_window else None)
        res = compute_mdir(tr_z, tr_n, tr_e, lo, hi)
        self.last_mdir_result = res
        self.current_params.azimuth = res["azimuth"]
        self.current_params.incidence = res["incidence"]
        for t in (tr_z, tr_n, tr_e):
            t.inci = res["incidence"]
        msg = (
            f"MDIR Polarization ({tr_z.station}): Backazimuth = {res['azimuth']:.1f} deg, "
            f"Incidence = {res['incidence']:.1f} deg, Rectilinearity = {res['rectilinearity']:.3f}"
        )
        self.emit(msg)
        return msg

    def _cmd_polfil(self, args: List[str]) -> str:
        win = float(args[0]) if args else 1.0
        power = float(args[1]) if len(args) > 1 else 2.0
        mode = int(args[2]) if len(args) > 2 else 1
        tr_z, tr_n, tr_e = self._find_zne_triplet()
        pz, pn, pe = apply_polarization_filter(
            tr_z,
            tr_n,
            tr_e,
            coherence_win_sec=win,
            linearity_power=power,
            mode=mode,
            baz_deg=self.current_params.azimuth,
        )
        self.traces.extend([pz, pn, pe])
        msg = f"Applied polarization filter (mode={mode}, win={win}s, power={power}) -> appended 3 traces."
        self.emit(msg)
        return msg


    def _find_zne_triplet(self, prefer_station: Optional[str] = None) -> Tuple[Trace, Trace, Trace]:
        vis = self.visible_traces()
        if prefer_station:
            pref = prefer_station.strip().upper()
            tz = next((t for t in vis if t.station.strip().upper() == pref and t.comp == "Z"), None)
            tn = next((t for t in vis if t.station.strip().upper() == pref and t.comp == "N"), None)
            te = next((t for t in vis if t.station.strip().upper() == pref and t.comp == "E"), None)
            if tz and tn and te:
                return tz, tn, te
            raise ValueError(f"Station '{prefer_station}' requires 3-component (Z, N, E) traces in workspace")
        candidates: List[str] = []
        for t in vis:
            st = t.station.strip().upper()
            if st and st not in candidates:
                candidates.append(st)
        for st in candidates:
            tz = next((t for t in vis if t.station.strip().upper() == st and t.comp == "Z"), None)
            tn = next((t for t in vis if t.station.strip().upper() == st and t.comp == "N"), None)
            te = next((t for t in vis if t.station.strip().upper() == st and t.comp == "E"), None)
            if tz and tn and te:
                return tz, tn, te
        tz = next((t for t in vis if t.comp == "Z"), None)
        tn = next((t for t in vis if t.comp == "N"), None)
        te = next((t for t in vis if t.comp == "E"), None)
        if tz and tn and te:
            return tz, tn, te
        raise ValueError("Operation requires 3-component (Z, N, E) traces in workspace")
