"""Interactive and scriptable SH Command Interpreter (shc) supporting all SH waveform,
filtering, array, location, polarization, spectral, I/O, and macro commands.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import FitDiagnosticResult
from shm_app.core.models import AnalysisParamSet, Trace
from shm_app.io.evt import format_ev2view_summary
from shm_app.processing.filters import filter_trace  # kept on module for mock.patch compatibility
from shm_app.seismology.array_loc import SlownessCalibrationTable
from shm_app.seismology.traveltime import get_ttt_manager
from shm_app.interpreter.sh_cmds_analysis import _AnalysisCommandsMixin
from shm_app.interpreter.sh_cmds_io import _IOCommandsMixin
from shm_app.interpreter.sh_cmds_processing import _ProcessingCommandsMixin
from shm_app.interpreter.sh_parser import (
    _IO_PATH_COMMANDS,
    _KNOWN_QUALIFIERS,
    _safe_eval_math_expr,
    is_qualifier_token,
    split_quoted_tokens,
    split_statements,
    tokenize_command,
)


class SHCommandEngine(_IOCommandsMixin, _ProcessingCommandsMixin, _AnalysisCommandsMixin):
    """Executes SH command lines and .SHC scripts on a workspace of Traces and Parameter Sets."""

    def __init__(
        self,
        config: Optional[SHMConfig] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or get_config()
        self.ttt = get_ttt_manager(self.config.inputs_dir)
        self.output_callback = output_callback

        # Active workspace state
        self.traces: List[Trace] = []
        self.param_sets: List[AnalysisParamSet] = [
            AnalysisParamSet(),
            AnalysisParamSet(),
            AnalysisParamSet(),
        ]
        self.active_set_index: int = 0  # 0, 1, or 2
        self.norm_mode: str = "C"  # 'C' (Constant), 'A' (Auto), 'AW' (Auto Window), 'SW' (Single Window)
        self.global_zoom: float = 1.0
        self.time_window: Optional[Tuple[float, float]] = None  # relative (t_lo, t_hi) seconds
        self.view_window: Optional[Tuple[float, float]] = None  # view zoom bounds set via STW/DTW/TRESET
        self._view_window_version: int = 0
        self.title: str = "SeismicHandler Motif (Modernized)"
        self.loaded_filters: List[str] = []
        self.symbols: Dict[str, str] = {
            "SH_ROOT": self.config.shm_root,
            "PI": str(math.pi),
            "DS": "0",
        }
        self.quit_requested: bool = False
        self.last_fk_result: Optional[Dict[str, Any]] = None
        self.last_mdir_result: Optional[Dict[str, float]] = None
        self.last_fit_diagnostic: Optional[FitDiagnosticResult] = None
        self.fit_diagnostic_callback: Optional[Callable[[FitDiagnosticResult], None]] = None
        self.calib_table: Optional[SlownessCalibrationTable] = None
        self.last_focal_mechanism: Optional[Dict[str, Any]] = None
        self.last_brune_result: Optional[Dict[str, Any]] = None
        self.last_surfmag_result: Optional[Dict[str, Any]] = None
        self.last_identify_results: List[Dict[str, Any]] = []
        self.last_stationxml_inventory: Any = None

    @property
    def current_params(self) -> AnalysisParamSet:
        return self.param_sets[self.active_set_index]

    @property
    def params(self) -> AnalysisParamSet:
        return self.param_sets[self.active_set_index]

    @params.setter
    def params(self, value: AnalysisParamSet) -> None:
        self.param_sets[self.active_set_index] = value

    def _emit_fit_diagnostic(self, diag: Optional[FitDiagnosticResult]) -> None:
        self.last_fit_diagnostic = diag
        if diag is not None and self.fit_diagnostic_callback is not None:
            self.fit_diagnostic_callback(diag)

    def visible_traces(self) -> List[Trace]:
        return [t for t in self.traces if not t.hidden]

    def emit(self, msg: str) -> None:
        if self.output_callback is not None:
            self.output_callback(msg)
        else:
            print(msg)

    def parse_trace_list(self, spec: str) -> List[int]:
        """Parse an SH trace list specification (e.g. 'ALL', '1-5', '1,3,5', 'SEL') into 0-based indices
        into self.traces.
        """
        s = spec.strip().upper()
        n = len(self.traces)
        if n == 0:
            return []
        if not s or s in ("ALL", "*"):
            return [i for i, t in enumerate(self.traces) if not t.hidden]
        if s == "ALL_INC_HIDDEN":
            return list(range(n))
        if s in ("SEL", "SELECTED"):
            sel = [i for i, t in enumerate(self.traces) if t.selected and not t.hidden]
            return sel if sel else [i for i, t in enumerate(self.traces) if not t.hidden]

        indices: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo = int(lo_s) if lo_s.strip() else 1
                hi = int(hi_s) if hi_s.strip() else n
                for k in range(lo, hi + 1):
                    if 1 <= k <= n:
                        indices.append(k - 1)
            elif part.isdigit():
                k = int(part)
                if 1 <= k <= n:
                    indices.append(k - 1)
        return indices

    def execute_script(self, script_text: str, args: Optional[List[str]] = None) -> List[str]:
        """Execute a multi-line SH script with support for labels, GOTO, IF, SDEF, and #P1..#Pn parameters."""
        raw_lines = [ln.strip() for ln in script_text.splitlines()]
        # Substitute positional parameters #1, #2, ... in a single pass so #1 never corrupts #10+
        if args:
            def _sub_arg(m: re.Match[str]) -> str:
                idx = int(m.group(1))
                return args[idx - 1] if 1 <= idx <= len(args) else m.group(0)

            raw_lines = [re.sub(r"#([1-9]\d*)(?!\d)", _sub_arg, ln) for ln in raw_lines]

        labels: Dict[str, int] = {}
        for idx, ln in enumerate(raw_lines):
            if ln.endswith(":") and not ln.startswith("!"):
                labels[ln[:-1].strip().upper()] = idx

        pc = 0
        logs: List[str] = []
        max_steps = 5000
        steps = 0
        while pc < len(raw_lines) and not self.quit_requested and steps < max_steps:
            steps += 1
            line = raw_lines[pc]
            pc += 1
            if not line or line.startswith("!") or line.endswith(":"):
                continue

            # Check GOTO / IF ... GOTO
            u = line.upper()
            if u.startswith("GOTO "):
                target = line.split(None, 1)[1].strip().upper().rstrip(":")
                if target in labels:
                    pc = labels[target]
                continue

            res = self.execute_line(line)
            if res:
                logs.append(res)
                if res.startswith("__GOTO__:"):
                    target = res.split(":", 1)[1].strip().upper()
                    if target in labels:
                        pc = labels[target]

        return logs

    @staticmethod
    def _split_statements(s: str) -> List[str]:
        return split_statements(s)

    def _substitute_symbols(self, s: str) -> str:
        """Substitute $VAR (boundary-safe) and legacy unquoted \"VAR symbols."""
        if not self.symbols:
            return s
        out: List[str] = []
        i = 0
        n = len(s)
        sorted_syms = sorted(self.symbols.keys(), key=len, reverse=True)

        def _expand_dollars(text: str) -> str:
            def _rep(m: re.Match[str]) -> str:
                key = m.group(1)
                if key in self.symbols:
                    return self.symbols[key]
                up = key.upper()
                if up in self.symbols:
                    return self.symbols[up]
                return m.group(0)

            return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", _rep, text)

        while i < n:
            ch = s[i]
            if ch == "'":
                j = s.find("'", i + 1)
                if j != -1:
                    out.append(s[i : j + 1])
                    i = j + 1
                    continue
                out.append(ch)
                i += 1
            elif ch == '"':
                j = i + 1
                closed = False
                while j < n:
                    if s[j] == "\\" and j + 1 < n and s[j + 1] == '"':
                        j += 2
                        continue
                    if s[j] == '"':
                        closed = True
                        break
                    j += 1
                if closed:
                    out.append('"' + _expand_dollars(s[i + 1 : j]) + '"')
                    i = j + 1
                    continue
                matched = False
                rem = s[i + 1 :]
                for sym_k in sorted_syms:
                    if rem.startswith(sym_k) and (
                        len(rem) == len(sym_k) or not (rem[len(sym_k)].isalnum() or rem[len(sym_k)] == "_")
                    ):
                        out.append(self.symbols[sym_k])
                        i += 1 + len(sym_k)
                        matched = True
                        break
                if not matched:
                    out.append(ch)
                    i += 1
            else:
                j = i
                while j < n and s[j] not in ("'", '"'):
                    j += 1
                out.append(_expand_dollars(s[i:j]))
                i = j
        return "".join(out)

    def execute_line(self, line: str) -> str:
        """Execute a single SH command line and return any formatted text response."""
        s = line.strip()
        if not s or s.startswith("!"):
            return ""

        # Split multiple semicolon-separated statements outside quotes first so SDEF/CALC
        # take effect for subsequent statements on the same line.
        stmts = self._split_statements(s)
        if len(stmts) > 1:
            outputs = [self.execute_line(part) for part in stmts]
            return "\n".join(o for o in outputs if o)
        if not stmts:
            return ""
        s = stmts[0]

        # Substitute symbol variables $VAR (or legacy unquoted "VAR)
        s = self._substitute_symbols(s)

        tokens, quals = self._tokenize_command(s)
        if not tokens:
            return ""

        verb = tokens[0].upper()
        args = tokens[1:]

        try:
            return self._dispatch(verb, args, quals)
        except Exception as exc:
            msg = f"SH Error in {verb}: {exc}"
            self.emit(msg)
            return msg

    @staticmethod
    @staticmethod
    def _split_quoted_tokens(cmd: str) -> List[tuple[str, bool]]:
        return split_quoted_tokens(cmd)

    @classmethod
    def _is_qualifier_token(cls, p: str, was_quoted: bool, verb: Optional[str]) -> bool:
        return is_qualifier_token(p, was_quoted, verb)

    @classmethod
    def _tokenize_command(cls, cmd: str) -> tuple[List[str], Dict[str, str]]:
        return tokenize_command(cmd)

    def _dispatch(self, verb: str, args: List[str], quals: Dict[str, str]) -> str:
        if verb in ("QUIT", "EXIT", "BYE"):
            self.quit_requested = True
            msg = "Exiting SH."
            self.emit(msg)
            return msg

        if verb in ("HELP", "?"):
            return self._cmd_help(args)

        if verb == "ECHO":
            msg = " ".join(args)
            self.emit(msg)
            return msg

        if verb == "SDEF":
            if len(args) >= 1:
                name = args[0].upper()
                val = " ".join(args[1:]) if len(args) > 1 else ""
                self.symbols[name] = val
                msg = f"{name} = {val}"
                self.emit(msg)
                return msg
            return ""

        if verb == "CALC":
            return self._cmd_calc(args)

        if verb == "IF":
            return self._cmd_if(args)

        # --- File I/O Commands ---
        if verb == "READ":
            return self._cmd_read_qfile(args, quals)
        if verb == "WRITE":
            return self._cmd_write_qfile(args, quals)
        if verb in ("READS", "READM", "READSEED"):
            return self._cmd_reads(args, quals)
        if verb == "WRITESEED":
            return self._cmd_writeseed(args, quals)
        if verb == "READGSE":
            return self._cmd_readgse(args, quals)
        if verb == "WRITEGSE":
            return self._cmd_writegse(args, quals)
        if verb == "READA":
            return self._cmd_reada(args, quals)
        if verb == "WRITEA":
            return self._cmd_writea(args, quals)
        if verb == "READEVT":
            return self._cmd_readevt(args)
        if verb == "WRITEEVT":
            return self._cmd_writeevt(args)
        if verb == "READSXML":
            return self._cmd_readsxml(args, quals)
        if verb == "WRITEQML":
            return self._cmd_writeqml(args, quals)
        if verb == "READQML":
            return self._cmd_readqml(args)
        if verb == "WRITEGEOJSON":
            return self._cmd_writegeojson(args, quals)
        if verb == "REPORT":
            return self._cmd_report(args, quals)
        if verb == "SFD":
            return self._cmd_sfd(args)
        if verb in ("FETCH", "BGRFETCH"):
            return self._cmd_fetch(args, quals)

        # --- Display, Selection, Deletion, Windowing ---
        if verb in ("LIST", "DIR", "TRACES", "SH"):
            return self._cmd_list_traces()
        if verb in ("DEL", "DELETE"):
            return self._cmd_delete(args)
        if verb == "HIDE":
            return self._cmd_hide(args, hide=True)
        if verb in ("DISPLAY", "SHOW", "UNHIDE"):
            return self._cmd_hide(args, hide=False)
        if verb in ("SEL", "SELECT"):
            return self._cmd_select(args)
        if verb == "TRESET":
            return self._cmd_treset()
        if verb == "NORM":
            if args:
                mode = args[0].upper()
                if mode not in ("A", "C", "AW", "SW"):
                    raise ValueError(f"Unknown normalization mode '{mode}'; expected A, C, AW, or SW")
                self.norm_mode = mode
            msg = f"Normalization mode: {self.norm_mode}"
            self.emit(msg)
            return msg
        if verb == "ZOOM":
            return self._cmd_zoom(args, quals)
        if verb == "STW":
            if len(args) >= 2:
                lo, hi = float(args[0]), float(args[1])
                if hi < lo:
                    lo, hi = hi, lo
                self.time_window = (lo, hi)
                self.view_window = (lo, hi)
                self._view_window_version += 1
                msg = f"Time window set to [{self.time_window[0]:.2f}, {self.time_window[1]:.2f}] s"
                self.emit(msg)
                return msg
            self.time_window = None
            self.view_window = None
            self._view_window_version += 1
            msg = "Time window reset to full trace duration"
            self.emit(msg)
            return msg
        if verb == "DTW":
            self.time_window = None
            self.view_window = None
            self._view_window_version += 1
            msg = "Time window reset"
            self.emit(msg)
            return msg
        if verb == "TITLE":
            self.title = " ".join(args)
            msg = f"Title: {self.title}"
            self.emit(msg)
            return msg

        # --- Filtering & Waveform Processing ---
        if verb == "FILI":
            self.loaded_filters = list(args)
            msg = f"Loaded filter(s): {', '.join(self.loaded_filters)}"
            self.emit(msg)
            return msg
        if verb == "FILTER":
            return self._cmd_filter(args, quals)
        if verb == "BUTTER":
            return self._cmd_butter(args, quals)
        if verb in ("ADDFIL", "COLOUR_ADDFIL", "COLOR_ADDFIL"):
            return self._cmd_addfil(args, quals)
        if verb == "AUTOCUT":
            return self._cmd_autocut(args, quals)
        if verb == "DEMEAN":
            return self._cmd_demean(args, quals)
        if verb == "TREND":
            return self._cmd_trend(args)
        if verb == "DESPIKE":
            return self._cmd_despike(args)
        if verb == "RESAMPLE":
            return self._cmd_resample(args)
        if verb == "DECIMATE":
            return self._cmd_decimate(args)
        if verb == "CUT":
            return self._cmd_cut(args, quals)
        if verb == "SHIFT":
            return self._cmd_shift(args)
        if verb in ("INT", "INTEGRATE"):
            return self._cmd_int(args)
        if verb in ("DERIVE", "DIFF"):
            return self._cmd_derive(args, quals)
        if verb == "HILBERT":
            return self._cmd_hilbert(args)
        if verb in ("ENVELOPE", "ENV"):
            return self._cmd_envelope(args)
        if verb == "SPHASE":
            return self._cmd_sphase(args)
        if verb == "FLIP":
            return self._cmd_mult(args, factor_override=-1.0)
        if verb == "MULT":
            return self._cmd_mult(args)
        if verb == "SUM":
            return self._cmd_sum(args, quals)
        if verb == "PWS":
            return self._cmd_pws(args)
        if verb == "MEND":
            return self._cmd_mend(args)
        if verb == "ATTENUATE":
            return self._cmd_attenuate(args)
        if verb == "SPIKING":
            return self._cmd_spiking(args)
        if verb == "DECONV":
            return self._cmd_deconv(args)
        if verb == "CREATE":
            return self._cmd_create(args)

        # --- 3-Component & Polarization ---
        if verb in ("ROT", "ROT2", "ROTAZ"):
            return self._cmd_rot2(args)
        if verb in ("ROT3", "ROTINCI"):
            return self._cmd_rot3(args)
        if verb == "UVW":
            return self._cmd_uvw(args, quals)
        if verb == "MDIR":
            return self._cmd_mdir(args)
        if verb == "POLFIL":
            return self._cmd_polfil(args)

        # --- Array & Location & Picking ---
        if verb == "BEAM":
            return self._cmd_beam(args, quals)
        if verb == "ALIGN":
            return self._cmd_align(args)
        if verb == "VESPA":
            return self._cmd_vespa(args)
        if verb == "FK":
            return self._cmd_fk(args, quals)
        if verb == "CALIB":
            return self._cmd_calib(args, quals)
        if verb == "LOCATE":
            return self._cmd_locate(args, quals)
        if verb == "IDENTIFY":
            return self._cmd_identify(args, quals)
        if verb in ("PICK", "AUTOPICK"):
            return self._cmd_pick(args)
        if verb == "CORRPICK":
            return self._cmd_corrpick(args)
        if verb == "CORR":
            return self._cmd_corr(args)
        if verb in ("AMPL", "MAG", "MAGNITUDE"):
            return self._cmd_mag(args)
        if verb == "SURFMAG":
            return self._cmd_surfmag(args, quals)
        if verb == "THEO":
            return self._cmd_theo(args)

        # --- Spectrum & Diagnostics Fit ---
        if verb == "SPECTRUM":
            return self._cmd_spectrum(args, quals)
        if verb == "FIT":
            return self._cmd_fit(args, quals)

        # Check if verb is a .SHC script in globals or current directory
        for cand in (verb + ".SHC", verb.lower() + ".shc", os.path.join(self.config.globals_dir, verb + ".SHC")):
            if os.path.isfile(cand):
                with open(cand, "r", encoding="latin-1", errors="ignore") as f:
                    logs = self.execute_script(f.read(), args=args)
                return "\n".join(logs)

        raise ValueError(f"Unknown command {verb!r} (type HELP for list of commands)")

    # ------------------------------------------------------------------
    # Individual command implementations
    # ------------------------------------------------------------------

    _HELP_TOPICS: Dict[str, str] = {
        "READ": "READ <qfile> [rec_list]  — Read traces from SH Q-file (.QHD/.QBN). Example: READ grf_01 1-3",
        "WRITE": "WRITE <qfile> [trlist]  — Write traces to SH Q-file (.QHD/.QBN). Example: WRITE out 1,2",
        "READS": "READS <sfd|mseed|dir> [t_start] [dur] [streams]  — Read Mini-SEED file/directory or SFD index. Aliases: READM, READSEED",
        "WRITESEED": "WRITESEED <file> [trlist]  — Export traces to Mini-SEED file.",
        "READGSE": "READGSE <file>  — Read GSE1/GSE2 waveform file.",
        "WRITEGSE": "WRITEGSE <file> [trlist]  — Write traces to GSE2 waveform file.",
        "READA": "READA <file>  — Read ASCII waveform file.",
        "WRITEA": "WRITEA <file> [trlist]  — Write traces to ASCII file.",
        "READEVT": "READEVT <file.evt>  — Read SH event parameter & pick bulletin.",
        "WRITEEVT": "WRITEEVT <file.evt>  — Write SH event parameter & pick bulletin.",
        "READSXML": "READSXML <file.xml>  — Load FDSN StationXML station/channel metadata and responses.",
        "WRITEQML": "WRITEQML <file.qml> [trlist] [/PREFIX=...]  — Export QuakeML 1.2 XML event bulletin.",
        "READQML": "READQML <file.qml>  — Load QuakeML 1.2 event parameters, picks, and focal mechanism.",
        "WRITEGEOJSON": "WRITEGEOJSON <file.geojson> [trlist] [/NOSTATIONS] [/NOELLIPSE]  — Export GeoJSON FeatureCollection.",
        "REPORT": "REPORT <file.pdf|file.svg> [trlist] [/FORMAT=PDF|SVG] [/TITLE=...] [/AGENCY=...]  — Generate vector observatory bulletin.",
        "SFD": "SFD <dir> [outfile]  — Generate SFD index file from Mini-SEED directory.",
        "FETCH": "FETCH /BGR <net> <stations> <channels> <start_time> <duration> [/OUT=...] [/APPEND] [/NOSFD]  — Fetch FDSN/BGR waveforms. Alias: BGRFETCH",
        "LIST": "LIST  — Display workspace trace table. Aliases: DIR, TRACES, SH",
        "DEL": "DEL [trlist]  — Delete traces from workspace (default ALL). Alias: DELETE",
        "HIDE": "HIDE [trlist]  — Hide traces from display.",
        "DISPLAY": "DISPLAY [trlist]  — Show hidden traces. Aliases: SHOW, UNHIDE",
        "SEL": "SEL [trlist|NONE]  — Select traces or clear selection. Alias: SELECT",
        "TRESET": "TRESET  — Reset global/per-trace zoom, offsets, hidden flags, and time window.",
        "NORM": "NORM <C|A|AW|SW>  — Set amplitude normalization mode.",
        "ZOOM": "ZOOM [trlist] <factor> [/REL]  — Set global or per-trace amplitude zoom factor.",
        "STW": "STW [<lo_sec> <hi_sec>]  — Set analysis/view time window (no args resets).",
        "DTW": "DTW  — Reset time window to full trace duration.",
        "TITLE": "TITLE <text>  — Set workspace/bulletin title.",
        "FILI": "FILI <filter...>  — Set default loaded filter name(s).",
        "FILTER": "FILTER [filter_expr] [/LIST=...] [/REC|/FFT] [/LPC] [/AUTOCUT] [/ZEROPHASE]  — Apply named or composite filter.",
        "BUTTER": "BUTTER <LP|HP|BP|BS> <f_lo> [f_hi] [order] [/LIST=...] [/AUTOCUT] [/ZEROPHASE]  — Apply Butterworth filter.",
        "ADDFIL": "ADDFIL <filter> [color] [/LIST=...] [/CLEAR|/OFF] [/REC|/FFT]  — Assign or clear trace overlay filter. Aliases: COLOUR_ADDFIL, COLOR_ADDFIL",
        "AUTOCUT": "AUTOCUT [trlist] [settle_sec] [/LIST=...] [/SETTLE=...] [/ZEROPHASE]  — Trim filter startup transient.",
        "DEMEAN": "DEMEAN [trlist] [<lo_sec> <hi_sec>]  — Subtract trace mean (full trace or window).",
        "TREND": "TREND [trlist]  — Remove least-squares linear trend.",
        "DESPIKE": "DESPIKE [trlist] [critmul]  — Remove single-sample spikes.",
        "RESAMPLE": "RESAMPLE [trlist] <new_dt>  — Resample traces to new sample interval dt (s).",
        "DECIMATE": "DECIMATE [trlist] <factor>  — Integer-factor decimation with anti-alias filtering.",
        "CUT": "CUT [trlist] <lo_sec> <hi_sec>  — Cut traces to relative time window [lo, hi] s.",
        "SHIFT": "SHIFT [trlist] <sec>  — Shift trace display_offset by <sec> seconds.",
        "INT": "INT [trlist]  — Integrate waveforms. Alias: INTEGRATE",
        "DERIVE": "DERIVE [trlist] [mode]  — Differentiate waveforms. Alias: DIFF",
        "HILBERT": "HILBERT [trlist]  — Compute Hilbert transform (90 deg phase shift).",
        "ENVELOPE": "ENVELOPE [trlist]  — Compute analytic envelope. Alias: ENV",
        "SPHASE": "SPHASE [trlist]  — Compute instantaneous phase (-180..180 deg).",
        "FLIP": "FLIP [trlist]  — Invert waveform polarity (* -1).",
        "MULT": "MULT [trlist] <factor>  — Multiply waveform samples by scalar factor.",
        "SUM": "SUM [trlist] [/NONORM]  — Stack traces into a SUM trace.",
        "PWS": "PWS [trlist] [power]  — Phase-Weighted Stack.",
        "MEND": "MEND <idx_lo> <idx_hi> [trlist]  — Linearly interpolate damaged sample range.",
        "ATTENUATE": "ATTENUATE <t_star> [trlist]  — Apply Futterman causal Q attenuation.",
        "SPIKING": "SPIKING [filter_len] [prewhitening]  — Levinson spiking deconvolution on trace 1.",
        "DECONV": "DECONV [water_level]  — Water-level spectral division deconvolution (trace 1 / trace 2).",
        "CREATE": "CREATE <KUEPPER|SPIKE|SINE|...> [dur] [dt] [pos] [ampl] [width]  — Create synthetic trace.",
        "ROT2": "ROT2 [azimuth_deg]  — Rotate N, E traces to R, T. Aliases: ROT, ROTAZ",
        "ROT3": "ROT3 [azimuth_deg] [inci_deg]  — 3D rotation Z, N, E -> L, Q, T. Alias: ROTINCI",
        "UVW": "UVW [/INV]  — Transform STS-2 ZNE <-> UVW.",
        "MDIR": "MDIR [<lo_sec> <hi_sec>]  — 3C covariance polarization analysis (azimuth, incidence, rectilinearity).",
        "POLFIL": "POLFIL [win_sec] [power] [mode]  — Sliding-window 3C polarization filter.",
        "BEAM": "BEAM [azimuth] [slowness] [ref_station]  — Form plane-wave array beam.",
        "ALIGN": "ALIGN [phase]  — Align traces on phase pick and append ALIGN stack.",
        "VESPA": "VESPA [azimuth] [s_lo] [s_hi] [s_step] [power]  — Compute slowness vespagram.",
        "FK": "FK [f_lo] [f_hi] [max_slo] [/NOCALIB]  — Frequency-wavenumber (FK) slowness/azimuth analysis.",
        "CALIB": "CALIB [/ARRAY[=name]|<file>|/OFF] [/NOAPPLY]  — Load/apply/clear empirical slowness calibration.",
        "LOCATE": "LOCATE [phase] [/GEIGER|/NET|/SLOWNESS|/FREE] [/NOCALIB]  — Locate event (plane-wave, slowness, or Geiger).",
        "IDENTIFY": "IDENTIFY [trlist] [/TIME=... /SLO=... /DIST=... /DEPTH=...] [/NOUPDATE]  — Probabilistic phase identification.",
        "PICK": "PICK [trlist]  — Automatic Baer-Kradolfer P-onset picker. Alias: AUTOPICK",
        "CORRPICK": "CORRPICK [phase]  — Cross-correlation array picker.",
        "CORR": "CORR [<t1> <t2> [<lo_sec> <hi_sec>]]  — Normalized cross-correlation between two traces.",
        "MAG": "MAG [phase]  — Compute ML, mb, MS, mB magnitudes. Aliases: AMPL, MAGNITUDE",
        "SURFMAG": "SURFMAG [trlist] [/ALLCOMP] [/VELOCITY] [/PERIOD_LO=...] [/PERIOD_HI=...] [/DIST=...]  — Surface-wave Ms measurement.",
        "THEO": "THEO [dist_deg] [depth_km]  — Compute theoretical phase arrival markers.",
        "SPECTRUM": "SPECTRUM [/FFT|/PSD|/MEM] [trlist]  — Compute amplitude/power/MEM spectrum.",
        "FIT": "FIT </BRUNE|/BOATWRIGHT|/POL|/FOCAL|/LOCATE> [args...]  — Run inversion with diagnostic payload.",
        "SDEF": "SDEF <var> <val>  — Define symbol variable ($VAR).",
        "CALC": (
            "CALC <var> = <expr>  — Evaluate math expression and store in symbol $VAR.\n"
            "  Functions: sin(x), cos(x), tan(x), sqrt(x), log(x) [base-10], abs(x)\n"
            "  Constants: pi, e   Operators: +, -, *, /, //, %, **"
        ),
        "IF": (
            "IF <lhs> <op> <rhs> <cmd>  — Conditional execution (or IF ... GOTO <label>).\n"
            "  Numeric ops: EQI, EQ, ==, =, NEI, NE, !=, GTI, GT, >, GEI, GE, >=, LTI, LT, <, LEI, LE, <=\n"
            "  String ops : EQS, EQ, ==, =, NES, NE, !="
        ),
        "GOTO": "GOTO <label>  — Jump to '<label>:' in .SHC script.",
        "ECHO": "ECHO <text...>  — Print message to console/stdout.",
        "QUIT": "QUIT  — Exit SHC interpreter. Aliases: EXIT, BYE",
        "HELP": "HELP [command]  — Show command summary or detailed help for <command>. Alias: ?",
        "ALIASES": (
            "Command Aliases:\n"
            "  QUIT: EXIT, BYE | HELP: ? | LIST: DIR, TRACES, SH | DEL: DELETE | DISPLAY: SHOW, UNHIDE\n"
            "  SEL: SELECT | READS: READM, READSEED | FETCH: BGRFETCH | ADDFIL: COLOUR_ADDFIL, COLOR_ADDFIL\n"
            "  INT: INTEGRATE | DERIVE: DIFF | ENVELOPE: ENV | ROT2: ROT, ROTAZ | ROT3: ROTINCI\n"
            "  PICK: AUTOPICK | MAG: AMPL, MAGNITUDE"
        ),
    }

    _HELP_ALIASES: Dict[str, str] = {
        "?": "HELP",
        "EXIT": "QUIT",
        "BYE": "QUIT",
        "READM": "READS",
        "READSEED": "READS",
        "BGRFETCH": "FETCH",
        "DIR": "LIST",
        "TRACES": "LIST",
        "SH": "LIST",
        "DELETE": "DEL",
        "SHOW": "DISPLAY",
        "UNHIDE": "DISPLAY",
        "SELECT": "SEL",
        "COLOUR_ADDFIL": "ADDFIL",
        "COLOR_ADDFIL": "ADDFIL",
        "INTEGRATE": "INT",
        "DIFF": "DERIVE",
        "ENV": "ENVELOPE",
        "ROT": "ROT2",
        "ROTAZ": "ROT2",
        "ROTINCI": "ROT3",
        "AUTOPICK": "PICK",
        "AMPL": "MAG",
        "MAGNITUDE": "MAG",
    }

    def _cmd_help(self, args: List[str]) -> str:
        if args:
            raw_topic = args[0].strip().upper().lstrip("/")
            topic = self._HELP_ALIASES.get(raw_topic, raw_topic)
            if topic in self._HELP_TOPICS:
                msg = self._HELP_TOPICS[topic]
                if raw_topic != topic:
                    msg = f"[{raw_topic} -> {topic}] {msg}"
            else:
                msg = f"No help topic found for '{args[0]}'. Type HELP for list of commands."
            self.emit(msg)
            return msg

        msg = (
            "SH Command Interpreter (Modernized SHC / SHM):\n"
            "  I/O        : READ <qfile> [list], WRITE <qfile> [list], READS|READM|READSEED <sfd/mseed> [t_start] [dur] [streams],\n"
            "               WRITESEED <file> [list], READGSE <file>, WRITEGSE <file>, READA <file>, WRITEA <file>,\n"
            "               READEVT <file>, WRITEEVT <file>, READSXML <xml>, WRITEQML <qml>, READQML <qml>,\n"
            "               WRITEGEOJSON <geojson>, REPORT <file.pdf|file.svg>, SFD <dir> [outfile],\n"
            "               FETCH|BGRFETCH /BGR <net> <stations> <channels> <start_time> <duration>\n"
            "  Traces     : LIST|DIR|TRACES|SH, DEL|DELETE <list>, HIDE <list>, DISPLAY|SHOW|UNHIDE <list>,\n"
            "               SEL|SELECT <list>, TRESET, NORM <C|A|AW|SW>, ZOOM [list] <factor>, STW <lo> <hi>, DTW, TITLE <text>\n"
            "  Filtering  : FILI <filter>, FILTER <filter> [/REC|/FFT] [/AUTOCUT], BUTTER <LP|HP|BP|BS> <f1> [f2] [order] [/AUTOCUT],\n"
            "               ADDFIL|COLOUR_ADDFIL <filter> [color] [/LIST=...], AUTOCUT [list] [settle_sec]\n"
            "  Processing : DEMEAN [list [lo hi]], TREND [list], DESPIKE [list] [crit], RESAMPLE <dt>, DECIMATE <factor>,\n"
            "               CUT [list] <lo> <hi>, SHIFT [list] <sec>, INT|INTEGRATE [list], DERIVE|DIFF [list], HILBERT [list],\n"
            "               ENVELOPE|ENV [list], SPHASE [list], FLIP [list], MULT [list] <fac>, SUM [list], PWS [list],\n"
            "               MEND <idx_lo> <idx_hi>, ATTENUATE <t_star>, SPIKING <len> <prew>, DECONV, CREATE <kind>\n"
            "  3-Comp     : ROT|ROT2|ROTAZ <az>, ROT3|ROTINCI <az> <inci>, UVW [/INV], MDIR [lo hi], POLFIL [win] [pow] [mode]\n"
            "  Array/Loc  : BEAM <az> <slo> [ref], ALIGN [phase], VESPA <az> <s_lo> <s_hi> <s_step> [pow],\n"
            "               FK [f_lo] [f_hi] [max_slo], CALIB [/ARRAY|<file>], LOCATE [/PLANE|/SLOWNESS|/GEIGER],\n"
            "               IDENTIFY [list], PICK|AUTOPICK [list], CORRPICK [phase], CORR [t1 t2 [lo hi]],\n"
            "               AMPL|MAG|MAGNITUDE [phase], SURFMAG [list], THEO [dist] [depth]\n"
            "  Spectrum   : SPECTRUM [/FFT|/PSD|/MEM] [list], FIT </BRUNE|/BOATWRIGHT|/POL|/FOCAL|/LOCATE>\n"
            "  Scripting  : SDEF <var> <val>, CALC <var> = <expr> (sin,cos,tan,sqrt,log,abs,pi,e),\n"
            "               IF <lhs> <op: EQ|EQI|EQS|NE|NEI|NES|GT|GTI|GE|GEI|LT|LTI|LE|LEI|==|!=|>|>=|<|<=> <rhs> <cmd>,\n"
            "               GOTO <label>, ECHO, HELP|? [cmd|ALIASES], QUIT|EXIT|BYE"
        )
        self.emit(msg)
        return msg

    def _cmd_calc(self, args: List[str]) -> str:
        # Format: CALC VAR = EXPR or CALC VAR = <val1> <op> <val2>
        joined = " ".join(args)
        if "=" in joined:
            lhs, rhs = joined.split("=", 1)
            var = lhs.strip().upper().lstrip("$")
            expr = rhs.strip()
        elif len(args) >= 2:
            var = args[0].upper().lstrip("$")
            expr = " ".join(args[1:])
        else:
            return ""

        val = _safe_eval_math_expr(expr)
        self.symbols[var] = f"{val:.6g}" if isinstance(val, float) else str(val)
        msg = f"{var} = {self.symbols[var]}"
        self.emit(msg)
        return msg

    def _cmd_if(self, args: List[str]) -> str:
        if len(args) < 4:
            return ""
        lhs, op, rhs = args[0], args[1].upper(), args[2]
        rest_cmd = " ".join(args[3:])
        try:
            lv, rv = float(lhs), float(rhs)
            cond = (
                (op in ("EQI", "EQ", "==", "=") and lv == rv)
                or (op in ("NEI", "NE", "!=") and lv != rv)
                or (op in ("GTI", "GT", ">") and lv > rv)
                or (op in ("GEI", "GE", ">=") and lv >= rv)
                or (op in ("LTI", "LT", "<") and lv < rv)
                or (op in ("LEI", "LE", "<=") and lv <= rv)
            )
        except ValueError:
            cond = (op in ("EQS", "EQ", "==", "=") and lhs == rhs) or (op in ("NES", "NE", "!=") and lhs != rhs)

        if cond:
            if rest_cmd.upper().startswith("GOTO "):
                return f"__GOTO__:{rest_cmd.split(None, 1)[1].strip()}"
            return self.execute_line(rest_cmd)
        return ""

    def _cmd_list_traces(self) -> str:
        if not self.traces:
            msg = "Workspace is empty (0 traces)."
            self.emit(msg)
            return msg
        lines = ["Idx  Stat   Chan  Samples   dt(s)   Start Time                Filter        Picks"]
        lines.append("-" * 82)
        for i, tr in enumerate(self.traces, start=1):
            pk_str = ",".join(p.name for p in tr.picks) if tr.picks else "-"
            flag = "H" if tr.hidden else ("*" if tr.selected else " ")
            lines.append(
                f"{i:2d}{flag}  {tr.station:<5s}  {tr.channel3:<4s}  {tr.length:7d}  {tr.delta:6.3f}   "
                f"{tr.start_time_str:<24s}  {tr.filter_name:<12s}  {pk_str}"
            )
        out = "\n".join(lines)
        self.emit(out)
        return out

    def _cmd_delete(self, args: List[str]) -> str:
        spec = args[0] if args else "ALL"
        idxs = set(self.parse_trace_list(spec))
        before = len(self.traces)
        self.traces = [t for i, t in enumerate(self.traces) if i not in idxs]
        msg = f"Deleted {before - len(self.traces)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_hide(self, args: List[str], hide: bool = True) -> str:
        spec = args[0] if args else "ALL"
        idxs = self.parse_trace_list(spec if hide else "ALL_INC_HIDDEN")
        if not hide and spec.upper() not in ("ALL", "*"):
            idxs = [int(x) - 1 for x in spec.split(",") if x.strip().isdigit()]
        for i in idxs:
            if 0 <= i < len(self.traces):
                self.traces[i].hidden = hide
        msg = f"{'Hidden' if hide else 'Displayed'} {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_select(self, args: List[str]) -> str:
        if not args or args[0].upper() == "NONE":
            for t in self.traces:
                t.selected = False
            msg = "Cleared trace selection."
            self.emit(msg)
            return msg
        idxs = set(self.parse_trace_list(args[0]))
        for i, t in enumerate(self.traces):
            t.selected = i in idxs
        msg = f"Selected {len(idxs)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_treset(self) -> str:
        self.global_zoom = 1.0
        self.norm_mode = "C"
        self.time_window = None
        self.view_window = None
        self._view_window_version += 1
        for tr in self.traces:
            tr.display_offset = 0.0
            tr.norm = 1.0
            tr.hidden = False
        msg = "Reset trace zooms, shifts, and time windows."
        self.emit(msg)
        return msg

    def _cmd_zoom(self, args: List[str], quals: Dict[str, str]) -> str:
        if len(args) == 1:
            fac = float(args[0])
            self.global_zoom = fac if "REL" not in quals else self.global_zoom * fac
            msg = f"Global zoom = {self.global_zoom:.3g}"
            self.emit(msg)
            return msg
        if len(args) >= 2:
            idxs = self.parse_trace_list(args[0])
            fac = float(args[1])
            for i in idxs:
                self.traces[i].norm = fac if "REL" not in quals else self.traces[i].norm * fac
            msg = f"Updated zoom on {len(idxs)} trace(s) to {fac:.3g}"
            self.emit(msg)
            return msg
        return ""

