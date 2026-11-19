"""Interactive and scriptable SH Command Interpreter (shc) supporting all SH waveform,
filtering, array, location, polarization, spectral, I/O, and macro commands.
"""

from __future__ import annotations

import ast
import math
import operator
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import (
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time
from shm_app.io.evt import format_ev2view_summary, read_evt_file, write_evt_file
from shm_app.io.geojson import write_geojson
from shm_app.io.gse import read_gse_file, write_gse2_file
from shm_app.io.mseed import generate_sfd_file, read_from_sfd, read_mseed_file, write_mseed_file
from shm_app.io.qfile import read_ascii_file, read_qfile, write_ascii_file, write_qfile
from shm_app.io.quakeml import parse_quakeml, write_quakeml
from shm_app.processing.filters import autocut_trace, filter_trace
from shm_app.processing.polarization import (
    analyze_polarization_diagnostic,
    apply_polarization_filter,
    compute_mdir,
    rotate_2d,
    rotate_3d_lqt,
    rotate_STS2_zne_to_uvw,
)
from shm_app.processing.spectrum import (
    build_brune_fit_diagnostic,
    compute_fft_spectrum,
    compute_mem_spectrum,
    compute_welch_power_spectrum,
    find_surface_wave_amplitude,
    invert_brune_spectrum,
    scan_surface_waves,
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
from shm_app.seismology.array_loc import (
    SlownessCalibrationTable,
    align_traces_on_phase,
    compute_beam,
    compute_fk_analysis,
    compute_trace_window_magnitudes,
    compute_vespagram,
    correlate_traces,
    corrpick_array,
    fit_depth_phase,
    identify_phase,
    identify_trace_picks,
    load_calibration_for_array,
    locate_by_slowness_azimuth,
    locate_by_sp_diff,
    locate_network_geiger,
    pick_onset_baer_kradolfer,
    plane_wave_locate,
)
from shm_app.seismology.focal_mechanism import (
    extract_polarities_from_traces,
    solve_focal_mechanism,
)
from shm_app.seismology.traveltime import get_ttt_manager, sphere_diff


_SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

_SAFE_MATH_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log10,
    "abs": abs,
}


def _eval_math_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_math_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_CONSTANTS:
        return _SAFE_CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
        return _SAFE_UNARY_OPS[type(node.op)](_eval_math_ast(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
        return _SAFE_BIN_OPS[type(node.op)](_eval_math_ast(node.left), _eval_math_ast(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_MATH_FUNCS and not node.keywords:
        args = [_eval_math_ast(a) for a in node.args]
        return _SAFE_MATH_FUNCS[node.func.id](*args)
    raise ValueError(f"Unsupported expression node in CALC: {ast.dump(node)}")


def _safe_eval_math_expr(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _eval_math_ast(tree)


class SHCommandEngine:
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
        # Substitute positional parameters #1, #2, ...
        if args:
            for idx, val in enumerate(args, start=1):
                raw_lines = [ln.replace(f"#{idx}", val) for ln in raw_lines]

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

    def execute_line(self, line: str) -> str:
        """Execute a single SH command line and return any formatted text response."""
        s = line.strip()
        if not s or s.startswith("!"):
            return ""

        # Substitute symbol variables $VAR or "VAR
        for sym_k, sym_v in self.symbols.items():
            s = s.replace(f"${sym_k}", sym_v).replace(f'"{sym_k}', sym_v)

        # Support multiple semicolon-separated statements on one line
        if ";" in s:
            outputs = [self.execute_line(part) for part in s.split(";") if part.strip()]
            return "\n".join(o for o in outputs if o)

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
    def _tokenize_command(cmd: str) -> Tuple[List[str], Dict[str, str]]:
        """Split SH command line into positional arguments and /QUALIFIER[=VALUE] dictionary."""
        raw_parts = cmd.split()
        tokens: List[str] = []
        quals: Dict[str, str] = {}

        for p in raw_parts:
            # A token starting with '/' is a qualifier unless it looks like an absolute file path
            if p.startswith("/") and not os.path.exists(p) and ("/" not in p[1:] or "=" in p):
                q_body = p[1:]
                if "=" in q_body:
                    k, v = q_body.split("=", 1)
                    quals[k.strip().upper()] = v.strip()
                else:
                    for sub_q in q_body.split("/"):
                        if sub_q:
                            quals[sub_q.strip().upper()] = "TRUE"
            else:
                tokens.append(p)
        return tokens, quals

    def _dispatch(self, verb: str, args: List[str], quals: Dict[str, str]) -> str:
        if verb in ("QUIT", "EXIT", "BYE"):
            self.quit_requested = True
            return "Exiting SH."

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
                return f"{name} = {val}"
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
            return f"Normalization mode: {self.norm_mode}"
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
                return f"Time window set to [{self.time_window[0]:.2f}, {self.time_window[1]:.2f}] s"
            self.time_window = None
            self.view_window = None
            self._view_window_version += 1
            return "Time window reset to full trace duration"
        if verb == "DTW":
            self.time_window = None
            self.view_window = None
            self._view_window_version += 1
            return "Time window reset"
        if verb == "TITLE":
            self.title = " ".join(args)
            return f"Title: {self.title}"

        # --- Filtering & Waveform Processing ---
        if verb == "FILI":
            self.loaded_filters = list(args)
            return f"Loaded filter(s): {', '.join(self.loaded_filters)}"
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

    def _cmd_help(self, args: List[str]) -> str:
        msg = (
            "SH Command Interpreter (Modernized SHC / SHM):\n"
            "  I/O        : READ <qfile> [list], WRITE <qfile> [list], READS <sfd/mseed> [t_start] [dur] [streams],\n"
            "               WRITESEED <file> [list], READGSE <file>, WRITEGSE <file>, READA <file>, WRITEA <file>,\n"
            "               READEVT <file>, WRITEEVT <file>, READSXML <xml>, WRITEQML <qml>, READQML <qml>,\n"
            "               WRITEGEOJSON <geojson>, REPORT <file.pdf|file.svg>, SFD <dir> [outfile],\n"
            "               FETCH /BGR <net> <stations> <channels> <start_time> <duration>\n"
            "  Traces     : LIST, DEL <list>, HIDE <list>, DISPLAY <list>, SEL <list>, TRESET, NORM <C|A|AW|SW>,\n"
            "               ZOOM [list] <factor>, STW <lo> <hi>, DTW, TITLE <text>\n"
            "  Filtering  : FILI <filter>, FILTER <filter> [/AUTOCUT], BUTTER <LP|HP|BP|BS> <f1> [f2] [order] [/AUTOCUT],\n"
            "               ADDFIL / COLOUR_ADDFIL <filter> [color] [/LIST=...], AUTOCUT [list] [settle_sec]\n"
            "  Processing : DEMEAN [list], TREND [list], DESPIKE [list] [crit], RESAMPLE <dt>, DECIMATE <factor>,\n"
            "               CUT [list] <lo> <hi>, SHIFT [list] <sec>, INT [list], DERIVE [list], HILBERT [list],\n"
            "               ENVELOPE [list], SPHASE [list], FLIP [list], MULT [list] <fac>, SUM [list], PWS [list],\n"
            "               MEND <idx_lo> <idx_hi>, ATTENUATE <t_star>, SPIKING <len> <prew>, DECONV, CREATE <kind>\n"
            "  3-Comp     : ROT2 <az>, ROT3 <az> <inci>, UVW [/INV], MDIR [lo] [hi], POLFIL [win] [pow] [mode]\n"
            "  Array/Loc  : BEAM <az> <slo> [ref], ALIGN [phase], VESPA <az> <s_lo> <s_hi> <s_step> [pow],\n"
            "               FK [f_lo] [f_hi] [max_slo], CALIB [/ARRAY|<file>], LOCATE [/PLANE|/SLOWNESS|/GEIGER],\n"
            "               IDENTIFY [list], PICK [list], CORRPICK [phase], CORR <t1> <t2> [lo] [hi],\n"
            "               MAG [phase], SURFMAG [list], THEO [dist] [depth]\n"
            "  Spectrum   : SPECTRUM [/FFT|/PSD|/MEM] [list], FIT </BRUNE|/BOATWRIGHT|/POL|/FOCAL|/LOCATE>\n"
            "  Scripting  : SDEF <var> <val>, CALC <var> = <expr>, IF <cond> <cmd>, GOTO <label>, ECHO, QUIT"
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
        return f"{var} = {self.symbols[var]}"

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

    def _cmd_read_qfile(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READ requires a Q-file path")
        fpath = args[0]
        rec_list: Optional[List[int]] = None
        if len(args) > 1 and args[1].upper() != "ALL":
            rec_list = []
            for part in args[1].split(","):
                if "-" in part:
                    a, b = part.split("-", 1)
                    rec_list.extend(range(int(a), int(b) + 1))
                elif part.isdigit():
                    rec_list.append(int(part))
        trs = read_qfile(fpath, rec_indices=rec_list, config=self.config)
        self.traces.extend(trs)
        msg = f"Read {len(trs)} trace(s) from Q-file {fpath}"
        self.emit(msg)
        return msg

    def _cmd_write_qfile(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITE requires a Q-file path")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        qhd, _qbn = write_qfile(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to {qhd}"
        self.emit(msg)
        return msg

    def _cmd_reads(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READS requires an SFD or Mini-SEED file/directory path")
        path = args[0]
        if path.upper() == "INQUIRE":
            path = os.path.join(self.config.data_examples_dir, "mseed")

        t_start: Optional[float] = None
        dur: Optional[float] = None
        streams: Optional[List[str]] = None

        if len(args) >= 2:
            try:
                t_start = parse_sh_time(args[1])
            except ValueError:
                streams = [s.strip() for s in args[1].split(",")]
        if len(args) >= 3:
            try:
                dur = float(args[2])
            except ValueError:
                streams = [s.strip() for s in args[2].split(",")]
        if len(args) >= 4:
            streams = [s.strip() for s in args[3].split(",")]
            if len(args) >= 5:
                comp_arg = args[4].strip().lower()
                chan_arg = (args[5].strip().lower() if len(args) >= 6 else "bh")
                streams = [f"{st.lower()}-{chan_arg}-{c}" if "-" not in st else st for st in streams for c in comp_arg]

        if os.path.isdir(path):
            sfd_cand = os.path.join(path, "sfdlist.sfd")
            if os.path.isfile(sfd_cand):
                trs = read_from_sfd(sfd_cand, streams=streams, t_start=t_start, duration=dur, config=self.config)
            else:
                trs = []
                for fn in sorted(os.listdir(path)):
                    fp = os.path.join(path, fn)
                    if os.path.isfile(fp) and not fn.endswith(".sfd"):
                        trs.extend(read_mseed_file(fp, t_start=t_start, duration=dur, config=self.config))
        elif path.lower().endswith(".sfd"):
            trs = read_from_sfd(path, streams=streams, t_start=t_start, duration=dur, config=self.config)
        else:
            trs = read_mseed_file(path, t_start=t_start, duration=dur, config=self.config)

        self.traces.extend(trs)
        msg = f"Read {len(trs)} Mini-SEED trace(s) from {path}"
        self.emit(msg)
        return msg

    def _cmd_writeseed(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITESEED requires an output filepath")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        write_mseed_file(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to Mini-SEED {fpath}"
        self.emit(msg)
        return msg

    def _cmd_readgse(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READGSE requires a GSE file path")
        fpath = args[0]
        trs = read_gse_file(fpath, config=self.config)
        self.traces.extend(trs)
        msg = f"Read {len(trs)} GSE trace(s) from {fpath}"
        self.emit(msg)
        return msg

    def _cmd_writegse(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEGSE requires a GSE output path")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        write_gse2_file(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to GSE2 file {fpath}"
        self.emit(msg)
        return msg

    def _cmd_reada(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READA requires an ASCII file path")
        fpath = args[0]
        trs = read_ascii_file(fpath, config=self.config)
        self.traces.extend(trs)
        msg = f"Read {len(trs)} ASCII trace(s) from {fpath}"
        self.emit(msg)
        return msg

    def _cmd_writea(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEA requires an ASCII output path")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        write_ascii_file(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to ASCII file {fpath}"
        self.emit(msg)
        return msg

    def _cmd_readevt(self, args: List[str]) -> str:
        if not args:
            raise ValueError("READEVT requires an .evt file path")
        param, picks = read_evt_file(args[0])
        self.param_sets[self.active_set_index] = param
        # Attach picks to matching traces
        for p in picks:
            for tr in self.traces:
                if tr.station.upper() == p.station.upper() and (not p.comp or tr.comp.upper() == p.comp.upper()):
                    tr.add_or_replace_pick(p)
                    break
        summary = format_ev2view_summary(param, picks)
        self.emit(summary)
        return summary

    def _cmd_writeevt(self, args: List[str]) -> str:
        if not args:
            raise ValueError("WRITEEVT requires an .evt output path")
        all_picks: List[PhasePick] = []
        for tr in self.traces:
            all_picks.extend(tr.picks)
        write_evt_file(args[0], self.current_params, all_picks)
        msg = f"Wrote event parameters ({len(all_picks)} picks) to {args[0]}"
        self.emit(msg)
        return msg

    def _cmd_readsxml(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READSXML requires a StationXML file path")
        fpath = args[0]
        inv = self.config.load_stationxml(fpath)
        self.last_stationxml_inventory = inv
        n_sta = len(inv.stations)
        n_cha = sum(len(s.channels) for s in inv.stations)
        # Optionally enrich existing workspace traces that lack coordinates
        for tr in self.traces:
            st = self.config.get_station(tr.station)
            if st is not None and (st.lat != 0.0 or st.lon != 0.0) and tr.lat == 0.0 and tr.lon == 0.0:
                tr.lat = st.lat
                tr.lon = st.lon
        msg = f"Loaded StationXML from {fpath}: {n_sta} station(s), {n_cha} channel(s)."
        self.emit(msg)
        return msg

    def _cmd_writeqml(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEQML requires an output filepath")
        fpath = args[0]
        list_spec = args[1] if len(args) > 1 else quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        trs = [self.traces[i] for i in idxs]
        prefix = quals.get("PREFIX", "smi:shm-modern")
        write_quakeml(
            fpath,
            trs,
            self.current_params,
            public_id_prefix=prefix,
            focal_mechanism=self.last_focal_mechanism,
        )
        msg = f"Wrote QuakeML 1.2 ({len(trs)} trace(s)) to {fpath}"
        self.emit(msg)
        return msg

    def _cmd_readqml(self, args: List[str]) -> str:
        if not args:
            raise ValueError("READQML requires a QuakeML file path")
        fpath = args[0]
        qml = parse_quakeml(fpath)
        param = qml["params"]
        picks = qml.get("phase_picks", [])
        fms = qml.get("focal_mechanisms", [])
        if fms:
            fm0 = fms[0]
            np1 = fm0.get("nodalPlane1")
            if isinstance(np1, dict):
                fm_dict: Dict[str, Any] = {
                    "strike": float(np1.get("strike", 0.0)),
                    "dip": float(np1.get("dip", 0.0)),
                    "rake": float(np1.get("rake", 0.0)),
                }
                np2 = fm0.get("nodalPlane2")
                if isinstance(np2, dict):
                    fm_dict["strike2"] = float(np2.get("strike", 0.0))
                    fm_dict["dip2"] = float(np2.get("dip", 0.0))
                    fm_dict["rake2"] = float(np2.get("rake", 0.0))
                self.last_focal_mechanism = fm_dict
                setattr(param, "focal_mechanism", fm_dict)

        self.param_sets[self.active_set_index] = param
        for p in picks:
            for tr in self.traces:
                if tr.station.upper() == p.station.upper() and (not p.comp or tr.comp.upper() == p.comp.upper()):
                    tr.add_or_replace_pick(p)
                    break
            else:
                for tr in self.traces:
                    if tr.station.upper() == p.station.upper():
                        tr.add_or_replace_pick(p)
                        break

        msg = (
            f"Read QuakeML {qml.get('schema_version', '1.2')} from {fpath}: "
            f"Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, "
            f"{len(picks)} pick(s)."
        )
        self.emit(msg)
        return msg

    def _cmd_writegeojson(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEGEOJSON requires an output filepath")
        fpath = args[0]
        list_spec = args[1] if len(args) > 1 else quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        trs = [self.traces[i] for i in idxs]
        write_geojson(
            fpath,
            trs,
            self.current_params,
            include_stations=("NOSTATIONS" not in quals),
            include_ellipse=("NOELLIPSE" not in quals),
            focal_mechanism=self.last_focal_mechanism,
        )
        msg = f"Wrote GeoJSON ({len(trs)} trace(s)) to {fpath}"
        self.emit(msg)
        return msg

    def _cmd_report(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("REPORT requires an output filepath (.pdf or .svg)")
        fpath = args[0]
        list_spec = args[1] if len(args) > 1 else quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        trs = [self.traces[i] for i in idxs]
        fmt = quals.get("FORMAT", quals.get("FMT"))
        if fmt is None:
            if "SVG" in quals:
                fmt = "svg"
            elif "PDF" in quals:
                fmt = "pdf"
        title = quals.get("TITLE", self.title)
        agency = quals.get("AGENCY", "SHM-Modern Observatory")

        from shm_app.gui.reporting import generate_observatory_bulletin

        out_path = generate_observatory_bulletin(
            output_path=fpath,
            traces=trs,
            params=self.current_params,
            fit_result=self.last_fit_diagnostic,
            focal_mech=self.last_focal_mechanism,
            title=title,
            format=fmt,
            agency=agency,
            config=self.config,
        )
        msg = f"Generated observatory bulletin report: {out_path}"
        self.emit(msg)
        return msg

    def _cmd_sfd(self, args: List[str]) -> str:
        dpath = args[0] if args else "."
        out_sfd = args[1] if len(args) > 1 else os.path.join(dpath, "sfdlist.sfd")
        recs = generate_sfd_file(dpath, out_sfd, config=self.config)
        msg = f"Generated {len(recs)} SFD index entries in {out_sfd}"
        self.emit(msg)
        return msg

    def _cmd_fetch(self, args: List[str], quals: Dict[str, str]) -> str:
        from shm_app.plugins.bgr_fetch import (
            BGR_FDSN_DATASELECT_URL,
            DEFAULT_DOWNLOAD_DIR,
            expand_channels,
            fetch_bgr_waveforms,
            parse_station_list,
        )

        if len(args) >= 5:
            network = args[0]
            stations = parse_station_list(args[1])
            channels = expand_channels(args[2])
            start_str = args[3]
            dur_s = float(args[4])
        elif len(args) == 4:
            network = quals.get("NET", quals.get("NETWORK", "GR"))
            stations = parse_station_list(args[0])
            channels = expand_channels(args[1])
            start_str = args[2]
            dur_s = float(args[3])
        elif len(args) == 3:
            network = quals.get("NET", quals.get("NETWORK", "GR"))
            stations = parse_station_list(args[0])
            channels = expand_channels(quals.get("CHA", quals.get("CHAN", "BH")))
            start_str = args[1]
            dur_s = float(args[2])
        else:
            raise ValueError(
                "FETCH usage: FETCH /BGR <network> <stations> <channels> <start_time> <duration>"
            )

        out_dir = quals.get("OUT", DEFAULT_DOWNLOAD_DIR)
        loc = quals.get("LOC", quals.get("LOCATION", "*"))
        base_url = quals.get("URL", BGR_FDSN_DATASELECT_URL)
        append = "APPEND" in quals
        build_sfd = "NOSFD" not in quals

        summary = fetch_bgr_waveforms(
            stations=stations,
            start_time=start_str,
            duration=dur_s,
            output_dir=out_dir,
            network=network,
            channels=channels,
            location=loc,
            base_url=base_url,
            build_sfd=build_sfd,
            config=self.config,
        )
        if append:
            self.traces.extend(summary.traces)
        elif summary.traces:
            self.traces = list(summary.traces)
        ok_cnt = sum(1 for r in summary.results if r.status == "ok")
        msg = (
            f"Fetched {len(summary.traces)} Mini-SEED trace(s) from {ok_cnt}/{len(summary.results)} "
            f"BGR station(s) into {summary.output_dir}"
        )
        self.emit(msg)
        return msg

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
        return f"{'Hidden' if hide else 'Displayed'} {len(idxs)} trace(s)."

    def _cmd_select(self, args: List[str]) -> str:
        if not args or args[0].upper() == "NONE":
            for t in self.traces:
                t.selected = False
            return "Cleared trace selection."
        idxs = set(self.parse_trace_list(args[0]))
        for i, t in enumerate(self.traces):
            t.selected = i in idxs
        return f"Selected {len(idxs)} trace(s)."

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
        return "Reset trace zooms, shifts, and time windows."

    def _cmd_zoom(self, args: List[str], quals: Dict[str, str]) -> str:
        if len(args) == 1:
            fac = float(args[0])
            self.global_zoom = fac if "REL" not in quals else self.global_zoom * fac
            return f"Global zoom = {self.global_zoom:.3g}"
        if len(args) >= 2:
            idxs = self.parse_trace_list(args[0])
            fac = float(args[1])
            for i in idxs:
                self.traces[i].norm = fac if "REL" not in quals else self.traces[i].norm * fac
            return f"Updated zoom on {len(idxs)} trace(s) to {fac:.3g}"
        return ""

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
            self.traces[i] = filter_trace(
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

    def locate_by_sp_diff(self) -> Optional[Dict[str, Any]]:
        res = locate_by_sp_diff(
            self.visible_traces(),
            depth_km=self.current_params.depth,
            ttt=self.ttt,
            config=self.config,
        )
        if res is not None:
            p = self.current_params
            p.distance = res["distance"]
            p.origin = res["origin"]
        return res

    def fit_depth_phase(self) -> Optional[Dict[str, Any]]:
        p = self.current_params
        dist = p.distance if p.distance > 0 else 65.0
        res = fit_depth_phase(
            self.visible_traces(),
            distance_deg=dist,
            ttt=self.ttt,
            config=self.config,
        )
        if res is not None:
            p.depth = res["depth"]
            p.d_depth = res["d_depth"]
            p.depth_type = 3
        return res

    @staticmethod
    def _apply_magnitudes_to_pick(pk: PhasePick, res: Dict[str, Any]) -> None:
        pk.ampl = res["ampl_nm"]
        pk.period = res["period"]
        pk.mag_ml = res["ml"]
        pk.mag_mb = res["mb"]
        pk.mag_ms = res["ms"]
        pk.mag_mbb = res["mbb"]

    def measure_amplitude_magnitudes(
        self,
        trace_idx: int,
        rel_lo: float,
        rel_hi: float,
        prefer_phase: str = "P",
    ) -> Optional[Dict[str, Any]]:
        if not (0 <= trace_idx < len(self.traces)):
            return None
        tr = self.traces[trace_idx]
        p = self.current_params
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        dist = p.distance if p.distance > 0 else 50.0
        if has_hypo:
            st = self.config.get_station(tr.station)
            if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                dist, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                tr.distance = dist
        elif tr.distance > 0.0:
            dist = tr.distance
        res = compute_trace_window_magnitudes(tr, rel_lo, rel_hi, dist, p.depth)
        pk = tr.get_pick(prefer_phase) or (tr.picks[0] if tr.picks else None)
        if pk is not None:
            self._apply_magnitudes_to_pick(pk, res)
        p.mag_ml = res["ml"]
        p.mag_mb = res["mb"]
        p.mag_ms = res["ms"]
        p.mag_mbb = res["mbb"]
        return {**res, "station": tr.station}

    def _cmd_filter(self, args: List[str], quals: Dict[str, str]) -> str:
        f_expr = args[0] if args else (self.loaded_filters[0] if self.loaded_filters else "WWSSN_SP")
        list_spec = quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        mode = "REC" if ("REC" in quals or f_expr.upper().startswith("R")) else ("FFT" if "FFT" in quals else "AUTO")
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
            self.traces[i] = filter_trace(
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
        mode = "REC" if ("REC" in quals or f_expr.upper().startswith("R")) else ("FFT" if "FFT" in quals else "AUTO")
        for i in idxs:
            tr = self.traces[i]
            prev_global = tr.filter_name
            filtered = filter_trace(tr, f_expr, mode=mode, config=self.config)
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
            return "No traces available for AUTOCUT."

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
        spec = args[0] if args else "ALL"
        lo = float(args[1]) if len(args) > 2 else None
        hi = float(args[2]) if len(args) > 2 else None
        idxs = self.parse_trace_list(spec)
        for i in idxs:
            self.traces[i] = demean_trace(self.traces[i], lo, hi)
        return f"Demeaned {len(idxs)} trace(s)."

    def _cmd_trend(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = detrend_trace(self.traces[i])
        return f"Detrended {len(idxs)} trace(s)."

    def _cmd_despike(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        crit = float(args[1]) if len(args) > 1 else 5.0
        total_spikes = 0
        for i in idxs:
            self.traces[i], scnt, _dcnt = despike_trace(self.traces[i], critmul=crit)
            total_spikes += scnt
        return f"Despiked {len(idxs)} trace(s) ({total_spikes} spikes removed)."

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
        return f"Resampled {len(idxs)} trace(s) to dt={new_dt} s."

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
        return f"Decimated {len(idxs)} trace(s) by factor {fac}."

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
        return f"Cut {len(idxs)} trace(s) to [{lo:.2f}, {hi:.2f}] s."

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
        return f"Shifted {len(idxs)} trace(s) by {sec:+.3f} s."

    def _cmd_int(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = integrate_trace(self.traces[i])
        return f"Integrated {len(idxs)} trace(s)."

    def _cmd_derive(self, args: List[str], quals: Dict[str, str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        mode = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1
        for i in idxs:
            self.traces[i] = differentiate_trace(self.traces[i], mode=mode)
        return f"Differentiated {len(idxs)} trace(s)."

    def _cmd_hilbert(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = hilbert_trace(self.traces[i])
        return f"Computed Hilbert transform on {len(idxs)} trace(s)."

    def _cmd_envelope(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = envelope_trace(self.traces[i])
        return f"Computed analytic envelope on {len(idxs)} trace(s)."

    def _cmd_sphase(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        for i in idxs:
            self.traces[i] = instantaneous_phase_trace(self.traces[i])
        return f"Computed instantaneous phase on {len(idxs)} trace(s)."

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
        return f"Multiplied {len(idxs)} trace(s) by {fac:g}."

    def _cmd_sum(self, args: List[str], quals: Dict[str, str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        trs = [self.traces[i] for i in idxs]
        s_tr = sum_traces(trs, normalize_weights=("NONORM" not in quals))
        if s_tr is not None:
            self.traces.append(s_tr)
            return f"Appended SUM trace from {len(trs)} trace(s)."
        return "No traces to sum."

    def _cmd_pws(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        power = float(args[1]) if len(args) > 1 else 2.0
        trs = [self.traces[i] for i in idxs]
        p_tr = phase_weighted_stack(trs, power=power)
        if p_tr is not None:
            self.traces.append(p_tr)
            return f"Appended Phase-Weighted Stack (PWS, power={power}) from {len(trs)} trace(s)."
        return "No traces for PWS."

    def _cmd_mend(self, args: List[str]) -> str:
        if len(args) < 2:
            raise ValueError("Usage: MEND <idx_lo> <idx_hi> [trlist]")
        i0, i1 = int(args[0]), int(args[1])
        idxs = self.parse_trace_list(args[2] if len(args) > 2 else "ALL")
        for i in idxs:
            self.traces[i] = mend_trace(self.traces[i], i0, i1)
        return f"Mended samples [{i0}..{i1}] on {len(idxs)} trace(s)."

    def _cmd_attenuate(self, args: List[str]) -> str:
        t_star = float(args[0]) if args else 1.0
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        for i in idxs:
            self.traces[i] = attenuate_trace(self.traces[i], t_star=t_star)
        return f"Applied attenuation t*={t_star} s to {len(idxs)} trace(s)."

    def _cmd_spiking(self, args: List[str]) -> str:
        flen = int(args[0]) if args else 80
        prew = float(args[1]) if len(args) > 1 else 0.01
        if not self.traces:
            return "No trace available for spiking filter."
        tr = self.traces[0]
        f = spiking_filter_levinson(tr.data, filter_len=flen, prewhitening=prew)
        out = tr.copy(copy_picks=False)
        out.data = np.convolve(tr.data, f, mode="same")
        out.station = "SPIK"
        self.traces.append(out)
        return f"Computed spiking deconvolution trace (filter_len={flen})."

    def _cmd_deconv(self, args: List[str]) -> str:
        if len(self.traces) < 2:
            raise ValueError("DECONV requires at least 2 traces (numerator and denominator)")
        wl = float(args[0]) if args else 0.01
        res = spectral_division_deconvolve(self.traces[0], self.traces[1], water_level=wl)
        res.station = "DECV"
        self.traces.append(res)
        return "Appended water-level deconvolved trace DECV."

    def _cmd_create(self, args: List[str]) -> str:
        kind = args[0] if args else "KUEPPER"
        dur = float(args[1]) if len(args) > 1 else 60.0
        dt = float(args[2]) if len(args) > 2 else 0.05
        pos = float(args[3]) if len(args) > 3 else 15.0
        ampl = float(args[4]) if len(args) > 4 else 100.0
        width = float(args[5]) if len(args) > 5 else 2.0
        tr = create_synthetic_trace(kind, length_sec=dur, dt=dt, pos_sec=pos, ampl=ampl, width=width)
        self.traces.append(tr)
        return f"Created synthetic {kind.upper()} trace ({tr.length} samples)."

    def _cmd_rot2(self, args: List[str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        vis = self.visible_traces()
        tr_n = next((t for t in vis if t.comp == "N"), None)
        tr_e = next((t for t in vis if t.comp == "E"), None)
        if tr_n is None or tr_e is None:
            if len(vis) >= 2:
                tr_n, tr_e = vis[0], vis[1]
            else:
                raise ValueError("Need at least 2 horizontal traces (N, E) for 2D rotation")
        r_tr, t_tr = rotate_2d(tr_n, tr_e, az)
        self.traces.extend([r_tr, t_tr])
        return f"Rotated ({tr_n.station} N,E) by {az:.1f} deg -> appended R, T traces."

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
        return f"Rotated 3D ({tr_z.station} Z,N,E) by az={az:.1f}, inci={inc:.1f} -> appended L, Q, T traces."

    def _cmd_uvw(self, args: List[str], quals: Dict[str, str]) -> str:
        inv = "INV" in quals
        tr_z, tr_n, tr_e = self._find_zne_triplet()
        t1, t2, t3 = rotate_STS2_zne_to_uvw(tr_z, tr_n, tr_e, inverse=inv)
        self.traces.extend([t1, t2, t3])
        return f"Transformed STS-2 {'UVW->ZNE' if inv else 'ZNE->UVW'}."

    def _cmd_mdir(self, args: List[str]) -> str:
        tr_z, tr_n, tr_e = self._find_zne_triplet()
        lo = float(args[0]) if len(args) >= 2 else (self.time_window[0] if self.time_window else None)
        hi = float(args[1]) if len(args) >= 2 else (self.time_window[1] if self.time_window else None)
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
        return f"Applied polarization filter (mode={mode}, win={win}s, power={power}) -> appended 3 traces."

    def _cmd_beam(self, args: List[str], quals: Dict[str, str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        slo = float(args[1]) if len(args) > 1 else self.current_params.slowness
        ref = args[2] if len(args) > 2 else "CENTRE"
        vis = self.visible_traces()
        b_tr, _delays = compute_beam(vis, az, slo, ref_station=ref, config=self.config)
        self.traces.append(b_tr)
        msg = f"Created BEAM trace (az={az:.1f} deg, slo={slo:.2f} s/deg, ref={ref}) from {len(vis)} traces."
        self.emit(msg)
        return msg

    def _cmd_align(self, args: List[str]) -> str:
        ph = args[0] if args else (self.current_params.phase or "P")
        vis = self.visible_traces()
        a_tr = align_traces_on_phase(vis, phase_name=ph, create_sum_trace=True)
        if a_tr is not None:
            self.traces.append(a_tr)
            return f"Aligned traces on phase {ph} and appended ALIGN stack."
        return f"No picks found for phase {ph} to align."

    def _cmd_vespa(self, args: List[str]) -> str:
        az = float(args[0]) if args else self.current_params.azimuth
        s_lo = float(args[1]) if len(args) > 1 else 2.0
        s_hi = float(args[2]) if len(args) > 2 else 10.0
        s_step = float(args[3]) if len(args) > 3 else 0.5
        power = float(args[4]) if len(args) > 4 else 2.0
        vis = self.visible_traces()
        v_trs, _times, _slos, _mat = compute_vespagram(
            vis,
            azimuth_deg=az,
            slow_lo=s_lo,
            slow_hi=s_hi,
            slow_step=s_step,
            power=power,
            config=self.config,
        )
        self.traces.extend(v_trs)
        return f"Generated {len(v_trs)} Vespagram traces (az={az:.1f}, slo={s_lo}..{s_hi} s/deg, N={power})."

    def _cmd_fk(self, args: List[str], quals: Optional[Dict[str, str]] = None) -> str:
        q = quals or {}
        f_lo = float(args[0]) if args else 0.5
        f_hi = float(args[1]) if len(args) > 1 else 3.0
        max_slo = float(args[2]) if len(args) > 2 else 12.0
        lo = self.time_window[0] if self.time_window else None
        hi = self.time_window[1] if self.time_window else None
        vis = self.visible_traces()
        use_calib = self.calib_table if ("NOCALIB" not in q) else None
        res = compute_fk_analysis(
            vis,
            t_lo=lo,
            t_hi=hi,
            f_lo=f_lo,
            f_hi=f_hi,
            max_slowness=max_slo,
            config=self.config,
            calib_table=use_calib,
            apply_calibration=True if use_calib is not None else None,
        )
        self.last_fk_result = res
        self.current_params.azimuth = res["azimuth"]
        self.current_params.slowness = res["slowness"]
        cal_tag = " [calibrated]" if res.get("calib_applied") else ""
        msg = (
            f"FK Result{cal_tag}: Azimuth = {res['azimuth']:.1f} deg, Slowness = {res['slowness']:.2f} s/deg "
            f"(Vel = {res['apparent_velocity_kms']:.2f} km/s, Semblance = {res['semblance']:.3f})"
        )
        self.emit(msg)
        return msg

    def _cmd_calib(self, args: List[str], quals: Dict[str, str]) -> str:
        if "OFF" in quals or "CLEAR" in quals or (args and args[0].upper() in ("OFF", "NONE", "CLEAR")):
            self.calib_table = None
            msg = "Slowness calibration cleared."
            self.emit(msg)
            return msg

        tbl: Optional[SlownessCalibrationTable] = None
        src_desc = ""

        if "FILE" in quals:
            fpath = quals["FILE"]
            tbl = SlownessCalibrationTable.from_file(fpath)
            src_desc = fpath
        elif args and (
            os.path.isfile(args[0])
            or args[0].lower().endswith((".dat", ".txt"))
            or os.sep in args[0]
        ):
            fpath = args[0]
            tbl = SlownessCalibrationTable.from_file(fpath)
            src_desc = fpath
        elif "ARRAY" in quals or args:
            arr_val = quals.get("ARRAY", "")
            if arr_val and arr_val != "TRUE":
                target: Any = arr_val
                src_desc = arr_val
            elif args:
                target = args[0]
                src_desc = args[0]
            else:
                vis_stas = [t.station for t in self.visible_traces() if t.station]
                target = vis_stas if vis_stas else "GRF"
                src_desc = ",".join(vis_stas[:4]) if vis_stas else "GRF"
            tbl = load_calibration_for_array(target, inputs_dir=self.config.inputs_dir, config=self.config)
            if tbl is None or len(tbl) == 0:
                raise ValueError(f"No slowness calibration table found for array/station {src_desc!r}")
        else:
            if self.calib_table is not None:
                msg = f"Active slowness calibration table: {len(self.calib_table)} entries ({self.calib_table.source_path or 'in-memory'})."
                self.emit(msg)
                return msg
            vis_stas = [t.station for t in self.visible_traces() if t.station]
            target = vis_stas if vis_stas else "GRF"
            tbl = load_calibration_for_array(target, inputs_dir=self.config.inputs_dir, config=self.config)
            if tbl is None or len(tbl) == 0:
                raise ValueError("No slowness calibration table found for current workspace")
            src_desc = tbl.source_path or "array"

        self.calib_table = tbl
        msg = f"Loaded slowness calibration table ({len(tbl)} entries) from {tbl.source_path or src_desc}."

        # If workspace already has a slowness solution, apply calibration immediately unless /NOAPPLY
        p = self.current_params
        if "NOAPPLY" not in quals and p.slowness > 0.0:
            raw_slo = float(p.residuals.get("raw_slowness", p.slowness)) if p.residuals else float(p.slowness)
            raw_az = float(p.residuals.get("raw_azimuth", p.azimuth)) if p.residuals else float(p.azimuth)
            ph = p.phase or "P"
            vis = self.visible_traces()
            ref_tr = vis[0] if vis else None
            pk = ref_tr.get_pick(ph) if ref_tr else None
            onset_ep = pk.onset_epoch if (pk and pk.onset) else (
                parse_sh_time(p.origin) if p.origin else (ref_tr.start_epoch if ref_tr else 0.0)
            )
            st_code = p.source if p.source in self.config.stations else (ref_tr.station if ref_tr else "GRA1")
            cal_param = locate_by_slowness_azimuth(
                st_code,
                ph,
                onset_ep,
                raw_slo,
                raw_az,
                p.depth,
                config=self.config,
                ttt=self.ttt,
                calib_table=self.calib_table,
                apply_calibration=True,
            )
            self.param_sets[self.active_set_index] = cal_param
            for tr in vis:
                pk_tr = tr.get_pick(ph)
                if pk_tr is not None:
                    pk_tr.l_slowness = cal_param.slowness
                    pk_tr.l_azim = cal_param.azimuth
            msg += (
                f" Applied calibration: slo {raw_slo:.2f}->{cal_param.slowness:.2f} s/deg, "
                f"baz {raw_az:.1f}->{cal_param.azimuth:.1f} deg, Dist={cal_param.distance:.2f} deg."
            )

        self.emit(msg)
        return msg

    def _cmd_locate(self, args: List[str], quals: Dict[str, str]) -> str:
        ph = args[0] if args else (self.current_params.phase or "P")
        vis = self.visible_traces()
        use_calib = self.calib_table if ("NOCALIB" not in quals) else None

        if "GEIGER" in quals or "NET" in quals:
            all_picks = [p for t in vis for p in t.picks]
            param, _resids, diag = locate_network_geiger(
                all_picks,
                depth_km=self.current_params.depth,
                free_depth=("FREE" in quals),
                config=self.config,
                ttt=self.ttt,
                return_diagnostics=True,
            )
            if self.last_focal_mechanism is not None:
                setattr(param, "focal_mechanism", self.last_focal_mechanism)
                diag.context_data["focal_mechanism"] = self.last_focal_mechanism
            self.param_sets[self.active_set_index] = param
            self._emit_fit_diagnostic(diag)
            msg = (
                f"Network Location: Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, "
                f"Origin={param.origin}, Region={param.region_name}"
            )
            self.emit(msg)
            return msg

        if "SLOWNESS" in quals:
            ref_tr = vis[0] if vis else None
            pk = ref_tr.get_pick(ph) if ref_tr else None
            onset_ep = pk.onset_epoch if (pk and pk.onset) else (ref_tr.start_epoch if ref_tr else 0.0)
            st_code = ref_tr.station if ref_tr else "GRA1"
            param = locate_by_slowness_azimuth(
                st_code,
                ph,
                onset_ep,
                self.current_params.slowness,
                self.current_params.azimuth,
                self.current_params.depth,
                config=self.config,
                ttt=self.ttt,
                calib_table=use_calib,
                apply_calibration=True if use_calib is not None else None,
            )
            self.param_sets[self.active_set_index] = param
            msg = (
                f"Slowness Location: Dist={param.distance:.2f} deg, Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, "
                f"Origin={param.origin}, Region=[{param.region_id}] {param.region_name}"
            )
            self.emit(msg)
            return msg

        # Default: Plane-wave array location across phase picks for `ph` (deduplicated by station)
        picked_traces: List[Trace] = []
        arr_times: List[float] = []
        seen_stations: set[str] = set()
        for tr in vis:
            pk = tr.get_pick(ph)
            if pk is not None and pk.onset and tr.station not in seen_stations:
                seen_stations.add(tr.station)
                picked_traces.append(tr)
                arr_times.append(pk.onset_epoch)

        if len(picked_traces) < 3:
            raise ValueError(f"Plane-wave location requires at least 3 picks for phase '{ph}' (found {len(picked_traces)})")

        coords = self.config.relative_xy_km([t.station for t in picked_traces], ref_station="CENTRE")
        res = plane_wave_locate(
            arr_times,
            [c[0] for c in coords],
            [c[1] for c in coords],
            elev_m=[c[2] for c in coords],
            vp_kms=[c[3] for c in coords],
            dt=picked_traces[0].delta,
            calib_table=use_calib,
            apply_calibration=True if use_calib is not None else None,
            stations=[t.station for t in picked_traces],
            config=self.config,
        )
        for tr, resid in zip(picked_traces, res["residuals"]):
            pk = tr.get_pick(ph)
            if pk:
                pk.resid = resid
                pk.slowness = res.get("raw_slowness", res["slowness"]) if res.get("calib_applied") else res["slowness"]
                pk.beam_azim = res.get("raw_azimuth", res["azimuth"]) if res.get("calib_applied") else res["azimuth"]
                if res.get("calib_applied"):
                    pk.l_slowness = res["slowness"]
                    pk.l_azim = res["azimuth"]

        # Pass raw slowness/azimuth when calib_table is passed so locate_by_slowness_azimuth calibrates once and populates metadata
        raw_slo = res.get("raw_slowness", res["slowness"]) if res.get("calib_applied") else res["slowness"]
        raw_az = res.get("raw_azimuth", res["azimuth"]) if res.get("calib_applied") else res["azimuth"]
        param = locate_by_slowness_azimuth(
            picked_traces[0].station,
            ph,
            arr_times[0],
            raw_slo,
            raw_az,
            self.current_params.depth,
            config=self.config,
            ttt=self.ttt,
            calib_table=use_calib,
            apply_calibration=True if res.get("calib_applied") else False,
        )
        param.d_azimuth = res["d_azimuth"]
        param.d_slowness = res["d_slowness"]
        self.param_sets[self.active_set_index] = param
        cal_note = " [calibrated]" if res.get("calib_applied") else ""
        msg = (
            f"Plane Wave Fit{cal_note} ({len(picked_traces)} stations): BAZ = {res['azimuth']:.1f} +/- {res['d_azimuth']:.1f} deg, "
            f"Slowness = {res['slowness']:.2f} +/- {res['d_slowness']:.2f} s/deg -> "
            f"Dist = {param.distance:.1f} deg ({param.region_name})"
        )
        self.emit(msg)
        return msg

    def _cmd_identify(self, args: List[str], quals: Dict[str, str]) -> str:
        p = self.current_params
        depth_km = float(quals["DEPTH"]) if "DEPTH" in quals else float(p.depth)
        dist_override = float(quals["DIST"]) if "DIST" in quals else None
        cand_phases: Optional[List[str]] = None
        if "PHASES" in quals:
            cand_phases = [x.strip() for x in quals["PHASES"].split(",") if x.strip()]

        sigma_t = float(quals.get("SIGMA_T", 2.0))
        sigma_s = float(quals.get("SIGMA_S", 1.0))

        # Direct travel-time + slowness phase identification via /TIME=<sec>
        if "TIME" in quals:
            t_obs_rel = float(quals["TIME"])
            slo_meas = float(quals.get("SLO", p.slowness))
            dist_val = dist_override if dist_override is not None else float(p.distance)
            if dist_val <= 0.0:
                raise ValueError("IDENTIFY /TIME requires /DIST=<deg> or a valid distance in params")
            baz_meas = float(quals["BAZ"]) if "BAZ" in quals else (float(p.azimuth) if p.azimuth > 0.0 else None)
            baz_theo = float(quals["BAZ_THEO"]) if "BAZ_THEO" in quals else None
            cands = identify_phase(
                t_obs_rel_sec=t_obs_rel,
                slowness_meas_s_deg=slo_meas,
                dist_deg=dist_val,
                depth_km=depth_km,
                baz_meas_deg=baz_meas,
                baz_theo_deg=baz_theo,
                sigma_t=sigma_t,
                sigma_s=sigma_s,
                w_s=1.0 if slo_meas > 0.0 else 0.0,
                candidate_phases=cand_phases,
                ttt=self.ttt,
            )
            self.last_identify_results = cands
            if not cands:
                return "No matching seismic phases found."
            lines = [f"Phase Identification (t={t_obs_rel:.2f}s, slo={slo_meas:.2f}s/deg, dist={dist_val:.1f}deg):"]
            for rank, c in enumerate(cands[:5], start=1):
                lines.append(
                    f"  {rank}. {c['phase']:<7s} prob={c['probability']:.3f} chi2={c['chi2']:.2f} "
                    f"dt={c['t_resid']:+.2f}s ds={c['s_resid']:+.2f}s/deg"
                )
            msg = "\n".join(lines)
            self.emit(msg)
            return msg

        list_spec = quals.get("LIST", args[0] if args else "ALL")
        idxs = self.parse_trace_list(list_spec)
        target_traces = [self.traces[i] for i in idxs if self.traces[i].picks]
        if not target_traces:
            raise ValueError("IDENTIFY requires picked traces in workspace (or /TIME=... /DIST=...)")

        if "ORIGIN" in quals:
            try:
                orig_ep = float(quals["ORIGIN"])
            except ValueError:
                orig_ep = parse_sh_time(quals["ORIGIN"])
        elif p.origin:
            orig_ep = parse_sh_time(p.origin)
        elif target_traces[0].torig > 0.0:
            orig_ep = float(target_traces[0].torig)
        else:
            orig_ep = float(target_traces[0].start_epoch)

        update_names = ("NOUPDATE" not in quals) and ("DRY" not in quals)
        max_chi2 = float(quals["MAX_CHI2"]) if "MAX_CHI2" in quals else None
        def_slo = float(quals["SLO"]) if "SLO" in quals else (float(p.slowness) if p.slowness > 0.0 else None)
        def_baz = float(quals["BAZ"]) if "BAZ" in quals else (float(p.azimuth) if p.azimuth > 0.0 else None)
        baz_theo = float(quals["BAZ_THEO"]) if "BAZ_THEO" in quals else None

        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        all_results: List[Dict[str, Any]] = []
        for tr in target_traces:
            if dist_override is not None:
                dist_i = dist_override
            elif has_hypo:
                st = self.config.get_station(tr.station)
                if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                    dist_i, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = dist_i
                elif tr.lat != 0.0 or tr.lon != 0.0:
                    dist_i, _, _ = sphere_diff(tr.lat, tr.lon, p.lat, p.lon)
                    tr.distance = dist_i
                else:
                    dist_i = float(tr.distance) if tr.distance > 0.0 else float(p.distance)
            elif tr.distance > 0.0:
                dist_i = float(tr.distance)
            else:
                dist_i = float(p.distance)

            if dist_i <= 0.0:
                raise ValueError(f"IDENTIFY requires epicentral distance for trace {tr.station} (set params.distance, hypocenter, or /DIST=<deg>)")

            res_tr = identify_trace_picks(
                [tr],
                origin_epoch=orig_ep,
                dist_deg=dist_i,
                depth_km=depth_km,
                baz_theo_deg=baz_theo,
                default_slowness=def_slo,
                default_baz=def_baz,
                update_pick_names=update_names,
                max_chi2=max_chi2,
                sigma_t=sigma_t,
                sigma_s=sigma_s,
                candidate_phases=cand_phases,
                ttt=self.ttt,
                config=self.config,
            )
            all_results.extend(res_tr)

        self.last_identify_results = all_results
        if not all_results:
            return "No phase candidates matched."

        lines = [f"Identified {len(all_results)} pick(s):"]
        for r in all_results:
            top3 = ", ".join(f"{c['phase']}({c['probability']:.2f})" for c in r["candidates"][:3])
            lines.append(
                f"  {r['station']} {r['old_phase']} -> {r['identified_phase']} "
                f"(prob={r['probability']:.3f}, chi2={r['chi2']:.2f}, dt={r['t_resid']:+.2f}s) | top: {top3}"
            )
        msg = "\n".join(lines)
        self.emit(msg)
        return msg

    def _cmd_pick(self, args: List[str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "ALL")
        lo = self.time_window[0] if self.time_window else None
        hi = self.time_window[1] if self.time_window else None
        count = 0
        for i in idxs:
            pk = pick_onset_baer_kradolfer(self.traces[i], t_lo=lo, t_hi=hi)
            if pk is not None:
                self.traces[i].add_or_replace_pick(pk)
                count += 1
        msg = f"Auto-picked P onset on {count} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_corrpick(self, args: List[str]) -> str:
        ph = args[0] if args else "P"
        vis = self.visible_traces()
        pks = corrpick_array(vis, ref_trace_idx=0, phase_name=ph)
        msg = f"Cross-correlation picked '{ph}' across {len(pks)} trace(s)."
        self.emit(msg)
        return msg

    def _cmd_corr(self, args: List[str]) -> str:
        vis = self.visible_traces()
        if len(vis) < 2:
            raise ValueError("CORR requires at least 2 traces")
        t1_idx = int(args[0]) - 1 if len(args) > 0 and args[0].isdigit() else 0
        t2_idx = int(args[1]) - 1 if len(args) > 1 and args[1].isdigit() else 1
        lo = float(args[2]) if len(args) > 3 else 0.0
        hi = float(args[3]) if len(args) > 3 else min(10.0, vis[t1_idx].duration)
        lags, corrs, best_lag, max_c = correlate_traces(vis[t1_idx], vis[t2_idx], lo, hi)
        tr_c = Trace(
            data=corrs,
            delta=vis[t1_idx].delta,
            station="CORR",
            comp="Z",
            start_epoch=vis[t1_idx].start_epoch + float(lags[0]),
            comment=f"Max corr={max_c:.3f} at lag={best_lag:+.3f}s",
        )
        self.traces.append(tr_c)
        return f"Cross-correlation peak = {max_c:.3f} at lag = {best_lag:+.3f} s."

    def _cmd_mag(self, args: List[str]) -> str:
        ph = args[0] if args else "P"
        p = self.current_params
        global_dist = p.distance if p.distance > 0 else 50.0
        depth = p.depth
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        vis = self.visible_traces()
        mls, mbs, mss, mbbs = [], [], [], []
        dists_used = []

        for tr in vis:
            dist_i = global_dist
            if has_hypo:
                st = self.config.get_station(tr.station)
                if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                    dist_i, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = dist_i
            elif tr.distance > 0.0:
                dist_i = tr.distance

            pk = tr.get_pick(ph)
            if pk is not None and pk.onset:
                rel_t = pk.onset_epoch - tr.start_epoch
                lo, hi = max(0.0, rel_t), min(tr.duration, rel_t + 8.0)
            elif self.time_window:
                lo, hi = self.time_window[0], self.time_window[1]
            else:
                lo, hi = 0.0, tr.duration

            res = compute_trace_window_magnitudes(tr, lo, hi, dist_i, depth)
            if res["ampl_nm"] > 0:
                mls.append(res["ml"])
                mbs.append(res["mb"])
                mss.append(res["ms"])
                mbbs.append(res["mbb"])
                dists_used.append(dist_i)
                if pk is not None:
                    self._apply_magnitudes_to_pick(pk, res)

        n_mags = len(mbs)
        p.mags_count = n_mags
        if n_mags > 0:
            p.mag_ml = float(np.mean(mls))
            p.mag_mb = float(np.mean(mbs))
            p.mag_ms = float(np.mean(mss))
            p.mag_mbb = float(np.mean(mbbs))
            if n_mags >= 2:
                p.d_ml = float(np.std(mls, ddof=1))
                p.d_mb = float(np.std(mbs, ddof=1))
                p.d_ms = float(np.std(mss, ddof=1))
                p.d_mbb = float(np.std(mbbs, ddof=1))
            else:
                p.d_ml = 0.0
                p.d_mb = 0.0
                p.d_ms = 0.0
                p.d_mbb = 0.0
        avg_dist = float(np.mean(dists_used)) if dists_used else global_dist
        msg = (
            f"Mean Magnitudes ({n_mags} traces, Dist={avg_dist:.1f} deg): "
            f"ML={p.mag_ml:.2f}, mb={p.mag_mb:.2f}, "
            f"MS={p.mag_ms:.2f}, mB={p.mag_mbb:.2f}"
        )
        self.emit(msg)
        return msg

    def _cmd_surfmag(self, args: List[str], quals: Dict[str, str]) -> str:
        p = self.current_params
        list_spec = quals.get("LIST")
        if list_spec is None and args:
            tok = args[0]
            if tok.upper() in ("ALL", "SEL", "SELECTED", "*") or "," in tok or "-" in tok or tok.isdigit():
                list_spec = tok
        if list_spec is None:
            list_spec = "ALL"

        idxs = self.parse_trace_list(list_spec)
        selected_traces = [self.traces[i] for i in idxs]
        if not selected_traces:
            raise ValueError("SURFMAG requires at least one trace in workspace")

        vertical_only = ("ALLCOMP" not in quals)
        if vertical_only:
            z_traces = [tr for tr in selected_traces if (tr.comp or "").upper() == "Z"]
            eff_traces = z_traces if z_traces else selected_traces
        else:
            eff_traces = selected_traces

        p_lo = float(quals.get("PERIOD_LO", quals.get("TMIN", 18.0)))
        p_hi = float(quals.get("PERIOD_HI", quals.get("TMAX", 22.0)))
        lo = float(quals["T_LO"]) if "T_LO" in quals else (self.time_window[0] if self.time_window else None)
        hi = float(quals["T_HI"]) if "T_HI" in quals else (self.time_window[1] if self.time_window else None)
        input_type = (
            "velocity"
            if ("VELOCITY" in quals or quals.get("INPUT", "").lower().startswith("v"))
            else "displacement"
        )
        dist_override = float(quals["DIST"]) if "DIST" in quals else None
        has_hypo = (p.lat != 0.0 or p.lon != 0.0)
        global_dist = p.distance if p.distance > 0.0 else 50.0

        dists: List[float] = []
        for tr in eff_traces:
            if dist_override is not None:
                dist_i = dist_override
            elif has_hypo:
                st = self.config.get_station(tr.station)
                if st is not None and (st.lat != 0.0 or st.lon != 0.0):
                    dist_i, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
                    tr.distance = dist_i
                elif tr.lat != 0.0 or tr.lon != 0.0:
                    dist_i, _, _ = sphere_diff(tr.lat, tr.lon, p.lat, p.lon)
                    tr.distance = dist_i
                else:
                    dist_i = float(tr.distance) if tr.distance > 0.0 else global_dist
            elif tr.distance > 0.0:
                dist_i = float(tr.distance)
            else:
                dist_i = global_dist
            dists.append(dist_i)

        scan_res = scan_surface_waves(
            eff_traces,
            distances_deg=dists,
            period_lo=p_lo,
            period_hi=p_hi,
            t_lo=lo,
            t_hi=hi,
            vertical_only=False,
            input_type=input_type,
        )
        self.last_surfmag_result = scan_res

        valid_ms = [float(s["ms"]) for s in scan_res["stations"] if s.get("ms") is not None and s["ms"] > 0.0]
        if scan_res["ms_mean"] is not None:
            p.mag_ms = float(scan_res["ms_mean"])
            p.d_ms = float(scan_res["ms_std"] or 0.0)
            p.mags_count = max(p.mags_count, len(valid_ms))

        for tr, st_res in zip(eff_traces, scan_res["stations"]):
            pk = tr.get_pick("LR") or tr.get_pick("LQ") or tr.get_pick("R") or (tr.picks[0] if tr.picks else None)
            if pk is None:
                pk = PhasePick(
                    name="LR",
                    onset=format_sh_time(tr.start_epoch + float(st_res["time_sec"])),
                    source="a",
                    filter_name=tr.filter_name,
                    station=tr.station,
                    comp=tr.comp,
                    chan1=tr.chan1,
                    chan2=tr.chan2,
                )
                tr.add_or_replace_pick(pk)
            pk.ampl = float(st_res["ampl_nm"])
            pk.ampl_time = float(st_res["time_sec"])
            pk.period = float(st_res["period"])
            if st_res["ms"] is not None:
                pk.mag_ms = float(st_res["ms"])

        ms_str = f"{p.mag_ms:.2f} +/- {p.d_ms:.2f}" if valid_ms else "N/A"
        max_ampl = max((float(s["ampl_nm"]) for s in scan_res["stations"]), default=0.0)
        msg = (
            f"Surface Wave Magnitude ({len(eff_traces)} Z-trace(s), {p_lo:.0f}-{p_hi:.0f}s): "
            f"MS={ms_str}, max A={max_ampl:.1f} nm"
        )
        self.emit(msg)
        return msg

    def _cmd_theo(self, args: List[str]) -> str:
        dist = float(args[0]) if args else self.current_params.distance
        depth = float(args[1]) if len(args) > 1 else self.current_params.depth
        vis = self.visible_traces()
        if not vis:
            return "No traces in workspace."
        orig_ep = parse_sh_time(self.current_params.origin) if self.current_params.origin else vis[0].start_epoch
        arrs = self.ttt.theoretical_arrivals(dist, depth, orig_ep)
        for tr in vis:
            tr.theo_picks.update(arrs)
        msg = f"Computed {len(arrs)} theoretical phase arrivals for Dist={dist:.1f} deg, Depth={depth:.1f} km."
        self.emit(msg)
        return msg

    def _cmd_spectrum(self, args: List[str], quals: Dict[str, str]) -> str:
        idxs = self.parse_trace_list(args[0] if args else "1")
        if not idxs:
            return "No trace selected for spectrum."
        tr = self.traces[idxs[0]]
        lo = self.time_window[0] if self.time_window else None
        hi = self.time_window[1] if self.time_window else None
        if "MEM" in quals:
            freqs, amps = compute_mem_spectrum(tr, lo, hi)
            kind = "MEM"
        elif "PSD" in quals:
            freqs, amps = compute_welch_power_spectrum(tr, lo, hi)
            kind = "PSD"
        else:
            freqs, amps = compute_fft_spectrum(tr, lo, hi)
            kind = "FFT"
        peak_i = int(np.argmax(amps)) if amps.size > 0 else 0
        f_peak = float(freqs[peak_i]) if freqs.size > 0 else 0.0
        a_peak = float(amps[peak_i]) if amps.size > 0 else 0.0
        msg = f"{kind} Spectrum ({tr.station} {tr.channel3}): Peak amplitude = {a_peak:.3e} at {f_peak:.3f} Hz"
        self.emit(msg)
        return msg

    def _cmd_fit(self, args: List[str], quals: Dict[str, str]) -> str:
        rem = list(args)
        mode: Optional[str] = None
        if "BOATWRIGHT" in quals:
            mode = "BOATWRIGHT"
        elif "BRUNE" in quals:
            mode = "BRUNE"
        elif any(k in quals for k in ("POL", "POLARIZATION", "MDIR")):
            mode = "POL"
        elif any(k in quals for k in ("FOCAL", "FM", "BEACHBALL")):
            mode = "FOCAL"
        elif any(k in quals for k in ("LOCATE", "GEIGER", "NET")):
            mode = "LOCATE"
        elif rem and rem[0].upper() in ("BRUNE", "BOATWRIGHT", "POL", "POLARIZATION", "MDIR", "FOCAL", "FM", "BEACHBALL", "LOCATE", "GEIGER", "NET"):
            tok = rem.pop(0).upper()
            if tok in ("POLARIZATION", "MDIR"):
                mode = "POL"
            elif tok in ("FM", "BEACHBALL"):
                mode = "FOCAL"
            elif tok in ("GEIGER", "NET"):
                mode = "LOCATE"
            else:
                mode = tok

        if mode is None:
            raise ValueError("FIT requires a mode qualifier: /BRUNE, /BOATWRIGHT, /POL, /FOCAL, or /LOCATE")

        if mode in ("BRUNE", "BOATWRIGHT"):
            model_name = "boatwright" if mode == "BOATWRIGHT" else "brune"
            list_spec = quals.get("LIST")
            t_lo = float(quals["T_LO"]) if "T_LO" in quals else (self.time_window[0] if self.time_window else None)
            t_hi = float(quals["T_HI"]) if "T_HI" in quals else (self.time_window[1] if self.time_window else None)

            if list_spec is None and rem:
                if len(rem) >= 3:
                    list_spec = rem[0]
                    t_lo = float(rem[1])
                    t_hi = float(rem[2])
                elif len(rem) == 2:
                    if "." in rem[0] or not rem[0].isdigit() or int(rem[0]) > len(self.traces):
                        t_lo = float(rem[0])
                        t_hi = float(rem[1])
                    else:
                        list_spec = rem[0]
                else:
                    list_spec = rem[0]

            if list_spec is None:
                list_spec = "SEL"
            idxs = self.parse_trace_list(list_spec)
            if not idxs:
                raise ValueError("FIT /BRUNE requires at least one trace in workspace")
            tr = self.traces[idxs[0]]

            noise_lo = float(quals["NOISE_LO"]) if "NOISE_LO" in quals else None
            noise_hi = float(quals["NOISE_HI"]) if "NOISE_HI" in quals else None
            f_lo = float(quals.get("F_LO", 0.1))
            f_hi = float(quals["F_HI"]) if "F_HI" in quals else None
            input_type = (
                "velocity"
                if ("VELOCITY" in quals or quals.get("INPUT", "").lower().startswith("v"))
                else "displacement"
            )
            phase = quals.get("PHASE", self.current_params.phase or "P")

            phys_kwargs: Dict[str, Any] = {}
            if "DIST_M" in quals:
                phys_kwargs["distance_m"] = float(quals["DIST_M"])
            elif "DIST" in quals:
                phys_kwargs["distance_m"] = float(quals["DIST"]) * 111194.9266
            elif tr.distance > 0.0:
                phys_kwargs["distance_m"] = float(tr.distance) * 111194.9266
            elif self.current_params.distance > 0.0:
                phys_kwargs["distance_m"] = float(self.current_params.distance) * 111194.9266

            if "VEL" in quals or "VEL_M_S" in quals:
                phys_kwargs["vel_m_s"] = float(quals.get("VEL", quals.get("VEL_M_S")))
            if "RHO" in quals or "RHO_KG_M3" in quals:
                phys_kwargs["rho_kg_m3"] = float(quals.get("RHO", quals.get("RHO_KG_M3")))
            if "WAVE" in quals:
                phys_kwargs["wave_type"] = quals["WAVE"]
            if "NOTSTAR" in quals:
                phys_kwargs["fit_t_star"] = False

            inv_res = invert_brune_spectrum(
                tr,
                t_lo=t_lo,
                t_hi=t_hi,
                noise_lo=noise_lo,
                noise_hi=noise_hi,
                model=model_name,
                f_lo=f_lo,
                f_hi=f_hi,
                input_type=input_type,
                **phys_kwargs,
            )
            diag = build_brune_fit_diagnostic(inv_res, station=tr.station, phase=phase)

            p = self.current_params
            p.corner_freq = float(inv_res["fc"])
            p.moment_m0 = float(inv_res["m0"])
            p.mag_mw = float(inv_res["mw"])
            p.rmse = float(inv_res["rms_log10"])
            p.wrmse = float(inv_res.get("wrmse_log10", inv_res["rms_log10"]))
            p.chi2_reduced = float(inv_res.get("chi2_reduced", 0.0))
            p.condition_number = float(inv_res.get("condition_number", 1.0))

            self.last_brune_result = inv_res
            self._emit_fit_diagnostic(diag)
            msg = (
                f"{model_name.title()} Fit ({tr.station} {tr.channel3}): "
                f"Omega0={inv_res['omega0']:.3e} nm*s, fc={inv_res['fc']:.3f} Hz, "
                f"t*={inv_res['t_star']:.3f} s, M0={inv_res['m0']:.3e} N*m, "
                f"Mw={inv_res['mw']:.2f}, dSigma={inv_res['stress_drop_bar']:.2f} bar, "
                f"RMS={inv_res['rms_log10']:.4f}"
            )
            self.emit(msg)
            return msg

        if mode == "POL":
            prefer_station = quals.get("STATION")
            if prefer_station is None and rem and not rem[0].replace(".", "").replace("-", "").isdigit():
                prefer_station = rem.pop(0)
            tr_z, tr_n, tr_e = self._find_zne_triplet(prefer_station=prefer_station)
            lo = float(quals["T_LO"]) if "T_LO" in quals else (
                float(rem[0]) if len(rem) >= 2 else (self.time_window[0] if self.time_window else None)
            )
            hi = float(quals["T_HI"]) if "T_HI" in quals else (
                float(rem[1]) if len(rem) >= 2 else (self.time_window[1] if self.time_window else None)
            )
            diag = analyze_polarization_diagnostic(tr_z, tr_n, tr_e, t_lo=lo, t_hi=hi)
            ctx = diag.context_data
            p = self.current_params
            p.azimuth = float(ctx.get("baz", 0.0))
            p.d_azimuth = float(ctx.get("sigma_theta_deg", 0.0))
            p.incidence = float(ctx.get("incidence", 0.0))
            p.rmse = float(diag.rmse)
            p.wrmse = float(diag.wrmse)
            p.chi2_reduced = float(diag.chi2_reduced)
            p.condition_number = float(diag.condition_number)
            for t in (tr_z, tr_n, tr_e):
                t.inci = p.incidence
            self.last_mdir_result = {
                "azimuth": p.azimuth,
                "incidence": p.incidence,
                "rectilinearity": float(ctx.get("rectilinearity", 0.0)),
                "planarity": float(ctx.get("planarity", 0.0)),
                "lambda1": float(ctx.get("lambda1", 0.0)),
                "lambda2": float(ctx.get("lambda2", 0.0)),
                "lambda3": float(ctx.get("lambda3", 0.0)),
                "rms_pol": float(ctx.get("rms_pol", 0.0)),
                "sigma_theta_deg": float(ctx.get("sigma_theta_deg", 0.0)),
                "sigma_phi_deg": float(ctx.get("sigma_phi_deg", 0.0)),
            }
            self._emit_fit_diagnostic(diag)
            msg = (
                f"Polarization Fit ({tr_z.station}): BAZ={p.azimuth:.1f} +/- {ctx.get('sigma_theta_deg', 0.0):.1f} deg, "
                f"Inc={p.incidence:.1f} +/- {ctx.get('sigma_phi_deg', 0.0):.1f} deg, "
                f"Rect={ctx.get('rectilinearity', 0.0):.3f}, RMS_pol={diag.rmse:.4f}"
            )
            self.emit(msg)
            return msg

        if mode == "FOCAL":
            vis = self.visible_traces()
            p = self.current_params
            ev_lat = float(quals["LAT"]) if "LAT" in quals else float(p.lat)
            ev_lon = float(quals["LON"]) if "LON" in quals else float(p.lon)
            depth_km = float(quals["DEPTH"]) if "DEPTH" in quals else float(p.depth)
            grid_step = float(quals.get("STEP", 10.0))
            pols = extract_polarities_from_traces(
                vis,
                ev_lat,
                ev_lon,
                depth_km=depth_km,
                config=self.config,
                ttt=self.ttt,
            )
            if not pols:
                raise ValueError(
                    "FIT /FOCAL requires at least one trace pick with non-zero polarity (sign +1/-1 or 'c'/'d')"
                )
            fm_res = solve_focal_mechanism(pols, grid_step_deg=grid_step, refine=("NOREFINE" not in quals))
            pol_tuples = [
                (float(m["azimuth"]), float(m["takeoff"]), int(m["observed"]))
                for m in fm_res["station_matches"]
            ]
            fm_payload: Dict[str, Any] = {
                **fm_res,
                "misfit": float(fm_res["misfit_ratio"]),
                "polarities": pol_tuples,
            }
            self.last_focal_mechanism = fm_payload
            setattr(p, "focal_mechanism", fm_payload)

            residuals: List[ResidualEntry] = []
            for m in fm_res["station_matches"]:
                obs_v = float(m["observed"])
                pred_v = float(m["predicted_sign"])
                res_v = obs_v - pred_v
                residuals.append(
                    ResidualEntry(
                        identifier=f"{m['station']}.P.FM",
                        observed=obs_v,
                        calculated=pred_v,
                        residual=res_v,
                        normalized_residual=abs(res_v),
                        weight=float(m["weight"]),
                        uncertainty=1.0,
                        azimuth_deg=float(m["azimuth"]),
                        station=str(m["station"]),
                        phase="P",
                        channel="Z",
                        is_outlier=not bool(m["matched"]),
                    )
                )

            fm_params = {
                "strike": ParameterUncertainty(
                    name="strike",
                    value=float(fm_res["strike"]),
                    standard_error=grid_step,
                    confidence_interval_95=(float(fm_res["strike"]) - grid_step, float(fm_res["strike"]) + grid_step),
                    unit="deg",
                ),
                "dip": ParameterUncertainty(
                    name="dip",
                    value=float(fm_res["dip"]),
                    standard_error=grid_step,
                    confidence_interval_95=(max(0.0, float(fm_res["dip"]) - grid_step), min(90.0, float(fm_res["dip"]) + grid_step)),
                    unit="deg",
                ),
                "rake": ParameterUncertainty(
                    name="rake",
                    value=float(fm_res["rake"]),
                    standard_error=grid_step,
                    confidence_interval_95=(float(fm_res["rake"]) - grid_step, float(fm_res["rake"]) + grid_step),
                    unit="deg",
                ),
            }
            diag = FitDiagnosticResult(
                fit_type=FitType.HYPOCENTER_GEIGER,
                title=f"Double-Couple Focal Mechanism Fit ({fm_res['n_obs']} polarities)",
                converged=bool(fm_res["converged"]),
                iterations=1,
                dof=max(1, int(fm_res["n_obs"]) - 3),
                rmse=float(fm_res["misfit_ratio"]),
                wrmse=float(fm_res["misfit_ratio"]),
                parameters=fm_params,
                residuals=residuals,
                outlier_count=int(fm_res["n_misfit"]),
                context_data={"focal_mechanism": fm_payload},
            )
            self._emit_fit_diagnostic(diag)
            msg = (
                f"Focal Mechanism Fit ({fm_res['n_obs']} polarities): "
                f"Strike={fm_res['strike']:.1f} deg, Dip={fm_res['dip']:.1f} deg, Rake={fm_res['rake']:.1f} deg "
                f"(Aux: {fm_res['strike2']:.1f}/{fm_res['dip2']:.1f}/{fm_res['rake2']:.1f}, "
                f"Misfit={fm_res['n_misfit']}/{fm_res['n_obs']} [{fm_res['misfit_ratio']:.2f}])"
            )
            self.emit(msg)
            return msg

        # mode == "LOCATE"
        vis = self.visible_traces()
        all_picks = [pk for t in vis for pk in t.picks]
        depth_km = float(quals["DEPTH"]) if "DEPTH" in quals else float(self.current_params.depth)
        free_depth = ("FREE" in quals)
        init_lat = float(quals["LAT"]) if "LAT" in quals else None
        init_lon = float(quals["LON"]) if "LON" in quals else None
        excl = [s.strip() for s in quals["EXCLUDE"].split(",") if s.strip()] if "EXCLUDE" in quals else None
        param, _resids, diag = locate_network_geiger(
            all_picks,
            depth_km=depth_km,
            free_depth=free_depth,
            initial_lat=init_lat,
            initial_lon=init_lon,
            config=self.config,
            ttt=self.ttt,
            excluded_stations=excl,
            return_diagnostics=True,
        )
        if self.last_focal_mechanism is not None:
            setattr(param, "focal_mechanism", self.last_focal_mechanism)
            diag.context_data["focal_mechanism"] = self.last_focal_mechanism
        self.param_sets[self.active_set_index] = param
        self._emit_fit_diagnostic(diag)
        msg = (
            f"Geiger Location Fit: Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, "
            f"RMSE={param.rmse:.3f} s, Origin={param.origin}, Region={param.region_name}"
        )
        self.emit(msg)
        return msg

    def _find_zne_triplet(self, prefer_station: Optional[str] = None) -> Tuple[Trace, Trace, Trace]:
        vis = self.visible_traces()
        candidates: List[str] = []
        if prefer_station:
            candidates.append(prefer_station.strip().upper())
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
        if len(vis) >= 3:
            return vis[0], vis[1], vis[2]
        raise ValueError("Operation requires 3-component (Z, N, E) traces in workspace")
