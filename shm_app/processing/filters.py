"""SHM filter parser (.FLF, .FLR, .FLT), FFT pole-zero filter, recursive IIR cascade filter,
Butterworth design, seismometer simulation filters, and true ground amplitude restitution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import re
from typing import List, Optional, Tuple
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace

FILTER_MAGIC = 1357913578

# Standard simulation filters in SHM
STANDARD_SIMULATION_FILTERS = [
    "WWSSN_SP",
    "WWSSN_LP",
    "SRO_LP",
    "KIRNOS",
    "WOODAND",
    "LRSM_SP",
    "LRSM_LP",
    "STANDARD_BP",
    "DISPLACE",
]


@dataclass
class FilterStage:
    """Single stage of an SH filter (.FLF store_id=1, .FLT store_id=2, .FLR store_id=3)."""

    store_id: int  # 1=poles/zeros, 2=tabulated freq, 3=recursive IIR
    norm: float = 1.0
    dt: float = 0.05  # used for store_id=3
    zeros: List[complex] = field(default_factory=list)  # store_id=1 (rad/s)
    poles: List[complex] = field(default_factory=list)  # store_id=1 (rad/s)
    freqs_hz: List[float] = field(default_factory=list)  # store_id=2
    amplitudes: List[float] = field(default_factory=list)  # store_id=2
    phases_deg: List[float] = field(default_factory=list)  # store_id=2
    num_a: List[float] = field(default_factory=list)  # store_id=3 numerator a_0..a_k
    den_b: List[float] = field(default_factory=list)  # store_id=3 denominator b_0..b_l
    comment: str = ""

    def inverted(self) -> "FilterStage":
        """Return the inverse transfer function of this stage (matching '-' prefix in SHM)."""
        if self.store_id == 1:
            return FilterStage(
                store_id=1,
                norm=1.0 / self.norm if self.norm != 0.0 else 1.0,
                zeros=list(self.poles),
                poles=list(self.zeros),
                comment=f"INV({self.comment})",
            )
        if self.store_id == 2:
            inv_ampls = [1.0 / a if a > 1e-30 else 0.0 for a in self.amplitudes]
            inv_phases = [-p for p in self.phases_deg]
            return FilterStage(
                store_id=2,
                norm=1.0 / self.norm if self.norm != 0.0 else 1.0,
                freqs_hz=list(self.freqs_hz),
                amplitudes=inv_ampls,
                phases_deg=inv_phases,
                comment=f"INV({self.comment})",
            )
        if self.store_id == 3:
            return FilterStage(
                store_id=3,
                norm=1.0 / self.norm if self.norm != 0.0 else 1.0,
                dt=self.dt,
                num_a=list(self.den_b),
                den_b=list(self.num_a),
                comment=f"INV({self.comment})",
            )
        return self


def parse_filter_file(filepath: str, pos: Optional[int] = None) -> List[FilterStage]:
    """Parse an SH .FLF, .FLR, or .FLT filter file into a list of FilterStages."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Filter file not found: {filepath}")

    with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
        raw_lines = [ln.strip() for ln in f]

    # Split into stages by '@'
    stage_chunks: List[List[str]] = []
    cur_chunk: List[str] = []
    for ln in raw_lines:
        if not ln:
            continue
        if ln.startswith("@"):
            if cur_chunk:
                stage_chunks.append(cur_chunk)
                cur_chunk = []
            continue
        cur_chunk.append(ln)
    if cur_chunk:
        stage_chunks.append(cur_chunk)

    stages: List[FilterStage] = []
    for idx, chunk in enumerate(stage_chunks, start=1):
        if pos is not None and idx != pos:
            continue
        stg = _parse_single_stage(chunk)
        if stg is not None:
            stages.append(stg)
    return stages


def _parse_single_stage(lines: List[str]) -> Optional[FilterStage]:
    comments = [ln[1:].strip() for ln in lines if ln.startswith("!")]
    data_lines = [ln for ln in lines if not ln.startswith("!")]
    if len(data_lines) < 2:
        return None

    # Flatten tokens from data_lines
    tokens: List[str] = []
    for ln in data_lines:
        tokens.extend(ln.split())

    if len(tokens) < 2:
        return None

    first_tok = int(tokens[0])
    if first_tok == FILTER_MAGIC:
        store_id = int(tokens[1])
        t_off = 2
    elif first_tok in (1, 2, 3):
        store_id = first_tok
        t_off = 1
    else:
        raise ValueError(f"Invalid filter magic number: {first_tok}")

    comment_str = " | ".join(comments)

    if store_id == 1:
        # Rational s-plane poles & zeros (rad/s)
        norm = float(tokens[t_off])
        n_zeros = int(tokens[t_off + 1])
        idx = t_off + 2
        zeros: List[complex] = []
        for _ in range(n_zeros):
            val_str = tokens[idx]
            idx += 1
            if val_str.startswith("(") and "," in val_str:
                re_s, im_s = val_str.strip("()").split(",", 1)
                zeros.append(complex(float(re_s), float(im_s)))
            else:
                re_v = float(val_str)
                im_v = float(tokens[idx])
                idx += 1
                zeros.append(complex(re_v, im_v))
        n_poles = int(tokens[idx])
        idx += 1
        poles: List[complex] = []
        for _ in range(n_poles):
            val_str = tokens[idx]
            idx += 1
            if val_str.startswith("(") and "," in val_str:
                re_s, im_s = val_str.strip("()").split(",", 1)
                poles.append(complex(float(re_s), float(im_s)))
            else:
                re_v = float(val_str)
                im_v = float(tokens[idx])
                idx += 1
                poles.append(complex(re_v, im_v))
        return FilterStage(
            store_id=1,
            norm=norm,
            zeros=zeros,
            poles=poles,
            comment=comment_str,
        )

    elif store_id == 2:
        # Tabulated frequency response: n_entries, then (freq_hz, ampl, phase_deg)
        n_entries = int(tokens[t_off])
        idx = t_off + 1
        freqs: List[float] = []
        ampls: List[float] = []
        phases: List[float] = []
        for _ in range(n_entries):
            if idx + 2 >= len(tokens):
                break
            freqs.append(float(tokens[idx]))
            ampls.append(float(tokens[idx + 1]))
            phases.append(float(tokens[idx + 2]))
            idx += 3
        return FilterStage(
            store_id=2,
            norm=1.0,
            freqs_hz=freqs,
            amplitudes=ampls,
            phases_deg=phases,
            comment=comment_str,
        )

    elif store_id == 3:
        # Recursive IIR filter: dt, norm, n_a, a_0..a_{n_a-1}, n_b, b_0..b_{n_b-1}
        dt = float(tokens[t_off])
        norm = float(tokens[t_off + 1])
        n_a = int(tokens[t_off + 2])
        idx = t_off + 3
        num_a = [float(tokens[idx + i]) for i in range(n_a)]
        idx += n_a
        n_b = int(tokens[idx]) if idx < len(tokens) else 0
        idx += 1
        den_b = [float(tokens[idx + i]) for i in range(n_b)]
        return FilterStage(
            store_id=3,
            norm=norm,
            dt=dt,
            num_a=num_a,
            den_b=den_b,
            comment=comment_str,
        )

    raise ValueError(f"Unsupported filter store_id: {store_id}")


def evaluate_stage_freq(stage: FilterStage, omega: np.ndarray) -> np.ndarray:
    """Evaluate complex frequency response H(i*omega) for angular frequencies omega (rad/s)."""
    if stage.store_id == 1:
        # H(i*w) = norm * prod(-z_re + i*(w - z_im)) / prod(-p_re + i*(w - p_im))
        jw = 1j * omega
        num = np.full(omega.shape, complex(stage.norm, 0.0), dtype=np.complex128)
        for z in stage.zeros:
            num = num * (jw - z)
        den = np.ones(omega.shape, dtype=np.complex128)
        for p in stage.poles:
            den = den * (jw - p)
        out = np.zeros(omega.shape, dtype=np.complex128)
        mask = np.abs(den) > 1e-30
        out[mask] = num[mask] / den[mask]
        return out

    elif stage.store_id == 2:
        freqs = omega / (2.0 * math.pi)
        tab_f = np.asarray(stage.freqs_hz, dtype=np.float64)
        tab_a = np.asarray(stage.amplitudes, dtype=np.float64)
        tab_p = np.radians(np.asarray(stage.phases_deg, dtype=np.float64))
        amp_interp = np.interp(freqs, tab_f, tab_a, left=0.0, right=0.0)
        ph_interp = np.interp(freqs, tab_f, tab_p, left=0.0, right=0.0)
        return amp_interp * np.exp(1j * ph_interp)

    elif stage.store_id == 3:
        # Evaluate digital transfer function on unit circle z^{-1} = exp(-i*omega*dt)
        z_inv = np.exp(-1j * omega * stage.dt)
        num = np.zeros(omega.shape, dtype=np.complex128)
        for i, a_i in enumerate(stage.num_a):
            num += a_i * (z_inv ** i)
        den = np.zeros(omega.shape, dtype=np.complex128)
        for j, b_j in enumerate(stage.den_b):
            den += b_j * (z_inv ** j)
        out = np.zeros(omega.shape, dtype=np.complex128)
        mask = np.abs(den) > 1e-30
        out[mask] = stage.norm * num[mask] / den[mask]
        return out

    return np.ones(omega.shape, dtype=np.complex128)


def apply_fft_filter(
    data: np.ndarray,
    dt: float,
    stages: List[FilterStage],
    taper_frac: float = 0.05,
    demean_first: bool = True,
) -> np.ndarray:
    """Apply one or more FilterStages in the frequency domain matching fftfiltr.c."""
    x = np.asarray(data, dtype=np.float64).copy()
    n = x.size
    if n == 0 or not stages:
        return x

    if demean_first:
        x = x - np.mean(x)

    # Cosine edge taper on input trace to suppress wrap-around ringing
    if taper_frac > 0.0 and n > 10:
        m = max(1, int(round(n * taper_frac)))
        win = 0.5 * (1.0 - np.cos(math.pi * np.arange(m) / m))
        x[:m] *= win
        x[-m:] *= win[::-1]

    # Next power of 2 >= 2 * n
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1

    spec = np.fft.rfft(x, n=nfft)
    freqs_hz = np.fft.rfftfreq(nfft, d=dt)
    omega = 2.0 * math.pi * freqs_hz

    total_resp = np.ones_like(spec, dtype=np.complex128)
    for stg in stages:
        total_resp *= evaluate_stage_freq(stg, omega)

    # Zero DC component if filter has a zero at s=0 or pole at s=0
    total_resp[0] = 0.0
    filtered_spec = spec * total_resp
    out = np.fft.irfft(filtered_spec, n=nfft)[:n]
    return out


def _burg_ar_predict_backward(x: np.ndarray, order: int = 20, n_back: int = 100) -> np.ndarray:
    """Predict n_back samples prior to x[0] using an AR model (matching use_lpc in recfiltr.c)."""
    if x.size <= order * 2:
        return np.zeros(n_back, dtype=np.float64)
    seg = x[: min(x.size, 500)]
    # Fit AR coefficients via Burg's method
    coeffs = burg_ar_coeffs(seg, order)
    # Reverse segment to predict backward from x[0]
    hist = list(seg[:order][::-1])
    pred: List[float] = []
    for _ in range(n_back):
        val = 0.0
        for k in range(order):
            val += coeffs[k] * hist[k]
        pred.append(val)
        hist = [val] + hist[:-1]
    return np.array(pred[::-1], dtype=np.float64)


def burg_ar_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    """Compute autoregressive coefficients a[1..order] using Burg's maximum entropy recursion."""
    n = x.size
    order = max(1, min(order, n // 2 - 1))
    ef = x.astype(np.float64).copy()
    eb = x.astype(np.float64).copy()
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    for m in range(1, order + 1):
        efp = ef[m:].copy()
        ebp = eb[m - 1 : -1].copy()
        num = -2.0 * np.dot(ebp, efp)
        den = np.dot(efp, efp) + np.dot(ebp, ebp)
        k = num / den if den > 1e-30 else 0.0
        # Update AR polynomial
        a_new = a.copy()
        for i in range(1, m):
            a_new[i] = a[i] + k * a[m - i]
        a_new[m] = k
        a = a_new
        # Update forward/backward errors
        ef[m:] = efp + k * ebp
        eb[m:] = ebp + k * efp

    return -a[1:]


def apply_recursive_filter(
    data: np.ndarray,
    dt: float,
    stages: List[FilterStage],
    use_lpc: bool = False,
    demean_first: bool = True,
) -> np.ndarray:
    """Apply cascade recursive IIR FilterStages in the time domain matching recfiltr.c."""
    x = np.asarray(data, dtype=np.float64).copy()
    n = x.size
    if n == 0 or not stages:
        return x

    if demean_first:
        x = x - np.mean(x)

    # Check if any stage is s-plane (store_id=1); if so, convert or apply via FFT
    if any(stg.store_id != 3 for stg in stages):
        return apply_fft_filter(x, dt, stages, demean_first=False)

    if use_lpc and n > 40:
        pre = _burg_ar_predict_backward(x, order=min(20, n // 4), n_back=min(200, n))
        work = np.concatenate([pre, x])
        n_pre = pre.size
    else:
        work = x
        n_pre = 0

    for stg in stages:
        a = np.asarray(stg.num_a, dtype=np.float64) * stg.norm
        b = np.asarray(stg.den_b, dtype=np.float64)
        if b.size > 0 and b[0] != 0.0 and b[0] != 1.0:
            a = a / b[0]
            b = b / b[0]

        na = a.size
        nb = b.size
        y = np.zeros_like(work)
        # Time-domain difference equation: y[n] = sum_{i=0}^{na-1} a[i]*x[n-i] - sum_{j=1}^{nb-1} b[j]*y[n-j]
        for idx in range(work.size):
            acc = 0.0
            for i in range(min(na, idx + 1)):
                acc += a[i] * work[idx - i]
            for j in range(1, min(nb, idx + 1)):
                acc -= b[j] * y[idx - j]
            if abs(acc) < 1e-30:
                acc = 0.0
            y[idx] = acc
        work = y

    return work[n_pre:]


def _butterworth_splane(
    ftype: str,
    order: int,
    w1: float,
    w2: float,
) -> Tuple[List[complex], List[complex], float]:
    """Construct continuous s-plane (zeros, poles, norm) for Butterworth LP/HP/BP/BS."""
    proto_poles = [
        cmath_exp_j(math.pi * (0.5 + (2 * k + 1) / (2.0 * order)))
        for k in range(order)
    ]
    s_zeros: List[complex] = []
    s_poles: List[complex] = []
    norm = 1.0

    if ftype in ("LP", "LOWPASS"):
        s_poles = [p * w1 for p in proto_poles]
        norm = float(w1 ** order)
    elif ftype in ("HP", "HIGHPASS"):
        s_zeros = [0j] * order
        s_poles = [w1 / p for p in proto_poles]
        norm = 1.0
    elif ftype in ("BP", "BANDPASS"):
        bw = w2 - w1
        w0_sq = w1 * w2
        s_zeros = [0j] * order
        for p in proto_poles:
            bh = 0.5 * bw * p
            disc = cmath_sqrt(bh * bh - w0_sq)
            s_poles.append(bh + disc)
            s_poles.append(bh - disc)
        norm = float(bw ** order)
    elif ftype in ("BS", "BANDSTOP", "BR"):
        bw = w2 - w1
        w0 = math.sqrt(w1 * w2)
        w0_sq = w1 * w2
        for _ in range(order):
            s_zeros.append(complex(0.0, w0))
            s_zeros.append(complex(0.0, -w0))
        for p in proto_poles:
            bh = 0.5 * bw / p
            disc = cmath_sqrt(bh * bh - w0_sq)
            s_poles.append(bh + disc)
            s_poles.append(bh - disc)
        norm = 1.0
    else:
        raise ValueError(f"Unknown Butterworth filter type: {ftype}")

    return s_zeros, s_poles, norm


def design_butterworth_stages(
    filter_type: str,
    order: int = 3,
    freq_lo: float = 0.5,
    freq_hi: float = 5.0,
    dt: Optional[float] = None,
    recursive: bool = True,
) -> List[FilterStage]:
    """Design Butterworth LP, HP, BP, or BS filter matching butfreq.c and butrec.c.

    Parameters
    ----------
    filter_type : 'LP', 'HP', 'BP', or 'BS'
    order : int (number of poles, typically 1..8)
    freq_lo : low corner frequency in Hz (or single corner for LP/HP)
    freq_hi : high corner frequency in Hz (for BP/BS)
    dt : sample interval in seconds (required if recursive=True)
    recursive : if True and dt is given, return bilinear-transformed recursive IIR stages (store_id=3);
                otherwise return continuous s-plane pole-zero stage (store_id=1).
    """
    ftype = filter_type.strip().upper()
    if ftype not in ("LP", "LOWPASS", "HP", "HIGHPASS", "BP", "BANDPASS", "BS", "BANDSTOP", "BR"):
        raise ValueError(f"Unknown Butterworth filter type: {filter_type}")
    order = max(1, min(12, int(order)))

    if not recursive or dt is None or dt <= 0:
        if ftype in ("LP", "LOWPASS", "HP", "HIGHPASS"):
            f1 = max(1e-5, freq_lo)
            f2 = f1
        else:
            f1 = max(1e-5, min(freq_lo, freq_hi))
            f2 = max(f1 + 1e-4, max(freq_lo, freq_hi))
        w1 = 2.0 * math.pi * f1
        w2 = 2.0 * math.pi * f2
        s_zeros, s_poles, norm = _butterworth_splane(ftype, order, w1, w2)
        return [
            FilterStage(
                store_id=1,
                norm=norm,
                zeros=s_zeros,
                poles=s_poles,
                comment=f"BUTTER_{ftype}_{order}_{freq_lo}_{freq_hi}",
            )
        ]

    # Bilinear transform s -> (2/dt) * (1 - z^{-1}) / (1 + z^{-1}) with frequency pre-warping
    return [_bilinear_stage_from_butter(ftype, order, freq_lo, freq_hi, dt)]


def _bilinear_stage_from_butter(
    ftype: str,
    order: int,
    freq_lo: float,
    freq_hi: float,
    dt: float,
) -> FilterStage:
    """Bilinear transform Butterworth design into digital IIR coefficients matching butrec.c."""
    nyq = 0.5 / dt
    f1 = max(1e-4, min(freq_lo, 0.99 * nyq))
    f2 = max(f1 + 1e-4, min(freq_hi, 0.99 * nyq))

    # Pre-warped analog frequencies for s = (2/dt)*(z-1)/(z+1)
    c = 2.0 / dt
    w1 = c * math.tan(math.pi * f1 * dt)
    w2 = c * math.tan(math.pi * f2 * dt)
    s_zeros, s_poles, s_gain = _butterworth_splane(ftype, order, w1, w2)

    # Map s-plane poles & zeros to z-plane via z = (c + s) / (c - s)
    n_p = len(s_poles)
    n_z = len(s_zeros)
    z_zeros = [(c + z) / (c - z) for z in s_zeros] + [-1.0 + 0j] * (n_p - n_z)
    z_poles = [(c + p) / (c - p) for p in s_poles]

    gain_num = s_gain * float(np.real(np.prod([c - z for z in s_zeros])))
    gain_den = float(np.real(np.prod([c - p for p in s_poles])))
    k_gain = gain_num / gain_den if abs(gain_den) > 1e-30 else 1.0

    poly_num = np.real(np.poly(z_zeros))
    poly_den = np.real(np.poly(z_poles))

    return FilterStage(
        store_id=3,
        norm=k_gain,
        dt=dt,
        num_a=list(poly_num),
        den_b=list(poly_den),
        comment=f"BUTTER_REC_{ftype}_{order}_{freq_lo}_{freq_hi}",
    )


def cmath_exp_j(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def cmath_sqrt(z: complex) -> complex:
    import cmath

    return cmath.sqrt(z)


def load_named_filter(
    filter_expr: str,
    stream: str = "grfo-bh-z",
    epoch: Optional[float] = None,
    dt: float = 0.05,
    config: Optional[SHMConfig] = None,
) -> List[FilterStage]:
    """Resolve an SHM filter expression (e.g. 'WWSSN_SP', '-GRSN+WWSSN_SP', 'STS2_M_WWSSN_SP',
    'BP 0.7 2.0 3', or a .FLF/.FLR filename) into a list of FilterStages.
    """
    cfg = config or get_config()
    expr = filter_expr.strip()
    if not expr or expr.upper() == "NONE":
        return []

    # Check if expr is a Butterworth shorthand like "BP 0.8 3.0 3" or "BUTTER BP 0.8 3.0 3"
    parts = expr.split()
    if parts[0].upper() in ("LP", "HP", "BP", "BS", "BUTTER"):
        idx = 1 if parts[0].upper() == "BUTTER" else 0
        ft = parts[idx].upper()
        f_lo = float(parts[idx + 1]) if len(parts) > idx + 1 else 1.0
        if ft in ("LP", "HP"):
            order = int(parts[idx + 2]) if len(parts) > idx + 2 else 3
            return design_butterworth_stages(ft, order=order, freq_lo=f_lo, dt=dt, recursive=True)
        f_hi = float(parts[idx + 2]) if len(parts) > idx + 2 else 5.0
        order = int(parts[idx + 3]) if len(parts) > idx + 3 else 3
        return design_butterworth_stages(ft, order=order, freq_lo=f_lo, freq_hi=f_hi, dt=dt, recursive=True)

    # Resolve exact file path / exact catalog filename before interpreting +/- chains
    exact_path = _find_exact_filter_file(expr, cfg)
    if exact_path is not None:
        return parse_filter_file(exact_path)

    # Check if expr is a chain of '+' or '-' filter tokens (e.g. "-STS2+WWSSN_SP" or "A-B")
    if "+" in expr or "-" in expr:
        stages: List[FilterStage] = []
        for tok in re.findall(r"[+-]?[^+-]+", expr):
            tok_s = tok.strip()
            inv = tok_s.startswith("-")
            name = tok_s.lstrip("+-").strip()
            if not name:
                continue
            sub_stages = _resolve_single_filter_file(name, stream, epoch, dt, cfg)
            for stg in sub_stages:
                stages.append(stg.inverted() if inv else stg)
        return stages

    return _resolve_single_filter_file(expr, stream, epoch, dt, cfg)


def _filter_search_dirs(cfg: SHMConfig) -> List[str]:
    return [
        cfg.filter_dir,
        os.path.join(cfg.filter_dir, "iaspei"),
        os.path.join(cfg.filter_dir, "classic"),
    ]


def _find_exact_filter_file(name: str, cfg: SHMConfig) -> Optional[str]:
    stripped = name.strip()
    if not stripped:
        return None
    if os.path.isfile(stripped):
        return stripped

    uname = stripped.upper()
    candidates: List[str] = []
    for ext in ("", ".FLF", ".FLR", ".FLT", ".flf", ".flr", ".flt"):
        candidates.append(stripped + ext)
        candidates.append(uname + ext)

    for sdir in _filter_search_dirs(cfg):
        if not os.path.isdir(sdir):
            continue
        dir_map = {fn.upper(): os.path.join(sdir, fn) for fn in os.listdir(sdir)}
        for cand in candidates:
            cu = cand.upper()
            if cu in dir_map and os.path.isfile(dir_map[cu]):
                return dir_map[cu]
    return None


def _resolve_single_filter_file(
    name: str,
    stream: str,
    epoch: Optional[float],
    dt: float,
    cfg: SHMConfig,
) -> List[FilterStage]:
    """Find and load a single filter name from SHM's filter directory or synthesize it."""
    exact_path = _find_exact_filter_file(name, cfg)
    if exact_path is not None:
        return parse_filter_file(exact_path)

    uname = name.strip().upper()
    srate_int = int(round(1.0 / dt)) if dt > 0 else 20
    tf_prefix = cfg.get_transfer_prefix(stream, epoch)

    # Rate-specific / instrument-specific simulation filter patterns in filter/
    candidates: List[str] = [
        f"G_{uname}.FLF",
        f"{srate_int}HZ_{uname}.FLR",
        f"{srate_int}HZ_{tf_prefix}_S+G_{uname}.FLR",
        f"{srate_int}HZ_{tf_prefix}_S+{uname}.FLR",
        f"{tf_prefix}_S+G_{uname}.FLF",
        f"{tf_prefix}_S+{uname}.FLF",
        f"{srate_int}HZ_{tf_prefix}_{uname}.FLR",
        f"{tf_prefix}_M_{uname}.FLF",
        f"{tf_prefix}_{uname}.FLF",
        f"STS2_M_{uname}.FLF",
        f"{srate_int}HZ_STS2_{uname}.FLR",
        f"GRSN_M_{uname}.FLF",
        f"{srate_int}HZ_GRSN_{uname}.FLR",
        f"{srate_int}HZ_GRSN_S+G_{uname}.FLR",
        f"GRF_M_{uname}.FLF",
        f"{srate_int}HZ_GRF_{uname}.FLR",
        f"{srate_int}HZ_GRF_S+G_{uname}.FLR",
    ]

    for sdir in _filter_search_dirs(cfg):
        if not os.path.isdir(sdir):
            continue
        dir_map = {fn.upper(): os.path.join(sdir, fn) for fn in os.listdir(sdir)}
        for cand in candidates:
            cu = cand.upper()
            if cu in dir_map and os.path.isfile(dir_map[cu]):
                return parse_filter_file(dir_map[cu])

    # Fallback synthetic pole-zero definitions for standard seismometers
    return [_synthetic_simulation_stage(uname)]


def _synthetic_simulation_stage(uname: str) -> FilterStage:
    """Synthetic s-plane pole-zero simulation response when no external file matches."""
    if "WWSSN_SP" in uname:
        # Short-period WWSSN (~1 Hz peak) with velocity-to-displacement integration zero/pole
        w0 = 2.0 * math.pi * 1.0
        return FilterStage(
            store_id=1,
            norm=w0 * w0,
            zeros=[0j, 0j],
            poles=[
                complex(-0.7 * w0, 0.714 * w0),
                complex(-0.7 * w0, -0.714 * w0),
                complex(-w0, 0.0),
            ],
            comment="SYNTH_WWSSN_SP",
        )
    if "WWSSN_LP" in uname or "SRO_LP" in uname:
        w0 = 2.0 * math.pi / 15.0
        return FilterStage(
            store_id=1,
            norm=w0 * w0,
            zeros=[0j, 0j],
            poles=[
                complex(-0.7 * w0, 0.714 * w0),
                complex(-0.7 * w0, -0.714 * w0),
                complex(-w0, 0.0),
            ],
            comment=f"SYNTH_{uname}",
        )
    if "WOODAND" in uname:
        # Wood-Anderson torsion seismometer: T0=0.8s, h=0.8 (IASPEI standard gain=1 or 2080)
        w0 = 2.0 * math.pi / 0.8
        h = 0.8
        wd = w0 * math.sqrt(1.0 - h * h)
        return FilterStage(
            store_id=1,
            norm=1.0,
            zeros=[0j],
            poles=[complex(-h * w0, wd), complex(-h * w0, -wd)],
            comment="SYNTH_WOODAND",
        )
    if "KIRNOS" in uname or "DISPLACE" in uname:
        # Broadband displacement restitution (high-passed integration at 0.02 Hz)
        w0 = 2.0 * math.pi * 0.02
        return FilterStage(
            store_id=1,
            norm=1.0,
            zeros=[0j],
            poles=[complex(-0.707 * w0, 0.707 * w0), complex(-0.707 * w0, -0.707 * w0)],
            comment=f"SYNTH_{uname}",
        )
    # Default bandpass 0.7 - 3.0 Hz
    return design_butterworth_stages("BP", order=3, freq_lo=0.7, freq_hi=3.0, recursive=False)[0]


def filter_trace(
    tr: Trace,
    filter_expr: str,
    mode: str = "AUTO",
    use_lpc: bool = False,
    taper_frac: float = 0.05,
    config: Optional[SHMConfig] = None,
) -> Trace:
    """Return a filtered copy of tr using filter_expr ('WWSSN_SP', 'BP 0.7 2.0 3', etc.)."""
    stages = load_named_filter(
        filter_expr,
        stream=tr.stream_name,
        epoch=tr.start_epoch,
        dt=tr.delta,
        config=config,
    )
    out_tr = tr.copy()
    if not stages:
        out_tr.filter_name = "NONE"
        return out_tr

    use_rec = mode.upper().startswith("R") or (
        mode.upper() == "AUTO" and all(s.store_id == 3 for s in stages)
    )
    if use_rec:
        out_tr.data = apply_recursive_filter(tr.data, tr.delta, stages, use_lpc=use_lpc)
    else:
        out_tr.data = apply_fft_filter(tr.data, tr.delta, stages, taper_frac=taper_frac)
    out_tr.filter_name = filter_expr.strip()
    return out_tr


def restitute_ground_displacement_nm(
    raw_ampl: float,
    period_sec: float,
    filter_name: str = "NONE",
) -> float:
    """Restitute true ground displacement amplitude in nm from measured trace amplitude (in nm/s or nm)
    and dominant period_sec, matching cu_ampl_restitution in cblib.c.
    """
    if period_sec <= 0.0:
        return abs(raw_ampl)

    fn = (filter_name or "NONE").upper()
    # If a displacement simulation filter (WWSSN_SP, WWSSN_LP, WOODAND, KIRNOS, DISPLACE) is already applied,
    # its output on velocity-calibrated traces is already in nm (up to the simulation filter transfer gain at period_sec).
    if any(k in fn for k in ("DISPLACE", "KIRNOS", "WOODAND", "WWSSN", "SRO", "LRSM")):
        return abs(raw_ampl)

    # Otherwise trace is in velocity (nm/s): convert to displacement (nm) via A_disp = A_vel * T / (2*pi)
    return abs(raw_ampl) * period_sec / (2.0 * math.pi)
