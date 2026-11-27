"""SH Q-file (.QHD + .QBN) waveform reader and writer."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.ascii_io import read_ascii_file, write_ascii_file

QFILE_MAGIC = 43981


def _resolve_qfile_pair(filepath: str) -> Tuple[str, str]:
    """Return (qhd_path, qbn_path) from a base path or .QHD/.QBN filename."""
    base, ext = os.path.splitext(filepath)
    if ext.upper() in (".QHD", ".QBN"):
        root = base
    else:
        root = filepath
    # Check existing case variants
    for qhd_ext, qbn_ext in ((".QHD", ".QBN"), (".qhd", ".qbn")):
        if os.path.isfile(root + qhd_ext) or os.path.isfile(root + qbn_ext):
            return root + qhd_ext, root + qbn_ext
    return root + ".QHD", root + ".QBN"


def read_qfile(
    filepath: str,
    rec_indices: Optional[List[int]] = None,
    config: Optional[SHMConfig] = None,
) -> List[Trace]:
    """Read an SH Q-file pair (.QHD + .QBN) and return a list of Traces."""
    cfg = config or get_config()
    qhd_path, qbn_path = _resolve_qfile_pair(filepath)
    if not os.path.isfile(qhd_path):
        raise FileNotFoundError(f"Q-header file not found: {qhd_path}")
    if not os.path.isfile(qbn_path):
        raise FileNotFoundError(f"Q-binary file not found: {qbn_path}")

    with open(qhd_path, "r", encoding="latin-1", errors="ignore") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    if not lines:
        return []

    first_parts = lines[0].split()
    if len(first_parts) < 3 or int(first_parts[0]) != QFILE_MAGIC:
        raise ValueError(f"Invalid QHD magic number in {qhd_path}: {lines[0]!r}")

    cmtlines = int(first_parts[1])
    trclines = int(first_parts[2])
    if cmtlines < 1 or trclines <= 0:
        raise ValueError(
            f"Invalid QHD header line counts in {qhd_path}: cmtlines={cmtlines}, trclines={trclines}"
        )

    body_lines = lines[cmtlines:]
    n_records = len(body_lines) // trclines

    raw_floats = np.fromfile(qbn_path, dtype=np.float32)
    traces: List[Trace] = []

    wanted = set(rec_indices) if rec_indices else None

    for rec_idx in range(n_records):
        rec_num = rec_idx + 1
        if wanted is not None and rec_num not in wanted:
            continue

        rec_chunk = body_lines[rec_idx * trclines : (rec_idx + 1) * trclines]
        concat = ""
        for ln in rec_chunk:
            pipe_pos = ln.find("|")
            concat += ln[pipe_pos + 1 :] if pipe_pos >= 0 else ln

        entries: Dict[str, str] = {}
        for item in concat.split("~"):
            item_s = item.strip()
            if not item_s or ":" not in item_s:
                continue
            k, v = item_s.split(":", 1)
            entries[k.strip().upper()] = v.strip()

        length = int(entries.get("L001", entries.get("L000", "0")) or 0)
        offset = int(entries.get("L800", "0") or 0)
        if offset < 0 or length < 0 or offset + length > int(raw_floats.size):
            raise ValueError(
                f"QBN byte-range out of bounds in {qbn_path} for record {rec_num}: "
                f"offset={offset}, length={length}, available_samples={int(raw_floats.size)}"
            )
        delta = float(entries.get("R000", "0.05") or 0.05)
        if delta <= 0.0:
            delta = 0.05
        station = entries.get("S000", "STAT").strip().upper()
        comp = (entries.get("C000", "Z").strip() or "Z")[0].upper()
        chan1 = (entries.get("C001", "B").strip() or "B")[0].upper()
        chan2 = (entries.get("C002", "H").strip() or "H")[0].upper()
        t_str = entries.get("T000", "")
        try:
            start_epoch = parse_sh_time(t_str) if t_str else 0.0
        except ValueError:
            start_epoch = 0.0

        data_slice = raw_floats[offset : offset + length].astype(np.float64)
        stat_info = cfg.get_station(station)

        tr = Trace(
            data=data_slice,
            delta=delta,
            station=station,
            comp=comp,
            chan1=chan1,
            chan2=chan2,
            start_epoch=start_epoch,
            distance=float(entries.get("R011", "0") or 0.0),
            azimuth=float(entries.get("R012", "0") or 0.0),
            inci=float(entries.get("R013", "0") or 0.0),
            depth=float(entries.get("R014", "0") or 0.0),
            magnitude=float(entries.get("R015", "0") or 0.0),
            lat=float(entries.get("R016", str(stat_info.lat)) or stat_info.lat),
            lon=float(entries.get("R017", str(stat_info.lon)) or stat_info.lon),
            slowness=float(entries.get("R018", "0") or 0.0),
            torig=float(entries.get("R019", "0") or 0.0),
            weight=float(entries.get("R020", "1") or 1.0),
            norm=float(entries.get("R021", "1") or 1.0),
            calib=float(entries.get("R026", "1") or 1.0),
            comment=entries.get("S021", ""),
            opinfo=entries.get("S023", ""),
            filter_name=entries.get("S001", "NONE") or "NONE",
            source_path=qhd_path,
        )
        traces.append(tr)

    return traces


def _sanitize_qhd_text(val: str) -> str:
    """Strip CR/LF and tilde delimiters so string metadata never breaks .QHD line structure."""
    return val.replace("\r", " ").replace("\n", " ").replace("~", " ")


def write_qfile(filepath: str, traces: List[Trace], trclines: int = 4) -> Tuple[str, str]:
    """Write a list of Traces to an SH Q-file pair (.QHD + .QBN)."""
    qhd_path, qbn_path = _resolve_qfile_pair(filepath)
    os.makedirs(os.path.dirname(os.path.abspath(qhd_path)), exist_ok=True)

    cmtlines = 1
    trclines = max(2, trclines)
    sample_offset = 0
    float_chunks: List[np.ndarray] = []

    with open(qhd_path, "w", encoding="latin-1") as qhd:
        qhd.write(f"{QFILE_MAGIC} {cmtlines} {trclines}\n")
        for idx, tr in enumerate(traces, start=1):
            nsmp = tr.length
            f32 = tr.data.astype(np.float32)
            float_chunks.append(f32)

            kv_pairs = [
                f"L001:{nsmp:06d}~",
                f"L800:{sample_offset:08d}~",
                f"R000:{tr.delta:.6g}~",
                f"S000:{_sanitize_qhd_text(tr.station)}~",
                f"C000:{tr.comp}~",
                f"C001:{tr.chan1}~",
                f"C002:{tr.chan2}~",
                f"T000:{tr.start_time_str}~",
                f"R011:{tr.distance:.4f}~",
                f"R012:{tr.azimuth:.4f}~",
                f"R013:{tr.inci:.4f}~",
                f"R014:{tr.depth:.4f}~",
                f"R015:{tr.magnitude:.4f}~",
                f"R016:{tr.lat:.5f}~",
                f"R017:{tr.lon:.5f}~",
                f"R018:{tr.slowness:.4f}~",
                f"R019:{tr.torig:.4f}~",
                f"R020:{tr.weight:.4f}~",
                f"R021:{tr.norm:.4f}~",
                f"R026:{tr.calib:.6e}~",
            ]
            if tr.filter_name and tr.filter_name != "NONE":
                kv_pairs.append(f"S001:{_sanitize_qhd_text(tr.filter_name)}~")
            if tr.comment:
                kv_pairs.append(f"S021:{_sanitize_qhd_text(tr.comment)}~")
            if tr.opinfo:
                kv_pairs.append(f"S023:{_sanitize_qhd_text(tr.opinfo)}~")

            # Pack into trclines lines prefixed with 'NN|'
            per_line = max(1, (len(kv_pairs) + trclines - 1) // trclines)
            for l_i in range(trclines):
                chunk_str = " ".join(kv_pairs[l_i * per_line : (l_i + 1) * per_line])
                qhd.write(f"{idx:02d}|{chunk_str}\n")

            sample_offset += nsmp

    all_floats = np.concatenate(float_chunks) if float_chunks else np.zeros(0, dtype=np.float32)
    all_floats.tofile(qbn_path)
    return qhd_path, qbn_path
