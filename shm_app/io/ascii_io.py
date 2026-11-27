"""SH ASCII (READA/WRITEA) waveform reader and writer."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time


def read_ascii_file(filepath: str, config: Optional[SHMConfig] = None) -> List[Trace]:
    """Read an SH ASCII trace file (READA/WRITEA format or raw sample column)."""
    cfg = config or get_config()
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ASCII file not found: {filepath}")

    with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("!")]

    traces: List[Trace] = []
    idx = 0
    nlines = len(lines)

    while idx < nlines:
        headers: Dict[str, str] = {}
        while idx < nlines and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", lines[idx]):
            k, v = lines[idx].split(":", 1)
            headers[k.strip().upper()] = v.strip()
            idx += 1

        expected_len = int(headers.get("LENGTH", "0") or 0)
        samples: List[float] = []
        while idx < nlines:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", lines[idx]):
                break
            for tok in lines[idx].split():
                try:
                    samples.append(float(tok))
                except ValueError:
                    pass
            idx += 1
            if expected_len > 0 and len(samples) >= expected_len:
                break

        if not samples:
            break
        if expected_len > 0:
            samples = samples[:expected_len]

        station = headers.get("STATION", "STAT").upper()
        comp = (headers.get("COMP", "Z") or "Z")[0].upper()
        chan1 = (headers.get("CHAN1", "B") or "B")[0].upper()
        chan2 = (headers.get("CHAN2", "H") or "H")[0].upper()
        delta = float(headers.get("DELTA", "0.05") or 0.05)
        t_str = headers.get("START", "")
        try:
            start_epoch = parse_sh_time(t_str) if t_str else 0.0
        except ValueError:
            start_epoch = 0.0

        stat_info = cfg.get_station(station)
        tr = Trace(
            data=np.array(samples, dtype=np.float64),
            delta=delta,
            station=station,
            comp=comp,
            chan1=chan1,
            chan2=chan2,
            start_epoch=start_epoch,
            calib=float(headers.get("CALIB", "1.0") or 1.0),
            distance=float(headers.get("DISTANCE", "0.0") or 0.0),
            azimuth=float(headers.get("AZIMUTH", "0.0") or 0.0),
            depth=float(headers.get("DEPTH", "0.0") or 0.0),
            slowness=float(headers.get("SLOWNESS", "0.0") or 0.0),
            lat=stat_info.lat,
            lon=stat_info.lon,
            comment=headers.get("COMMENT", ""),
            source_path=filepath,
        )
        traces.append(tr)

    return traces


def write_ascii_file(filepath: str, traces: List[Trace], cols_per_line: int = 5) -> None:
    """Write a list of Traces to an SH ASCII file (WRITEA format)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="ascii") as out:
        for tr in traces:
            out.write(f"DELTA: {tr.delta:.6g}\n")
            out.write(f"LENGTH: {tr.length}\n")
            out.write(f"STATION: {tr.station}\n")
            out.write(f"COMP: {tr.comp}\n")
            out.write(f"CHAN1: {tr.chan1}\n")
            out.write(f"CHAN2: {tr.chan2}\n")
            out.write(f"START: {tr.start_time_str}\n")
            out.write(f"CALIB: {tr.calib:.6e}\n")
            if tr.distance != 0.0:
                out.write(f"DISTANCE: {tr.distance:.4f}\n")
            if tr.azimuth != 0.0:
                out.write(f"AZIMUTH: {tr.azimuth:.4f}\n")
            if tr.depth != 0.0:
                out.write(f"DEPTH: {tr.depth:.4f}\n")
            if tr.slowness != 0.0:
                out.write(f"SLOWNESS: {tr.slowness:.4f}\n")
            if tr.comment:
                out.write(f"COMMENT: {tr.comment}\n")

            for i in range(0, tr.length, cols_per_line):
                chunk = tr.data[i : i + cols_per_line]
                out.write(" ".join(f"{val:13.5e}" for val in chunk) + "\n")
            out.write("\n")
