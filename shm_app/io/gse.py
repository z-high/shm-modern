"""GSE1 (WID1/DAT1/CHK1) and GSE2 (WID2/STA2/DAT2/CHK2) waveform reader and writer."""

from __future__ import annotations

import datetime
import logging
import math
import os
import re
from typing import List, Optional, Tuple
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import format_gse2_time, julian_to_month_day, parse_sh_time

logger = logging.getLogger(__name__)

# CM6 character lookup table matching gsedata.c:
# '+' -> 0, '-' -> 1, '0'..'9' -> 2..11, 'A'..'Z' -> 12..37, 'a'..'z' -> 38..63
CM6_ENCODE_CHARS = "+-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CM6_DECODE_MAP = {ch: idx for idx, ch in enumerate(CM6_ENCODE_CHARS)}

CHAN2_TO_CHAN3 = {
    "BZ": "BHZ",
    "BN": "BHN",
    "BE": "BHE",
    "LZ": "LHZ",
    "LN": "LHN",
    "LE": "LHE",
    "SZ": "SHZ",
    "SN": "SHN",
    "SE": "SHE",
    "VZ": "VHZ",
    "VN": "VHN",
    "VE": "VHE",
    "HZ": "HHZ",
    "HN": "HHN",
    "HE": "HHE",
}


def gse_checksum(samples: np.ndarray) -> int:
    """Compute GSE1/GSE2 modulo 100,000,000 checksum matching gsedata.c."""
    modulo = 100000000
    total = 0
    for val in samples.astype(np.int64):
        s_val = int(val)
        if abs(s_val) >= modulo:
            s_val = s_val - (s_val // modulo) * modulo
        total += s_val
        if abs(total) >= modulo:
            total = total - (total // modulo) * modulo
    return abs(total)


def decompress_cm6(lines: List[str], expected_samples: int, n_integrations: int = 2) -> np.ndarray:
    """Decompress GSE CM6 6-bit character stream and apply n_integrations cumulative sums."""
    diffs: List[int] = []
    accum = 0
    neg = False
    in_number = False

    for line in lines:
        for ch in line:
            code = CM6_DECODE_MAP.get(ch, -1)
            if code < 0:
                continue
            if not in_number:
                neg = bool(code & 0x10)
                cont = bool(code & 0x20)
                accum = code & 0x0F
                if cont:
                    in_number = True
                else:
                    diffs.append(-accum if neg else accum)
                    if len(diffs) >= expected_samples:
                        break
            else:
                cont = bool(code & 0x20)
                accum = (accum << 5) | (code & 0x1F)
                if not cont:
                    diffs.append(-accum if neg else accum)
                    in_number = False
                    if len(diffs) >= expected_samples:
                        break
        if len(diffs) >= expected_samples:
            break

    arr = np.array(diffs[:expected_samples], dtype=np.int64)
    for _ in range(n_integrations):
        arr = np.cumsum(arr)
    return arr


def compress_cm6(samples: np.ndarray, n_differences: int = 2, line_width: int = 80) -> List[str]:
    """Compress integer samples into GSE CM6 lines after applying n_differences forward differences."""
    arr = np.asarray(samples, dtype=np.int64).copy()
    for _ in range(n_differences):
        if arr.size > 1:
            arr[1:] = arr[1:] - arr[:-1]

    chars: List[str] = []
    for val in arr:
        v = int(val)
        neg = v < 0
        mag = abs(v)

        # Split magnitude into low 5-bit chunks, with top chunk holding 4 bits + sign bit
        chunks_5bit: List[int] = []
        while mag > 0x0F:
            chunks_5bit.append(mag & 0x1F)
            mag >>= 5
        first_byte = (mag & 0x0F) | (0x10 if neg else 0x00)
        if chunks_5bit:
            first_byte |= 0x20
        chars.append(CM6_ENCODE_CHARS[first_byte])

        for idx in range(len(chunks_5bit) - 1, -1, -1):
            c_byte = chunks_5bit[idx]
            if idx > 0:
                c_byte |= 0x20
            chars.append(CM6_ENCODE_CHARS[c_byte])

    joined = "".join(chars)
    return [joined[i : i + line_width] for i in range(0, len(joined), line_width)]


def read_gse_file(filepath: str, config: Optional[SHMConfig] = None) -> List[Trace]:
    """Read a GSE1 (WID1) or GSE2 (WID2) waveform file and return a list of Traces."""
    cfg = config or get_config()
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"GSE file not found: {filepath}")

    with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
        raw_lines = [ln.rstrip("\r\n") for ln in f]

    traces: List[Trace] = []
    idx = 0
    nlines = len(raw_lines)

    while idx < nlines:
        line = raw_lines[idx]
        if line.startswith("WID1 "):
            tr, idx = _parse_wid1_block(raw_lines, idx, filepath, cfg)
            if tr is not None:
                traces.append(tr)
        elif line.startswith("WID2 "):
            tr, idx = _parse_wid2_block(raw_lines, idx, filepath, cfg)
            if tr is not None:
                traces.append(tr)
        else:
            idx += 1

    return traces


def _parse_wid1_block(
    lines: List[str],
    start_idx: int,
    filepath: str,
    cfg: SHMConfig,
) -> Tuple[Optional[Trace], int]:
    """Parse a WID1 + DAT1 + CHK1 block."""
    line1 = lines[start_idx]
    if start_idx + 1 >= len(lines):
        logger.warning("Truncated WID1 block at line %d in %s", start_idx + 1, filepath)
        return None, start_idx + 1
    line2 = lines[start_idx + 1]

    # Parse WID1 line 1
    parts = line1.split()
    if len(parts) < 8 or len(parts[1]) < 7:
        logger.warning("Malformed WID1 header at line %d in %s: %r", start_idx + 1, filepath, line1)
        return None, start_idx + 1
    try:
        yj = parts[1]
        year, jday = int(yj[:4]), int(yj[4:7])
        hour, minute, sec, msec = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        nsamps = max(0, int(parts[6]))
        station = parts[7].upper()
        chan_raw = parts[9].upper() if len(parts) > 9 else "BZ"
        samprat = float(parts[10]) if len(parts) > 10 else 20.0
    except (ValueError, IndexError) as exc:
        logger.warning("Invalid WID1 numeric field at line %d in %s: %s", start_idx + 1, filepath, exc)
        return None, start_idx + 1

    # Remaining tokens after samprat: [instype] datatype [diff_flag]
    tail = parts[11:]
    diff_flag = 0
    datatype = "CM6"
    instype = ""
    if tail:
        if tail[-1].lstrip("-").isdigit() and len(tail[-1]) <= 2:
            diff_flag = int(tail[-1])
            tail = tail[:-1]
        if tail:
            datatype = tail[-1].upper()
            tail = tail[:-1]
        if tail:
            instype = tail[0]

    # Parse WID1 line 2: <calib> [<unit>] <calper> <lat> <lon> <elev> ...
    floats_l2 = [
        float(x)
        for x in re.findall(
            r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?",
            line2.replace("D", "E").replace("d", "e"),
        )
    ]
    calib = floats_l2[0] if len(floats_l2) > 0 else 1.0
    calper = floats_l2[1] if len(floats_l2) > 1 else 1.0
    lat = floats_l2[2] if len(floats_l2) > 2 else 0.0
    lon = floats_l2[3] if len(floats_l2) > 3 else 0.0

    try:
        month, day = julian_to_month_day(year, max(1, min(366, jday)))
        dt_obj = datetime.datetime(year, month, day, max(0, min(23, hour)), max(0, min(59, minute)), 0, tzinfo=datetime.timezone.utc)
    except ValueError as exc:
        logger.warning("Invalid WID1 timestamp at line %d in %s: %s", start_idx + 1, filepath, exc)
        return None, start_idx + 1
    start_epoch = dt_obj.timestamp() + sec + msec / 1000.0
    dt = 1.0 / samprat if samprat > 0 else 0.05

    chan3 = CHAN2_TO_CHAN3.get(chan_raw, chan_raw if len(chan_raw) == 3 else f"BH{chan_raw[-1:] or 'Z'}")

    # Advance to DAT1
    idx = start_idx + 2
    while idx < len(lines) and not lines[idx].startswith("DAT1"):
        idx += 1
    if idx < len(lines) and lines[idx].startswith("DAT1"):
        idx += 1

    data_lines: List[str] = []
    chk_val: Optional[int] = None
    while idx < len(lines):
        ln = lines[idx]
        if ln.startswith("CHK1"):
            chk_parts = ln.split()
            if len(chk_parts) >= 2:
                try:
                    chk_val = int(chk_parts[1])
                except ValueError:
                    pass
            idx += 1
            break
        if ln.startswith("WID1 ") or ln.startswith("WID2 ") or ln.startswith("STOP"):
            break
        data_lines.append(ln)
        idx += 1

    raw_samps = _decode_gse_samples(data_lines, datatype, nsamps, n_integrations=diff_flag)
    if chk_val is not None and raw_samps.size > 0:
        calc_chk = gse_checksum(raw_samps)
        # Store checksum metadata in extra_headers
        chk_ok = calc_chk == abs(chk_val)
        if not chk_ok:
            logger.warning(
                "GSE1 checksum mismatch for %s.%s in %s: expected=%d, computed=%d",
                station,
                chan3,
                filepath,
                abs(chk_val),
                calc_chk,
            )
    else:
        chk_ok = True

    # Convert to velocity (nm/s) if calib & calper given
    if not (math.isfinite(calib) and calib > 0 and math.isfinite(calper) and calper > 0):
        logger.warning(
            "GSE1 %s.%s in %s has invalid calibration (calib=%r, calper=%r); falling back to vel_scale=1.0",
            station,
            chan3,
            filepath,
            calib,
            calper,
        )
        vel_scale = 1.0
    else:
        vel_scale = calib * 2.0 * math.pi / calper
    stat_info = cfg.get_station(station)
    tr = Trace(
        data=raw_samps.astype(np.float64) * vel_scale,
        delta=dt,
        station=station,
        comp=chan3[2:3],
        chan1=chan3[0:1],
        chan2=chan3[1:2],
        start_epoch=start_epoch,
        calib=vel_scale,
        calper=calper,
        instype=instype or stat_info.instrument,
        lat=lat if lat != 0.0 else stat_info.lat,
        lon=lon if lon != 0.0 else stat_info.lon,
        source_path=filepath,
        extra_headers={"gse_format": "GSE1", "gse_datatype": datatype, "checksum_ok": chk_ok},
    )
    return tr, idx


def _parse_wid2_block(
    lines: List[str],
    start_idx: int,
    filepath: str,
    cfg: SHMConfig,
) -> Tuple[Optional[Trace], int]:
    """Parse a WID2 (+ optional STA2) + DAT2 + CHK2 block."""
    line = lines[start_idx]
    parts = line.split()
    if len(parts) < 5:
        logger.warning("Malformed WID2 header at line %d in %s: %r", start_idx + 1, filepath, line)
        return None, start_idx + 1

    # WID2 YYYY/MM/DD HH:MM:SS.SSS STAT CHAN AUX DATATYPE SAMPS SAMPRAT CALIB CALPER INSTYPE HANG VANG
    date_str = f"{parts[1]} {parts[2]}"
    try:
        start_epoch = parse_sh_time(date_str)
    except ValueError as exc:
        logger.warning("Invalid WID2 timestamp at line %d in %s: %s", start_idx + 1, filepath, exc)
        return None, start_idx + 1

    # Check if AUXID is blank in fixed columns 35..39
    try:
        if len(line) >= 80 and line[44:47].strip().upper() in ("CM6", "INT", "FLT"):
            station = line[29:34].strip().upper() or parts[3].upper()
            chan_raw = line[35:38].strip().upper() or parts[4].upper()
            auxid = line[39:43].strip()
            datatype = line[44:47].strip().upper()
            nsamps = max(0, int(line[48:56].strip()))
            samprat = float(line[57:68].strip())
            rem_tokens = line[69:].split()
            calib = float(rem_tokens[0]) if len(rem_tokens) > 0 else 1.0
            calper = float(rem_tokens[1]) if len(rem_tokens) > 1 else 1.0
            instype = rem_tokens[2] if len(rem_tokens) > 2 else ""
        else:
            if len(parts) < 8:
                logger.warning("Truncated WID2 header at line %d in %s: %r", start_idx + 1, filepath, line)
                return None, start_idx + 1
            station = parts[3].upper()
            chan_raw = parts[4].upper()
            auxid = ""
            datatype = parts[5].upper()
            nsamps = max(0, int(parts[6]))
            samprat = float(parts[7])
            calib = float(parts[8]) if len(parts) > 8 else 1.0
            calper = float(parts[9]) if len(parts) > 9 else 1.0
            instype = parts[10] if len(parts) > 10 else ""
    except (ValueError, IndexError) as exc:
        logger.warning("Invalid WID2 field at line %d in %s: %s", start_idx + 1, filepath, exc)
        return None, start_idx + 1

    chan3 = CHAN2_TO_CHAN3.get(chan_raw, chan_raw if len(chan_raw) == 3 else f"BH{chan_raw[-1:] or 'Z'}")
    dt = 1.0 / samprat if samprat > 0 else 0.05

    lat, lon, elev = 0.0, 0.0, 0.0
    network = ""
    idx = start_idx + 1
    while idx < len(lines) and not lines[idx].startswith("DAT2"):
        if lines[idx].startswith("STA2"):
            sp = lines[idx].split()
            if len(sp) >= 4:
                network = sp[1] if len(sp) >= 5 else ""
                try:
                    lat = float(sp[-3])
                    lon = float(sp[-2])
                    elev = float(sp[-1])
                except ValueError:
                    pass
        idx += 1
    if idx < len(lines) and lines[idx].startswith("DAT2"):
        idx += 1

    data_lines: List[str] = []
    chk_val: Optional[int] = None
    while idx < len(lines):
        ln = lines[idx]
        if ln.startswith("CHK2"):
            sp = ln.split()
            if len(sp) >= 2:
                try:
                    chk_val = int(sp[1])
                except ValueError:
                    pass
            idx += 1
            break
        if ln.startswith("WID2 ") or ln.startswith("WID1 ") or ln.startswith("STOP"):
            break
        data_lines.append(ln)
        idx += 1

    raw_samps = _decode_gse_samples(data_lines, datatype, nsamps, n_integrations=2)
    if chk_val is not None and raw_samps.size > 0:
        calc_chk = gse_checksum(raw_samps)
        chk_ok = calc_chk == abs(chk_val)
        if not chk_ok:
            logger.warning(
                "GSE2 checksum mismatch for %s.%s in %s: expected=%d, computed=%d",
                station,
                chan3,
                filepath,
                abs(chk_val),
                calc_chk,
            )
    else:
        chk_ok = True

    if not (math.isfinite(calib) and calib > 0 and math.isfinite(calper) and calper > 0):
        logger.warning(
            "GSE2 %s.%s in %s has invalid calibration (calib=%r, calper=%r); falling back to vel_scale=1.0",
            station,
            chan3,
            filepath,
            calib,
            calper,
        )
        vel_scale = 1.0
    else:
        vel_scale = calib * 2.0 * math.pi / calper
    stat_info = cfg.get_station(station)
    tr = Trace(
        data=raw_samps.astype(np.float64) * vel_scale,
        delta=dt,
        station=station,
        comp=chan3[2:3],
        chan1=chan3[0:1],
        chan2=chan3[1:2],
        auxid=auxid,
        network=network,
        start_epoch=start_epoch,
        calib=vel_scale,
        calper=calper,
        instype=instype or stat_info.instrument,
        lat=lat if lat != 0.0 else stat_info.lat,
        lon=lon if lon != 0.0 else stat_info.lon,
        source_path=filepath,
        extra_headers={
            "gse_format": "GSE2",
            "gse_datatype": datatype,
            "checksum_ok": chk_ok,
            "elevation": elev,
        },
    )
    return tr, idx


def _decode_gse_samples(
    data_lines: List[str],
    datatype: str,
    expected_samples: int,
    n_integrations: int = 2,
) -> np.ndarray:
    dtype = datatype.strip().upper()
    if dtype == "CM6":
        return decompress_cm6(data_lines, expected_samples, n_integrations=n_integrations)
    # INT, INTV, FLT
    vals: List[float] = []
    for ln in data_lines:
        for tok in ln.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
            if len(vals) >= expected_samples:
                break
        if len(vals) >= expected_samples:
            break
    arr = np.array(vals[:expected_samples], dtype=np.float64)
    if dtype in ("INT", "INTV"):
        return np.round(arr).astype(np.int64)
    return arr


def write_gse2_file(filepath: str, traces: List[Trace]) -> None:
    """Write a list of Traces into a GSE2.0 CM6 file with WID2, STA2, DAT2, and CHK2."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="ascii") as out:
        for tr in traces:
            calper = tr.calper if tr.calper > 0 else 1.0
            vel_scale = tr.calib if tr.calib > 0 else 1.0
            disp_calib = vel_scale * calper / (2.0 * math.pi)
            int_samples = np.round(tr.data / vel_scale).astype(np.int64)
            t_str = format_gse2_time(tr.start_epoch)
            samprat = 1.0 / tr.delta if tr.delta > 0 else 20.0
            inst = (tr.instype or "STS-2")[:6]
            hang = -1.0 if tr.comp == "Z" else (0.0 if tr.comp == "N" else 90.0)
            vang = 0.0 if tr.comp == "Z" else 90.0
            wid2 = (
                f"WID2 {t_str} {tr.station:<5s} {tr.channel3:<3s} {tr.auxid:<4s} "
                f"CM6 {int_samples.size:8d} {samprat:11.6f} {disp_calib:14.8e} "
                f"{calper:7.3f} {inst:<6s} {hang:5.1f} {vang:4.1f}"
            )
            out.write(wid2 + "\n")
            net = (tr.network or "GR")[:9]
            sta2 = f"STA2 {net:<9s} {tr.lat:9.5f} {tr.lon:10.5f} ED50         0.000  0.000"
            out.write(sta2 + "\n")
            out.write("DAT2\n")
            for cm_line in compress_cm6(int_samples, n_differences=2):
                out.write(cm_line + "\n")
            chk = gse_checksum(int_samples)
            out.write(f"CHK2 {chk:8d}\n\n")
