"""Mini-SEED reader/writer with Steim-1 and Steim-2 compression/decompression."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
import logging
import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import format_sh_time, julian_to_month_day, month_day_to_julian, parse_sh_time

logger = logging.getLogger(__name__)


@dataclass
class SeedRecordHeader:
    """Parsed 48-byte Mini-SEED fixed data header + Blockette 1000/1001."""

    seqno: str
    indicator: str
    station: str
    locid: str
    channel: str
    network: str
    start_epoch: float
    no_of_samples: int
    dt: float
    activity: int
    ioflags: int
    quality: int
    no_of_blockettes: int
    timecorr: int
    databegin: int
    first_blockette: int
    encoding_format: int = 10  # 10=Steim1, 11=Steim2
    word_order: int = 1  # 1=big-endian, 0=little-endian
    reclth: int = 4096
    big_endian_hdr: bool = True


def _compute_dt(fact: int, mult: int) -> float:
    """Compute sample interval dt in seconds from SEED smprate_fact and smprate_mult."""
    if fact == 0 or mult == 0:
        return 0.05
    if fact > 0 and mult > 0:
        return 1.0 / (float(fact) * float(mult))
    if fact > 0 and mult < 0:
        return -float(mult) / float(fact)
    if fact < 0 and mult > 0:
        return -float(fact) / float(mult)
    return 1.0 / (float(fact) * float(mult))


def parse_seed_header(buf: bytes) -> SeedRecordHeader:
    """Parse 48-byte fixed Mini-SEED header and Blockette 1000/1001."""
    if len(buf) < 48:
        raise ValueError("Buffer shorter than 48-byte SEED header")

    seqno = buf[0:6].decode("ascii", errors="ignore").replace("\x00", " ").strip()
    indicator = buf[6:7].decode("ascii", errors="ignore").replace("\x00", " ")
    reserved_a = buf[7]
    station = buf[8:13].decode("ascii", errors="ignore").replace("\x00", " ").strip().upper()
    locid = buf[13:15].decode("ascii", errors="ignore").replace("\x00", " ").strip()
    channel = buf[15:18].decode("ascii", errors="ignore").replace("\x00", " ").strip().upper()
    network = buf[18:20].decode("ascii", errors="ignore").replace("\x00", " ").strip().upper()

    # Check endianness using year at bytes 20..21
    year_be = struct.unpack(">H", buf[20:22])[0]
    big_endian = 1950 <= year_be <= 2050
    endian = ">" if big_endian else "<"

    year, jday, hour, minute, sec, _align, frac_0001 = struct.unpack(f"{endian}HHBBBBH", buf[20:30])
    nsmp, fact, mult, activity, ioflags, quality, nblock, timecorr, databegin, first_blk = struct.unpack(
        f"{endian}HhhBBBBiHH", buf[30:48]
    )

    dt = _compute_dt(fact, mult)
    month, day = julian_to_month_day(max(1970, min(2099, year)), max(1, min(366, jday)))
    base_dt = datetime.datetime(year, month, day, min(23, hour), min(59, minute), 0, tzinfo=datetime.timezone.utc)
    start_epoch = base_dt.timestamp() + sec + frac_0001 * 0.0001

    # Add timecorr if bit 1 of activity flags is NOT set (time correction not yet applied)
    if timecorr != 0 and not (activity & 0x02):
        start_epoch += timecorr * 0.0001

    reclth = 4096
    if 8 <= reserved_a < 24:
        reclth = 1 << reserved_a
    encoding_format = 10
    word_order = 1 if big_endian else 0

    # Walk blockettes
    blk_off = first_blk
    for _ in range(nblock):
        if blk_off <= 0 or blk_off + 4 > len(buf):
            break
        btype, next_blk = struct.unpack(f"{endian}HH", buf[blk_off : blk_off + 4])
        if btype == 1000 and blk_off + 8 <= len(buf):
            fmt, worder, recexp, _res = struct.unpack("BBBB", buf[blk_off + 4 : blk_off + 8])
            encoding_format = fmt
            word_order = worder
            if 8 <= recexp <= 24:
                reclth = 1 << recexp
        elif btype == 1001 and blk_off + 8 <= len(buf):
            _tqual, usec_corr, _res, _fcount = struct.unpack("BbBB", buf[blk_off + 4 : blk_off + 8])
            start_epoch += usec_corr * 1e-6
        blk_off = next_blk

    return SeedRecordHeader(
        seqno=seqno,
        indicator=indicator,
        station=station,
        locid=locid,
        channel=channel,
        network=network,
        start_epoch=start_epoch,
        no_of_samples=nsmp,
        dt=dt,
        activity=activity,
        ioflags=ioflags,
        quality=quality,
        no_of_blockettes=nblock,
        timecorr=timecorr,
        databegin=databegin if databegin >= 48 else 64,
        first_blockette=first_blk,
        encoding_format=encoding_format,
        word_order=word_order,
        reclth=reclth,
        big_endian_hdr=big_endian,
    )


def _sign_extend(val: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (val & (sign_bit - 1)) - (val & sign_bit)


def decompress_steim(
    frame_bytes: bytes,
    no_of_samples: int,
    steim_level: int = 1,
    big_endian: bool = True,
) -> np.ndarray:
    """Decompress Steim-1 (level=1) or Steim-2 (level=2) 64-byte frames matching steimalg.c."""
    if no_of_samples <= 0 or not frame_bytes:
        return np.zeros(0, dtype=np.int32)

    nframes = len(frame_bytes) // 64
    if nframes == 0:
        return np.zeros(0, dtype=np.int32)

    endian = ">" if big_endian else "<"
    diffs: List[int] = []
    x0 = 0

    for f_idx in range(nframes):
        if len(diffs) >= no_of_samples:
            break
        f_off = f_idx * 64
        words = struct.unpack(f"{endian}16I", frame_bytes[f_off : f_off + 64])
        w0 = words[0]
        if f_idx == 0:
            x0 = _sign_extend(words[1], 32)
            start_w = 3
        else:
            start_w = 1

        for w_idx in range(start_w, 16):
            if len(diffs) >= no_of_samples:
                break
            nib = (w0 >> (30 - 2 * w_idx)) & 0x03
            w = words[w_idx]

            if nib == 0:
                continue
            elif nib == 1:
                # Four 8-bit signed differences
                diffs.append(_sign_extend((w >> 24) & 0xFF, 8))
                diffs.append(_sign_extend((w >> 16) & 0xFF, 8))
                diffs.append(_sign_extend((w >> 8) & 0xFF, 8))
                diffs.append(_sign_extend(w & 0xFF, 8))
            elif nib == 2:
                if steim_level == 1:
                    # Two 16-bit signed differences
                    diffs.append(_sign_extend((w >> 16) & 0xFFFF, 16))
                    diffs.append(_sign_extend(w & 0xFFFF, 16))
                else:
                    # Steim-2 nibble 2: inspect top 2 bits dnib
                    dnib = (w >> 30) & 0x03
                    if dnib == 1:
                        diffs.append(_sign_extend(w & 0x3FFFFFFF, 30))
                    elif dnib == 2:
                        diffs.append(_sign_extend((w >> 15) & 0x7FFF, 15))
                        diffs.append(_sign_extend(w & 0x7FFF, 15))
                    elif dnib == 3:
                        diffs.append(_sign_extend((w >> 20) & 0x3FF, 10))
                        diffs.append(_sign_extend((w >> 10) & 0x3FF, 10))
                        diffs.append(_sign_extend(w & 0x3FF, 10))
            elif nib == 3:
                if steim_level == 1:
                    # One 32-bit signed difference
                    diffs.append(_sign_extend(w, 32))
                else:
                    # Steim-2 nibble 3: inspect top 2 bits dnib
                    dnib = (w >> 30) & 0x03
                    if dnib == 0:
                        for shift in (24, 18, 12, 6, 0):
                            diffs.append(_sign_extend((w >> shift) & 0x3F, 6))
                    elif dnib == 1:
                        for shift in (25, 20, 15, 10, 5, 0):
                            diffs.append(_sign_extend((w >> shift) & 0x1F, 5))
                    elif dnib == 2:
                        for shift in (24, 20, 16, 12, 8, 4, 0):
                            diffs.append(_sign_extend((w >> shift) & 0x0F, 4))

    diffs = diffs[:no_of_samples]
    if not diffs:
        return np.zeros(0, dtype=np.int32)

    arr = np.array(diffs, dtype=np.int64)
    arr[0] = x0
    out = np.cumsum(arr).astype(np.int32)
    return out


def compress_steim1_record(
    samples: np.ndarray,
    station: str,
    channel: str,
    network: str = "GR",
    locid: str = "",
    start_epoch: float = 0.0,
    dt: float = 0.05,
    seqno: int = 1,
    prev_sample: int = 0,
    reclth: int = 4096,
) -> Tuple[bytes, int]:
    """Pack integer samples into a single Steim-1 Mini-SEED record of size reclth.

    Returns (record_bytes, num_samples_packed).
    """
    if not isinstance(reclth, int) or isinstance(reclth, bool) or reclth < 128 or (reclth & (reclth - 1)) != 0:
        raise ValueError(f"Mini-SEED record length must be a power of 2 >= 128, got {reclth!r}")

    nframes = (reclth - 64) // 64
    smp = np.asarray(samples, dtype=np.int64)
    total_avail = int(smp.size)
    if total_avail == 0:
        return b"\x00" * reclth, 0

    # Compute differences in int64: d[0] = smp[0] - prev_sample, d[i] = smp[i] - smp[i-1]
    diffs = np.empty(total_avail, dtype=np.int64)
    diffs[0] = int(smp[0]) - int(prev_sample)
    if total_avail > 1:
        diffs[1:] = smp[1:] - smp[:-1]

    frames_out: List[bytes] = []
    idx = 0
    for f_idx in range(nframes):
        words = [0] * 16
        w0 = 0
        start_w = 3 if f_idx == 0 else 1
        for w_idx in range(start_w, 16):
            rem = total_avail - idx
            if rem <= 0:
                break
            # Check if next 4 fit in int8
            if rem >= 4 and all(-128 <= int(diffs[idx + k]) <= 127 for k in range(4)):
                b0 = int(diffs[idx]) & 0xFF
                b1 = int(diffs[idx + 1]) & 0xFF
                b2 = int(diffs[idx + 2]) & 0xFF
                b3 = int(diffs[idx + 3]) & 0xFF
                words[w_idx] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
                w0 |= 1 << (30 - 2 * w_idx)
                idx += 4
            elif rem >= 2 and all(-32768 <= int(diffs[idx + k]) <= 32767 for k in range(2)):
                h0 = int(diffs[idx]) & 0xFFFF
                h1 = int(diffs[idx + 1]) & 0xFFFF
                words[w_idx] = (h0 << 16) | h1
                w0 |= 2 << (30 - 2 * w_idx)
                idx += 2
            else:
                words[w_idx] = int(diffs[idx]) & 0xFFFFFFFF
                w0 |= 3 << (30 - 2 * w_idx)
                idx += 1
        words[0] = w0
        frames_out.append(struct.pack(">16I", *words))
        if idx >= total_avail:
            break

    # Pad remaining frames with zeros
    while len(frames_out) < nframes:
        frames_out.append(b"\x00" * 64)

    # Set x0 and xn in frame 0 words 1 and 2
    nsmp_packed = idx
    x0_val = int(smp[0]) & 0xFFFFFFFF
    xn_val = int(smp[nsmp_packed - 1]) & 0xFFFFFFFF
    f0_words = list(struct.unpack(">16I", frames_out[0]))
    f0_words[1] = x0_val
    f0_words[2] = xn_val
    frames_out[0] = struct.pack(">16I", *f0_words)

    # Build 48-byte fixed header + 8-byte Blockette 1000 + 8-byte padding = 64 bytes
    rounded_t = round(float(start_epoch) * 10000.0) / 10000.0
    dt_obj = datetime.datetime.fromtimestamp(rounded_t, tz=datetime.timezone.utc)
    jday = month_day_to_julian(dt_obj.year, dt_obj.month, dt_obj.day)
    frac_0001 = int(round((rounded_t - math.floor(rounded_t)) * 10000.0)) % 10000

    if dt > 0:
        rate = 1.0 / dt
        if abs(rate - round(rate)) < 1e-3 and rate >= 1.0:
            fact = int(round(rate))
            mult = 1
        else:
            fact = int(round(rate * 100.0))
            mult = -100
    else:
        fact, mult = 20, 1

    def _ascii_field(val: str, width: int) -> bytes:
        cleaned = (val or "").encode("ascii", errors="replace").decode("ascii")
        return f"{cleaned[:width]:<{width}s}".encode("ascii")

    recexp = int(round(math.log2(reclth)))
    seq_val = abs(int(seqno)) % 1000000
    seq_str = f"{seq_val:06d}".encode("ascii")
    stat_b = _ascii_field(station, 5)
    loc_b = _ascii_field(locid, 2)
    chan_b = _ascii_field(channel, 3)
    net_b = _ascii_field(network, 2)

    hdr = struct.pack(
        ">6scB5s2s3s2sHHBBBBHHhhBBBBiHH",
        seq_str,
        b"D",
        recexp,
        stat_b,
        loc_b,
        chan_b,
        net_b,
        dt_obj.year,
        jday,
        dt_obj.hour,
        dt_obj.minute,
        dt_obj.second,
        0,
        frac_0001,
        nsmp_packed,
        fact,
        mult,
        0,
        0,
        0,
        1,
        0,
        64,
        48,
    )
    blk1000 = struct.pack(">HHBBBB", 1000, 0, 10, 1, recexp, 0) + (b"\x00" * 8)
    return hdr + blk1000 + b"".join(frames_out), nsmp_packed


def _decode_record_samples(frame_bytes: bytes, hdr: SeedRecordHeader) -> np.ndarray:
    """Decode waveform samples from a Mini-SEED data payload for common SEED encodings."""
    nsmp = hdr.no_of_samples
    if nsmp <= 0 or not frame_bytes:
        return np.zeros(0, dtype=np.float64)

    big_end = (hdr.word_order != 0) if hdr.no_of_blockettes > 0 else hdr.big_endian_hdr
    endian = ">" if big_end else "<"
    fmt = hdr.encoding_format

    if fmt in (10, 11):
        steim_lvl = 2 if fmt == 11 else 1
        return decompress_steim(
            frame_bytes,
            nsmp,
            steim_level=steim_lvl,
            big_endian=big_end,
        ).astype(np.float64)

    dtype_map = {1: f"{endian}i2", 3: f"{endian}i4", 4: f"{endian}f4", 5: f"{endian}f8"}
    dt_str = dtype_map.get(fmt)
    if dt_str is None:
        # Fallback to Steim-1 for legacy records lacking Blockette 1000
        return decompress_steim(
            frame_bytes,
            nsmp,
            steim_level=1,
            big_endian=big_end,
        ).astype(np.float64)

    item_sz = np.dtype(dt_str).itemsize
    n_avail = min(nsmp, len(frame_bytes) // item_sz)
    if n_avail <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.frombuffer(frame_bytes, dtype=dt_str, count=n_avail).astype(np.float64)


def read_mseed_bytes(
    raw: bytes,
    t_start: Optional[float] = None,
    duration: Optional[float] = None,
    calib_override: Optional[float] = None,
    config: Optional[SHMConfig] = None,
    source_path: str = "",
) -> List[Trace]:
    """Decode raw Mini-SEED bytes into continuous/merged Traces in nm/s."""
    cfg = config or get_config()
    if len(raw) < 64:
        return []

    traces_by_key: Dict[Tuple[str, str, str, str], List[Tuple[float, float, np.ndarray]]] = {}
    offset = 0
    total_len = len(raw)

    while offset + 64 <= total_len:
        try:
            hdr = parse_seed_header(raw[offset : offset + min(512, total_len - offset)])
        except Exception as exc:
            logger.warning(
                "Stopping Mini-SEED parse at byte offset %d (%s): %s",
                offset,
                source_path or "<bytes>",
                exc,
            )
            break

        reclth = hdr.reclth if hdr.reclth >= 128 else 4096
        if offset + reclth > total_len:
            reclth = total_len - offset

        rec_end_epoch = hdr.start_epoch + max(0, hdr.no_of_samples - 1) * hdr.dt
        # Filter by time window if requested
        if t_start is not None and duration is not None:
            win_end = t_start + duration
            if rec_end_epoch < t_start or hdr.start_epoch > win_end:
                offset += reclth
                continue

        frame_bytes = raw[offset + hdr.databegin : offset + reclth]
        samples = _decode_record_samples(frame_bytes, hdr)

        if samples.size > 0:
            key = (hdr.station, hdr.channel, hdr.network, hdr.locid)
            traces_by_key.setdefault(key, []).append((hdr.start_epoch, hdr.dt, samples))

        offset += reclth

    out_traces: List[Trace] = []
    for (station, channel, network, locid), segments in traces_by_key.items():
        segments.sort(key=lambda x: x[0])

        groups: List[Tuple[float, float, List[np.ndarray]]] = []
        dt = segments[0][1]
        merged_start = segments[0][0]
        chunks: List[np.ndarray] = [segments[0][2]]
        cur_end = merged_start + segments[0][2].size * dt

        for seg_start, seg_dt, seg_data in segments[1:]:
            rate_changed = (
                dt <= 0
                or seg_dt <= 0
                or abs(seg_dt - dt) > max(1e-6, 0.01 * dt)
            )
            gap = seg_start - cur_end
            if rate_changed or gap >= 3600.0:
                groups.append((merged_start, dt, chunks))
                dt = seg_dt
                merged_start = seg_start
                chunks = [seg_data]
                cur_end = merged_start + seg_data.size * dt
                continue

            if abs(gap) <= 1.5 * dt:
                if gap < 0:
                    skip = int(round(-gap / dt))
                    if skip > 0:
                        if skip < seg_data.size:
                            chunks.append(seg_data[skip:])
                    else:
                        chunks.append(seg_data)
                else:
                    chunks.append(seg_data)
            elif gap > 1.5 * dt:
                # Pad small gap (< 3600s) with linear interpolation between boundary samples
                n_pad = int(round(gap / dt))
                if n_pad > 0:
                    pad = np.linspace(chunks[-1][-1], seg_data[0], n_pad + 2)[1:-1]
                    chunks.append(pad)
                chunks.append(seg_data)
            else:
                # Overlap: skip already-covered samples
                skip = int(round(-gap / dt))
                if skip < seg_data.size:
                    chunks.append(seg_data[skip:])
            cur_end = merged_start + sum(c.size for c in chunks) * dt

        groups.append((merged_start, dt, chunks))

        for grp_start, grp_dt, grp_chunks in groups:
            full_data = np.concatenate(grp_chunks) if len(grp_chunks) > 1 else grp_chunks[0]
            merged_start = grp_start
            dt = grp_dt

            # Trim to [t_start, t_start + duration] if requested
            if t_start is not None and duration is not None:
                i0 = max(0, int(round((t_start - merged_start) / dt)))
                i1 = min(full_data.size, max(0, int(round((t_start + duration - merged_start) / dt))))
                if i1 > i0:
                    merged_start = merged_start + i0 * dt
                    full_data = full_data[i0:i1]
                elif merged_start >= t_start + duration or (merged_start + full_data.size * dt) <= t_start:
                    continue

            chan1 = channel[0:1] if len(channel) >= 1 else "B"
            chan2 = channel[1:2] if len(channel) >= 2 else "H"
            comp = channel[2:3] if len(channel) >= 3 else "Z"
            stream_str = f"{station.lower()}-{chan1.lower()}{chan2.lower()}-{comp.lower()}"
            ch_meta = (
                cfg.get_channel_metadata(network, station, locid, channel, merged_start)
                if hasattr(cfg, "get_channel_metadata")
                else None
            )
            if calib_override is not None and calib_override > 0:
                calib = calib_override
            elif ch_meta is not None and hasattr(cfg, "_stationxml_calib_nm_per_count"):
                sxml_calib = cfg._stationxml_calib_nm_per_count(ch_meta)
                calib = sxml_calib if (sxml_calib is not None and sxml_calib > 0) else cfg.get_sensitivity(stream_str, merged_start)
            else:
                calib = cfg.get_sensitivity(stream_str, merged_start)
            scaled_data = full_data * calib

            stat_info = cfg.get_station(station)
            if ch_meta is not None and (ch_meta.latitude != 0.0 or ch_meta.longitude != 0.0):
                tr_lat = float(ch_meta.latitude)
                tr_lon = float(ch_meta.longitude)
            else:
                tr_lat = stat_info.lat
                tr_lon = stat_info.lon
            tr_inst = (ch_meta.sensor_description if (ch_meta is not None and ch_meta.sensor_description) else stat_info.instrument)
            extra_headers: Dict[str, Any] = {}
            if ch_meta is not None:
                extra_headers["azimuth"] = float(ch_meta.azimuth)
                extra_headers["dip"] = float(ch_meta.dip)
                extra_headers["station_lat"] = float(tr_lat)
                extra_headers["station_lon"] = float(tr_lon)
                extra_headers["station_elev"] = float(ch_meta.elevation)
                extra_headers["channel_depth_m"] = float(ch_meta.depth)

            tr = Trace(
                data=scaled_data,
                delta=dt,
                station=station,
                comp=comp,
                chan1=chan1,
                chan2=chan2,
                network=network,
                locid=locid,
                start_epoch=merged_start,
                calib=calib,
                lat=tr_lat,
                lon=tr_lon,
                instype=tr_inst,
                extra_headers=extra_headers,
                source_path=source_path,
            )
            out_traces.append(tr)

    return out_traces


def read_mseed_file(
    filepath: str,
    t_start: Optional[float] = None,
    duration: Optional[float] = None,
    calib_override: Optional[float] = None,
    config: Optional[SHMConfig] = None,
) -> List[Trace]:
    """Read a Mini-SEED file and return a list of continuous/merged Traces in nm/s."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Mini-SEED file not found: {filepath}")

    with open(filepath, "rb") as f:
        raw = f.read()

    return read_mseed_bytes(
        raw,
        t_start=t_start,
        duration=duration,
        calib_override=calib_override,
        config=config,
        source_path=filepath,
    )


def write_mseed_file(filepath: str, traces: List[Trace], reclth: int = 4096) -> None:
    """Write a list of Traces to a Steim-1 Mini-SEED file."""
    if not isinstance(reclth, int) or isinstance(reclth, bool) or reclth < 128 or (reclth & (reclth - 1)) != 0:
        raise ValueError(f"Mini-SEED record length must be a power of 2 >= 128, got {reclth!r}")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    seqno = 1
    with open(filepath, "wb") as f:
        for tr in traces:
            calib = tr.calib if tr.calib > 0 else 1.0
            raw_vals = np.asarray(tr.data, dtype=np.float64) / calib
            if raw_vals.size > 0 and not np.all(np.isfinite(raw_vals)):
                logger.warning(
                    "Trace %s.%s contains non-finite samples; replacing with 0 for Mini-SEED export",
                    tr.station,
                    tr.channel3,
                )
                raw_vals = np.nan_to_num(raw_vals, nan=0.0, posinf=0.0, neginf=0.0)
            counts = np.clip(np.round(raw_vals), -2147483648.0, 2147483647.0).astype(np.int32)
            idx = 0
            prev = 0
            while idx < counts.size:
                rec_start = tr.start_epoch + idx * tr.delta
                rec_bytes, n_packed = compress_steim1_record(
                    counts[idx:],
                    station=tr.station,
                    channel=tr.channel3,
                    network=tr.network or "GR",
                    locid=tr.locid or "",
                    start_epoch=rec_start,
                    dt=tr.delta,
                    seqno=seqno,
                    prev_sample=prev,
                    reclth=reclth,
                )
                if n_packed <= 0:
                    break
                f.write(rec_bytes)
                prev = int(counts[idx + n_packed - 1])
                idx += n_packed
                seqno = (seqno % 999999) + 1


# Backward-compatible re-exports from shm_app.io.sfd (placed after parse_seed_header/read_mseed_file).
from shm_app.io.sfd import (  # noqa: E402
    SFDRecord,
    generate_sfd_file,
    parse_sfd_file,
    read_from_sfd,
)
