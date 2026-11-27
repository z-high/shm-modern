"""SHM SFD index file (sfdlist.sfd) parser, generator, and windowed Mini-SEED reader."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
from typing import List, Optional, Tuple

from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.models import Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time


@dataclass
class SFDRecord:
    """Single line entry in an SFD index file (sfdlist.sfd)."""

    stream: str = ""
    filepath: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    t_start_str: str = ""
    t_end_str: str = ""
    recno: int = 1
    swap_hdr: int = 0
    reclth: int = 4096
    byteoff: int = 0
    calib: float = 1.0
    net_loc: str = ""


def parse_sfd_file(sfdfile: str, base_dir: Optional[str] = None) -> List[SFDRecord]:
    """Parse an SHM SFD index file (e.g. sfdlist.sfd)."""
    if not os.path.isfile(sfdfile):
        return []
    bdir = base_dir or os.path.dirname(os.path.abspath(sfdfile))
    records: List[SFDRecord] = []
    with open(sfdfile, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            rec = SFDRecord()
            for token in s.split():
                if len(token) > 2 and token[1] == ">":
                    tag = token[0]
                    val = token[2:]
                    if tag == "s":
                        rec.stream = val.lower()
                    elif tag == "n":
                        rec.filepath = val if os.path.isabs(val) else os.path.normpath(os.path.join(bdir, val))
                    elif tag == "b":
                        rec.t_start_str = val
                        try:
                            rec.t_start = parse_sh_time(val)
                        except ValueError:
                            rec.t_start = 0.0
                    elif tag == "e":
                        rec.t_end_str = val
                        try:
                            rec.t_end = parse_sh_time(val)
                        except ValueError:
                            rec.t_end = 0.0
                    elif tag == "r":
                        rec.recno = int(val)
                    elif tag == "h":
                        rec.swap_hdr = int(val)
                    elif tag == "l":
                        rec.reclth = int(val)
                    elif tag == "o":
                        rec.byteoff = int(val)
                    elif tag == "c":
                        rec.calib = float(val)
                    elif tag == "a":
                        rec.net_loc = val
            if rec.stream and rec.filepath:
                records.append(rec)
    return records


def generate_sfd_file(
    data_dir: str,
    output_sfd: Optional[str] = None,
    config: Optional[SHMConfig] = None,
) -> List[SFDRecord]:
    """Scan a directory for Mini-SEED files and generate an sfdlist.sfd index."""
    from shm_app.io.mseed import parse_seed_header

    cfg = config or get_config()
    records: List[Tuple[str, SFDRecord]] = []
    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if fname.endswith(".sfd") or fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "rb") as f:
                    first_chunk = f.read(512)
                    if len(first_chunk) < 64:
                        continue
                    hdr0 = parse_seed_header(first_chunk)
                    reclth = hdr0.reclth if hdr0.reclth >= 128 else 4096
                    f.seek(0, os.SEEK_END)
                    fsize = f.tell()
                    nrecs = fsize // reclth
                    if nrecs <= 0:
                        continue
                    f.seek((nrecs - 1) * reclth)
                    last_chunk = f.read(min(512, reclth))
                    hdr_last = parse_seed_header(last_chunk)
            except Exception:
                continue

            ch = hdr0.channel
            c1 = ch[0:1].lower() if len(ch) >= 1 else "b"
            c2 = ch[1:2].lower() if len(ch) >= 2 else "h"
            cp = ch[2:3].lower() if len(ch) >= 3 else "z"
            stream = f"{hdr0.station.lower()}-{c1}{c2}-{cp}"
            t_start = hdr0.start_epoch
            t_end = hdr_last.start_epoch + hdr_last.no_of_samples * hdr_last.dt
            calib = cfg.get_sensitivity(stream, t_start)
            rel_path = os.path.relpath(fpath, os.path.dirname(os.path.abspath(output_sfd))) if output_sfd else fpath
            rec = SFDRecord(
                stream=stream,
                filepath=os.path.abspath(fpath),
                t_start=t_start,
                t_end=t_end,
                t_start_str=format_sh_time(t_start),
                t_end_str=format_sh_time(t_end),
                recno=nrecs,
                swap_hdr=0 if hdr0.big_endian_hdr else 1,
                reclth=reclth,
                byteoff=0,
                calib=calib,
                net_loc=f"{hdr0.network:<2s}{hdr0.locid:<2s}",
            )
            records.append((rel_path, rec))

    records.sort(key=lambda item: (item[1].stream, item[1].t_start))
    if output_sfd:
        with open(output_sfd, "w", encoding="ascii") as out:
            for rel_path, r in records:
                out.write(
                    f"s>{r.stream:<11s} n>{rel_path} "
                    f"b>{r.t_start_str} e>{r.t_end_str} "
                    f"r>{r.recno:6d} h>{r.swap_hdr:1d} l>{r.reclth:5d} "
                    f"o>{r.byteoff:7d} c>{r.calib:9.4e} a>{r.net_loc}\n"
                )
    return [r for _, r in records]


def read_from_sfd(
    sfdfile: str,
    streams: Optional[List[str]] = None,
    t_start: Optional[float] = None,
    duration: Optional[float] = None,
    config: Optional[SHMConfig] = None,
) -> List[Trace]:
    """Read requested streams and time window using an SFD index file."""
    from shm_app.io.mseed import read_mseed_file

    cfg = config or get_config()
    records = parse_sfd_file(sfdfile)
    if not records:
        return []

    sfd_dir = os.path.dirname(os.path.abspath(sfdfile))
    wanted = {s.strip().lower() for s in streams} if streams else None
    out: List[Trace] = []
    for rec in records:
        if wanted is not None and rec.stream not in wanted:
            # Also support wildcard matching (e.g. 'gr*-bh-z' or 'bfo-bh-*')
            if not any(fnmatch.fnmatch(rec.stream, pat) for pat in wanted):
                continue
        if t_start is not None and duration is not None:
            if rec.t_end < t_start or rec.t_start > (t_start + duration):
                continue
        target_path = rec.filepath
        if not os.path.isfile(target_path):
            fallback_path = os.path.join(sfd_dir, os.path.basename(rec.filepath))
            if os.path.isfile(fallback_path):
                target_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Mini-SEED file referenced by SFD '{sfdfile}' not found: {rec.filepath}"
                )
        calib = rec.calib if rec.calib > 0 else None
        trs = read_mseed_file(
            target_path,
            t_start=t_start,
            duration=duration,
            calib_override=calib,
            config=cfg,
        )
        out.extend(trs)
    return out
