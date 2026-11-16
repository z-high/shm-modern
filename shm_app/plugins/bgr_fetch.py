"""BGR EIDA FDSN-WS Mini-SEED waveform downloader plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import os
import re
from typing import Callable, List, Optional, Sequence, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

from shm_app.core.config_station import (
    GERESS_STATIONS_DEFAULT,
    GRF_STATIONS,
    GRSN_STATIONS,
    SHMConfig,
    geress_stations,
    get_config,
    parse_station_list,
)
from shm_app.core.models import Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.mseed import (
    generate_sfd_file,
    parse_seed_header,
    read_mseed_bytes,
)

BGR_FDSN_DATASELECT_URL = "https://eida.bgr.de/fdsnws/dataselect/1/query"
DEFAULT_TIMEOUT_S = 45.0
DEFAULT_DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "shm_bgr_downloads")

HttpOpener = Callable[[str, float], Tuple[int, bytes]]


@dataclass
class StationDownloadResult:
    """Download outcome for a single requested station."""

    station: str
    network: str
    status: str  # "ok", "nodata", "error", "cancelled"
    filepath: Optional[str] = None
    url: str = ""
    n_bytes: int = 0
    n_traces: int = 0
    message: str = ""
    traces: List[Trace] = field(default_factory=list)


@dataclass(frozen=True)
class StationFetchTarget:
    """Single station target for multi-network / multi-endpoint FDSNWS fetches."""

    station: str
    network: str = "GR"
    location: str = "*"
    channels: Tuple[str, ...] = ("BHZ", "BHN", "BHE")
    base_url: str = BGR_FDSN_DATASELECT_URL
    distance_deg: float = 0.0
    azimuth_deg: float = 0.0


@dataclass
class BGRFetchSummary:
    """Aggregated outcome across all requested BGR stations."""

    output_dir: str
    sfd_path: Optional[str]
    results: List[StationDownloadResult]
    traces: List[Trace]
    cancelled: bool = False


def format_fdsn_time(epoch: float) -> str:
    """Format a UTC POSIX epoch float into FDSN ISO-8601 timestamp."""
    rounded = round(float(epoch) * 1000.0) / 1000.0
    dt = datetime.datetime.fromtimestamp(rounded, tz=datetime.timezone.utc)
    ms = dt.microsecond // 1000
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{ms:03d}" if ms else base


def parse_time_input(value: Union[float, int, str]) -> float:
    """Convert numeric epoch or SH/ISO timestamp string to UTC epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    return parse_sh_time(str(value).strip())


def expand_channels(
    chan_spec: Union[str, Sequence[str]],
    components: Sequence[str] = ("Z", "N", "E"),
) -> List[str]:
    """Expand a channel band/pattern spec and component letters into FDSN channel patterns."""
    raw_items = [chan_spec] if isinstance(chan_spec, str) else list(chan_spec)
    comps = [c.strip().upper() for c in components if c.strip()]
    out: List[str] = []
    seen = set()

    for item in raw_items:
        for tok in re.split(r"[\s,;]+", str(item).strip()):
            pat = tok.strip().upper()
            if not pat:
                continue
            if len(pat) >= 3 or "*" in pat or "?" in pat:
                cands = [pat]
            elif len(pat) == 2 and comps:
                cands = [f"{pat}{c}" for c in comps]
            elif len(pat) == 2:
                cands = [f"{pat}?"]
            else:
                cands = [f"{pat}*"]
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
    return out


def build_dataselect_url(
    base_url: str,
    network: str,
    station: str,
    channels: Union[str, Sequence[str]],
    start_epoch: float,
    end_epoch: float,
    location: str = "*",
    nodata: int = 404,
) -> str:
    """Construct a standard FDSNWS dataselect/1/query GET URL."""
    if end_epoch <= start_epoch:
        raise ValueError("End time must be after start time")
    sta = station.strip().upper()
    if not sta:
        raise ValueError("Station code must not be empty")
    chan_list = expand_channels(channels, components=()) if isinstance(channels, str) else list(channels)
    chan_str = ",".join(c.strip().upper() for c in chan_list if c.strip())
    if not chan_str:
        raise ValueError("At least one channel must be specified")

    net = (network or "GR").strip().upper()
    loc = (location or "*").strip()
    params = [
        ("network", net),
        ("station", sta),
    ]
    if loc and loc != "*":
        params.append(("location", loc))
    params.extend([
        ("channel", chan_str),
        ("starttime", format_fdsn_time(start_epoch)),
        ("endtime", format_fdsn_time(end_epoch)),
        ("nodata", str(int(nodata))),
    ])
    clean_base = (base_url or BGR_FDSN_DATASELECT_URL).strip().rstrip("?")
    return f"{clean_base}?{urllib.parse.urlencode(params)}"


def _default_http_open(url: str, timeout: float = DEFAULT_TIMEOUT_S) -> Tuple[int, bytes]:
    """Perform HTTP GET request for Mini-SEED bytes, mapping 204/404 to empty payload."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SHM-Modern/1.0 FDSN-Client",
            "Accept": "application/vnd.fdsn.mseed, application/octet-stream, */*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            return status, resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (204, 404):
            return int(exc.code), b""
        detail = ""
        try:
            detail = exc.read(256).decode("utf-8", errors="ignore").strip()
        except Exception:
            pass
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"HTTP {exc.code} ({exc.reason}){suffix}") from exc


def _safe_filename_part(val: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", val.strip())
    return cleaned or "UNK"


def _is_plausible_mseed(raw: bytes) -> bool:
    if len(raw) < 64:
        return False
    try:
        hdr = parse_seed_header(raw[: min(512, len(raw))])
        return hdr.indicator in ("D", "R", "Q", "M", " ") and hdr.reclth >= 64
    except Exception:
        return False


def fetch_station_mseed(
    station: str,
    start_epoch: float,
    duration: float,
    output_dir: str,
    network: str = "GR",
    channels: Union[str, Sequence[str]] = ("BHZ", "BHN", "BHE"),
    location: str = "*",
    base_url: str = BGR_FDSN_DATASELECT_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
    config: Optional[SHMConfig] = None,
    opener: Optional[HttpOpener] = None,
) -> StationDownloadResult:
    """Download one station's Mini-SEED stream from BGR FDSNWS and decode its Traces."""
    cfg = config or get_config()
    net = (network or "GR").strip().upper()
    sta = station.strip().upper()
    url = build_dataselect_url(
        base_url=base_url,
        network=net,
        station=sta,
        channels=channels,
        start_epoch=start_epoch,
        end_epoch=start_epoch + duration,
        location=location,
    )
    http_get = opener or _default_http_open
    try:
        status_code, raw = http_get(url, timeout)
    except Exception as exc:
        return StationDownloadResult(
            station=sta,
            network=net,
            status="error",
            url=url,
            message=str(exc),
        )

    if status_code in (204, 404) or not raw:
        return StationDownloadResult(
            station=sta,
            network=net,
            status="nodata",
            url=url,
            message="No data available on server",
        )

    if not _is_plausible_mseed(raw):
        snippet = raw[:160].decode("utf-8", errors="ignore").strip().replace("\n", " ")
        return StationDownloadResult(
            station=sta,
            network=net,
            status="error",
            url=url,
            n_bytes=len(raw),
            message=f"Unexpected non-MiniSEED response: {snippet or 'invalid header'}",
        )

    os.makedirs(output_dir, exist_ok=True)
    dt_tag = datetime.datetime.fromtimestamp(start_epoch, tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{_safe_filename_part(net)}.{_safe_filename_part(sta)}.{dt_tag}.mseed"
    out_path = os.path.join(output_dir, fname)

    with open(out_path, "wb") as f:
        f.write(raw)

    traces = read_mseed_bytes(
        raw,
        t_start=start_epoch,
        duration=duration,
        config=cfg,
        source_path=out_path,
    )
    if not traces:
        return StationDownloadResult(
            station=sta,
            network=net,
            status="nodata",
            filepath=out_path,
            url=url,
            n_bytes=len(raw),
            message="Saved Mini-SEED had 0 samples in requested window",
        )

    return StationDownloadResult(
        station=sta,
        network=net,
        status="ok",
        filepath=out_path,
        url=url,
        n_bytes=len(raw),
        n_traces=len(traces),
        message=f"{len(traces)} trace(s), {len(raw) / 1024.0:.1f} KiB",
        traces=traces,
    )


def fetch_fdsn_waveforms(
    targets: Sequence[StationFetchTarget],
    start_time: Union[float, int, str],
    duration: float,
    output_dir: str = DEFAULT_DOWNLOAD_DIR,
    timeout: float = DEFAULT_TIMEOUT_S,
    build_sfd: bool = True,
    config: Optional[SHMConfig] = None,
    opener: Optional[HttpOpener] = None,
    progress_cb: Optional[Callable[[int, int, StationDownloadResult], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BGRFetchSummary:
    """Download Mini-SEED waveforms across heterogeneous FDSNWS endpoints and networks."""
    cfg = config or get_config()
    target_list = list(targets)
    if not target_list:
        raise ValueError("No stations selected for FDSN download")
    start_ep = parse_time_input(start_time)
    dur_s = float(duration)
    if dur_s <= 0.0:
        raise ValueError("Duration must be greater than 0 seconds")

    os.makedirs(output_dir, exist_ok=True)
    results: List[StationDownloadResult] = []
    all_traces: List[Trace] = []
    cancelled = False
    total = len(target_list)

    for idx, tgt in enumerate(target_list, start=1):
        if cancel_check is not None and cancel_check():
            cancelled = True
            break
        res = fetch_station_mseed(
            station=tgt.station,
            start_epoch=start_ep,
            duration=dur_s,
            output_dir=output_dir,
            network=tgt.network,
            channels=list(tgt.channels),
            location=tgt.location,
            base_url=tgt.base_url,
            timeout=timeout,
            config=cfg,
            opener=opener,
        )
        results.append(res)
        if res.traces:
            all_traces.extend(res.traces)
        if progress_cb is not None:
            progress_cb(idx, total, res)

    sfd_path: Optional[str] = None
    if build_sfd and any(r.filepath for r in results):
        sfd_cand = os.path.join(output_dir, "sfdlist.sfd")
        generate_sfd_file(output_dir, output_sfd=sfd_cand, config=cfg)
        if os.path.isfile(sfd_cand):
            sfd_path = sfd_cand

    return BGRFetchSummary(
        output_dir=output_dir,
        sfd_path=sfd_path,
        results=results,
        traces=all_traces,
        cancelled=cancelled,
    )


def fetch_bgr_waveforms(
    stations: Union[str, Sequence[str]],
    start_time: Union[float, int, str],
    duration: float,
    output_dir: str = DEFAULT_DOWNLOAD_DIR,
    network: str = "GR",
    channels: Union[str, Sequence[str]] = ("BHZ", "BHN", "BHE"),
    location: str = "*",
    base_url: str = BGR_FDSN_DATASELECT_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
    build_sfd: bool = True,
    config: Optional[SHMConfig] = None,
    opener: Optional[HttpOpener] = None,
    progress_cb: Optional[Callable[[int, int, StationDownloadResult], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BGRFetchSummary:
    """Download Mini-SEED waveforms from BGR EIDA for multiple stations and a time window."""
    sta_list = parse_station_list(stations)
    if not sta_list:
        raise ValueError("No stations selected for BGR download")
    chan_list = expand_channels(channels) if isinstance(channels, str) else list(channels)
    if not chan_list:
        raise ValueError("No channels specified for BGR download")
    chan_tuple = tuple(chan_list)
    targets = [
        StationFetchTarget(
            station=sta,
            network=network,
            location=location,
            channels=chan_tuple,
            base_url=base_url,
        )
        for sta in sta_list
    ]
    return fetch_fdsn_waveforms(
        targets=targets,
        start_time=start_time,
        duration=duration,
        output_dir=output_dir,
        timeout=timeout,
        build_sfd=build_sfd,
        config=config,
        opener=opener,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )
