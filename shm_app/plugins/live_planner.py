"""Event relevance filtering, station selection, azimuthal thinning, and travel-time window planning."""

from __future__ import annotations

import datetime
import math
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from shm_app.core.config_station import (
    GRF_STATIONS,
    GRSN_STATIONS,
    SHMConfig,
    geress_stations,
    get_config,
    resolve_reference_station,
)
from shm_app.plugins.bgr_fetch import (
    BGR_FDSN_DATASELECT_URL,
    HttpOpener,
    StationFetchTarget,
    _default_http_open,
    expand_channels,
    format_fdsn_time,
)
from shm_app.plugins.live_events import LiveEvent
from shm_app.seismology.traveltime import TravelTimeManager, get_ttt_manager, sphere_diff

DEFAULT_LIVE_CACHE_DIR = os.path.expanduser("~/shm_live_events")

EIDA_ROUTING_URL = "https://www.orfeus-eu.org/eidaws/routing/1/query"
GFZ_STATION_URL = "https://geofon.gfz.de/fdsnws/station/1/query"
BGR_STATION_URL = "https://eida.bgr.de/fdsnws/station/1/query"
IRIS_STATION_URL = "https://service.iris.edu/fdsnws/station/1/query"

GFZ_DATASELECT_URL = "https://geofon.gfz.de/fdsnws/dataselect/1/query"
IRIS_DATASELECT_URL = "https://service.iris.edu/fdsnws/dataselect/1/query"

GRF_CORE_ANCHORS = ("GRA1", "GRA2", "GRB1", "GRC1", "GRFO")
GRSN_DISTRIBUTED_ANCHORS = ("BFO", "CLZ", "FUR", "WET", "TNS", "BUG", "RUE", "MOX", "BRG", "BSEG")


@dataclass
class LiveConfig:
    """User configuration for Live Event monitoring, filtering, station planning, and loading."""

    source_key: str = "emsc_ws"
    ref_station: str = "GRFO"
    ref_lat: float = 49.6919
    ref_lon: float = 11.2217

    # Relevance filter (tiered distance vs magnitude)
    use_tiered_filter: bool = True
    local_max_deg: float = 3.0
    min_mag_local: float = 2.0
    regional_max_deg: float = 20.0
    min_mag_regional: float = 3.8
    min_mag_tele: float = 5.5
    flat_min_mag: float = 4.5
    max_distance_deg: float = 180.0
    include_quarry_blasts: bool = False

    # Station selection strategy
    station_strategy: str = "observatory"  # "observatory" | "epicenter"
    station_preset: str = "AUTO"  # "AUTO" | "GRF" | "GRSN" | "GRF+GRSN" | "GERESS" | "CUSTOM"
    custom_stations: Tuple[str, ...] = ()
    max_stations: int = 8
    epicenter_max_radius_deg: float = 12.0
    channel_band: str = "BH"
    components: Tuple[str, ...] = ("Z", "N", "E")

    # Time window & FDSN archive grace settings
    pre_p_sec: float = 60.0
    post_s_sec: float = 180.0
    max_window_sec: float = 900.0
    archive_grace_sec: float = 75.0
    retry_delays_sec: Tuple[int, ...] = (90, 180, 360)

    # Workspace auto-load & post-processing policy
    load_policy: str = "notify"  # "notify" | "auto_idle" | "auto_always"
    auto_theo_phases: bool = True
    auto_filter_expr: str = ""
    output_base_dir: str = DEFAULT_LIVE_CACHE_DIR


@dataclass(frozen=True)
class EventEvaluation:
    """Result of evaluating an earthquake catalog event against observatory & magnitude rules."""

    relevant: bool
    tier: str  # "local" | "regional" | "tele"
    min_mag_required: float
    dist_deg: float
    baz_deg: float  # Backazimuth at observatory pointing toward epicenter
    az_ev_to_obs: float
    first_phase: str
    first_arrival_sec: float
    first_arrival_epoch: float
    slowness_s_deg: float
    region_id: int
    region_name: str
    sregion_id: int
    sregion_name: str
    region_display: str
    reason: str = ""


@dataclass(frozen=True)
class EventFetchPlan:
    """Complete station, time-window, and schedule plan for a single LiveEvent."""

    event: LiveEvent
    evaluation: EventEvaluation
    targets: Tuple[StationFetchTarget, ...]
    start_epoch: float
    end_epoch: float
    earliest_arrival_epoch: float
    ready_at_epoch: float
    output_dir: str
    strategy_used: str
    fallback_note: str = ""

    @property
    def duration_sec(self) -> float:
        return max(1.0, self.end_epoch - self.start_epoch)


def resolve_reference_coords(live_cfg: LiveConfig, shm_cfg: SHMConfig) -> Tuple[float, float, str]:
    """Resolve observatory reference coordinates from station database when available."""
    code, lat, lon = resolve_reference_station(
        shm_cfg,
        ref_station=live_cfg.ref_station or "",
        fallback_lat=live_cfg.ref_lat,
        fallback_lon=live_cfg.ref_lon,
        default_code="REF",
    )
    return lat, lon, code


def estimate_first_arrival(
    ttt: TravelTimeManager,
    dist_deg: float,
    depth_km: float,
) -> Tuple[str, float, float]:
    """Return (phase_name, travel_time_s, slowness_s_deg) for earliest expected wave arrival."""
    d = max(0.05, float(dist_deg))
    z = max(0.0, float(depth_km))
    if d <= 2.5:
        candidates = ("Pg", "Pn", "P")
    elif d <= 105.0:
        candidates = ("P", "Pn", "PKPdf")
    else:
        candidates = ("PKPdf", "PKPab", "PKPbc", "Pdiff", "P")

    best: Optional[Tuple[float, str, float]] = None
    for ph in candidates:
        tt = ttt.travel_time(ph, d, z)
        if tt is not None and tt > 0.0:
            slo = ttt.slowness(ph, d, z) or 0.0
            if best is None or tt < best[0]:
                best = (float(tt), ph, float(slo))
    if best is not None:
        return best[1], best[0], best[2]

    # Smooth fallback (~111.19 km/deg divided by ~7.5 km/s) if beyond table range
    approx_tt = max(1.0, (d * 111.19) / (6.0 if d <= 3.0 else 9.5))
    return ("Pg" if d <= 2.5 else "P"), approx_tt, (111.19 / 8.0 if d <= 90.0 else 4.5)


def estimate_coda_end_travel_time(
    ttt: TravelTimeManager,
    dist_deg: float,
    depth_km: float,
    first_tt_sec: float,
) -> float:
    """Return travel time (seconds from origin) of main S / Lg wave train end."""
    d = max(0.05, float(dist_deg))
    z = max(0.0, float(depth_km))
    candidates = ("Sg", "Sn", "S") if d <= 2.5 else ("S", "SKS", "SS")
    for ph in candidates:
        tt = ttt.travel_time(ph, d, z)
        if tt is not None and tt > first_tt_sec:
            return float(tt)
    return float(first_tt_sec) * 1.78 + 20.0


def evaluate_event_relevance(
    event: LiveEvent,
    live_cfg: LiveConfig,
    shm_cfg: Optional[SHMConfig] = None,
    ttt: Optional[TravelTimeManager] = None,
) -> EventEvaluation:
    """Compute distance, backazimuth, first arrival, Flinn-Engdahl region, and filter verdict."""
    cfg = shm_cfg or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)
    ref_lat, ref_lon, _ref_code = resolve_reference_coords(live_cfg, cfg)

    dist_deg, baz_deg, az_ev_to_obs = sphere_diff(ref_lat, ref_lon, event.lat, event.lon)
    first_phase, first_tt_sec, slowness = estimate_first_arrival(tm, dist_deg, event.depth_km)

    igreg, grname, isreg, srname = cfg.fer.lookup(event.lat, event.lon)
    region_display = (event.region or grname or srname or "Unknown Region").strip()

    if dist_deg <= float(live_cfg.local_max_deg):
        tier = "local"
        req_mag = float(live_cfg.min_mag_local)
    elif dist_deg <= float(live_cfg.regional_max_deg):
        tier = "regional"
        req_mag = float(live_cfg.min_mag_regional)
    else:
        tier = "tele"
        req_mag = float(live_cfg.min_mag_tele)

    if not live_cfg.use_tiered_filter:
        req_mag = float(live_cfg.flat_min_mag)

    ev_type_l = (event.event_type or "").lower()
    if not live_cfg.include_quarry_blasts and any(
        tok in ev_type_l for tok in ("quarry", "blast", "explosion", "sonic")
    ):
        relevant = False
        reason = f"Excluded event type ({event.event_type})"
    elif dist_deg > float(live_cfg.max_distance_deg):
        relevant = False
        reason = f"Distance {dist_deg:.1f}° > max {live_cfg.max_distance_deg:.1f}°"
    elif float(event.magnitude) < req_mag - 1e-6:
        relevant = False
        reason = f"M{event.magnitude:.1f} < {tier} threshold M{req_mag:.1f}"
    else:
        relevant = True
        reason = f"Matched {tier} rule (Δ={dist_deg:.1f}°, M{event.magnitude:.1f} >= M{req_mag:.1f})"

    return EventEvaluation(
        relevant=relevant,
        tier=tier,
        min_mag_required=req_mag,
        dist_deg=float(dist_deg),
        baz_deg=float(baz_deg),
        az_ev_to_obs=float(az_ev_to_obs),
        first_phase=first_phase,
        first_arrival_sec=float(first_tt_sec),
        first_arrival_epoch=float(event.origin_epoch) + float(first_tt_sec),
        slowness_s_deg=float(slowness),
        region_id=igreg,
        region_name=grname or region_display,
        sregion_id=isreg,
        sregion_name=srname,
        region_display=region_display,
        reason=reason,
    )


def _rank_candidates_by_distance(
    candidates: Sequence[Tuple[str, str, str]],
    ev_lat: float,
    ev_lon: float,
    shm_cfg: SHMConfig,
) -> List[Tuple[float, float, str, str, str]]:
    """Return list of (dist_deg, baz_deg, net, sta, url) sorted closest-first to epicenter."""
    ranked: List[Tuple[float, float, str, str, str]] = []
    seen = set()
    for net, sta, base_url in candidates:
        key = (net.upper(), sta.upper())
        if key in seen:
            continue
        seen.add(key)
        st = shm_cfg.get_station(sta)
        if st is not None and (abs(st.lat) > 1e-6 or abs(st.lon) > 1e-6):
            dist_deg, baz_deg, _ = sphere_diff(st.lat, st.lon, ev_lat, ev_lon)
        else:
            dist_deg, baz_deg = 999.0, 0.0
        ranked.append((dist_deg, baz_deg, key[0], key[1], base_url))
    ranked.sort(key=lambda item: (item[0], item[3]))
    return ranked


def select_observatory_targets(
    event: LiveEvent,
    evaluation: EventEvaluation,
    live_cfg: LiveConfig,
    shm_cfg: SHMConfig,
) -> List[StationFetchTarget]:
    """Select meaningful BGR/GRF/GRSN/GERESS stations for observatory-focused analysis."""
    channels = tuple(expand_channels(live_cfg.channel_band, live_cfg.components))
    max_n = max(1, int(live_cfg.max_stations))
    preset = (live_cfg.station_preset or "AUTO").upper()

    if preset == "CUSTOM" and live_cfg.custom_stations:
        pool = [("GR", s.strip().upper(), BGR_FDSN_DATASELECT_URL) for s in live_cfg.custom_stations if s.strip()]
    elif preset == "GRF":
        pool = [("GR", s, BGR_FDSN_DATASELECT_URL) for s in GRF_STATIONS]
    elif preset == "GRSN":
        pool = [("GR", s, BGR_FDSN_DATASELECT_URL) for s in GRSN_STATIONS]
    elif preset == "GERESS":
        pool = [("GR", s, BGR_FDSN_DATASELECT_URL) for s in geress_stations(shm_cfg)]
    elif preset == "GRF+GRSN":
        pool = [("GR", s, BGR_FDSN_DATASELECT_URL) for s in (*GRF_CORE_ANCHORS, *GRSN_DISTRIBUTED_ANCHORS, *GRF_STATIONS, *GRSN_STATIONS)]
    else:
        # AUTO adaptive by distance tier
        if evaluation.tier == "local":
            all_bgr = [*GRF_STATIONS, *GRSN_STATIONS, *geress_stations(shm_cfg)]
            ranked = _rank_candidates_by_distance(
                [("GR", s, BGR_FDSN_DATASELECT_URL) for s in all_bgr],
                event.lat,
                event.lon,
                shm_cfg,
            )
            return [
                StationFetchTarget(
                    station=sta,
                    network=net,
                    location="*",
                    channels=channels,
                    base_url=url,
                    distance_deg=dist_deg,
                    azimuth_deg=baz_deg,
                )
                for dist_deg, baz_deg, net, sta, url in ranked[:max_n]
            ]
        elif evaluation.tier == "regional":
            anchors = ["GRA1", "GRB1", "GRC1", "GRFO", *GRSN_DISTRIBUTED_ANCHORS, *GRSN_STATIONS]
            pool = [("GR", s, BGR_FDSN_DATASELECT_URL) for s in anchors]
        else:
            anchors = [*GRF_CORE_ANCHORS, "BFO", "CLZ", "FUR", "WET", "TNS", "BRG", *GRF_STATIONS, *GRSN_STATIONS]
            pool = [("GR", s, BGR_FDSN_DATASELECT_URL) for s in anchors]

    # Deduplicate preserving anchor priority, then enrich dist/azimuth
    targets: List[StationFetchTarget] = []
    seen_sta = set()
    for net, sta, url in pool:
        if sta in seen_sta:
            continue
        seen_sta.add(sta)
        st = shm_cfg.get_station(sta)
        if st is not None and (abs(st.lat) > 1e-6 or abs(st.lon) > 1e-6):
            dist_deg, baz_deg, _ = sphere_diff(st.lat, st.lon, event.lat, event.lon)
        else:
            dist_deg, baz_deg = evaluation.dist_deg, evaluation.baz_deg
        targets.append(
            StationFetchTarget(
                station=sta,
                network=net,
                location="*",
                channels=channels,
                base_url=url,
                distance_deg=dist_deg,
                azimuth_deg=baz_deg,
            )
        )
        if len(targets) >= max_n:
            break
    return targets


@dataclass(frozen=True)
class DiscoveredStation:
    """Candidate station returned by FDSNWS station catalog, EIDA routing, or local DB."""

    network: str
    station: str
    lat: float
    lon: float
    dataselect_url: str
    location: str = "*"
    dist_deg: float = 0.0
    az_from_ev: float = 0.0
    baz_to_ev: float = 0.0


def parse_fdsn_station_text(
    text: str,
    dataselect_url: str,
    ev_lat: float,
    ev_lon: float,
) -> List[DiscoveredStation]:
    """Parse FDSNWS station/1/query `format=text&level=station` pipe-delimited output."""
    out: List[DiscoveredStation] = []
    seen = set()
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 4:
            continue
        net, sta = cols[0].upper(), cols[1].upper()
        if not net or not sta or (net, sta) in seen:
            continue
        try:
            lat = float(cols[2])
            lon = float(cols[3])
        except ValueError:
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        seen.add((net, sta))
        dist_deg, baz_to_ev, az_from_ev = sphere_diff(lat, lon, ev_lat, ev_lon)
        out.append(
            DiscoveredStation(
                network=net,
                station=sta,
                lat=lat,
                lon=lon,
                dataselect_url=dataselect_url,
                location="*",
                dist_deg=dist_deg,
                az_from_ev=az_from_ev,
                baz_to_ev=baz_to_ev,
            )
        )
    return out


def parse_eida_routing_post(text: str) -> Dict[Tuple[str, str], str]:
    """Parse EIDA routing `service=dataselect&format=post` into {(NET, STA): dataselect_url}."""
    mapping: Dict[Tuple[str, str], str] = {}
    current_url = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("http://") or line.startswith("https://"):
            current_url = line
            continue
        if current_url:
            parts = line.split()
            if len(parts) >= 2:
                net, sta = parts[0].upper(), parts[1].upper()
                if net != "*" and sta != "*":
                    mapping.setdefault((net, sta), current_url)
    return mapping


def select_azimuthally_balanced_stations(
    candidates: Sequence[DiscoveredStation],
    max_stations: int,
) -> List[DiscoveredStation]:
    """Pick closest unique stations evenly distributed across 360° azimuthal sectors."""
    max_n = max(1, int(max_stations))
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (c.dist_deg, c.station))
    n_sectors = max(1, min(max_n, 8))
    sector_width = 360.0 / n_sectors

    chosen: List[DiscoveredStation] = []
    chosen_keys = set()
    used_sectors = set()

    # Pass 1: closest station per azimuthal sector around epicenter
    for cand in ordered:
        sector = int((cand.az_from_ev % 360.0) // sector_width)
        if sector in used_sectors or (cand.network, cand.station) in chosen_keys:
            continue
        used_sectors.add(sector)
        chosen.append(cand)
        chosen_keys.add((cand.network, cand.station))
        if len(chosen) >= max_n:
            return chosen

    # Pass 2: fill remaining quota with closest remaining stations
    for cand in ordered:
        if (cand.network, cand.station) in chosen_keys:
            continue
        chosen.append(cand)
        chosen_keys.add((cand.network, cand.station))
        if len(chosen) >= max_n:
            break
    return chosen


def _discover_epicenter_stations_local(
    event: LiveEvent,
    shm_cfg: SHMConfig,
    max_radius_deg: float,
) -> List[DiscoveredStation]:
    """Fallback station discovery using the local SHM station table (~1,395 stations)."""
    gr_set = set(GRF_STATIONS) | set(GRSN_STATIONS) | set(geress_stations(shm_cfg))
    out: List[DiscoveredStation] = []
    for code, st in shm_cfg.stations.items():
        if abs(st.lat) < 1e-6 and abs(st.lon) < 1e-6:
            continue
        dist_deg, baz_to_ev, az_from_ev = sphere_diff(st.lat, st.lon, event.lat, event.lon)
        if dist_deg <= max_radius_deg:
            # Prefer BGR endpoint for known GR stations; IRIS otherwise
            is_gr = code in gr_set
            out.append(
                DiscoveredStation(
                    network="GR" if is_gr else "*",
                    station=code,
                    lat=st.lat,
                    lon=st.lon,
                    dataselect_url=BGR_FDSN_DATASELECT_URL if is_gr else IRIS_DATASELECT_URL,
                    location="*",
                    dist_deg=dist_deg,
                    az_from_ev=az_from_ev,
                    baz_to_ev=baz_to_ev,
                )
            )
    return out


def select_epicenter_targets(
    event: LiveEvent,
    live_cfg: LiveConfig,
    shm_cfg: SHMConfig,
    opener: Optional[HttpOpener] = None,
    allow_network: bool = True,
    timeout: float = 10.0,
) -> Tuple[List[StationFetchTarget], str]:
    """Discover epicenter-near broadband stations via FDSNWS/EIDA routing with local fallback."""
    channels = tuple(expand_channels(live_cfg.channel_band, live_cfg.components))
    max_n = max(1, int(live_cfg.max_stations))
    max_rad = max(1.0, float(live_cfg.epicenter_max_radius_deg))
    http_open = opener or (_default_http_open if allow_network else None)

    discovered: List[DiscoveredStation] = []
    note = ""

    if http_open is not None:
        win_start = format_fdsn_time(event.origin_epoch - 120.0)
        win_end = format_fdsn_time(event.origin_epoch + 1800.0)
        band = live_cfg.channel_band.strip().upper() or "BH"
        cha_query = band if ("," in band or "?" in band or len(band) >= 3) else f"{band[:2]}?"
        station_sources = [
            (GFZ_STATION_URL, GFZ_DATASELECT_URL, "GFZ"),
            (BGR_STATION_URL, BGR_FDSN_DATASELECT_URL, "BGR"),
            (IRIS_STATION_URL, IRIS_DATASELECT_URL, "IRIS"),
        ]
        for sta_url, ds_url, _tag in station_sources:
            qs = urllib.parse.urlencode(
                {
                    "format": "text",
                    "level": "station",
                    "latitude": f"{event.lat:.4f}",
                    "longitude": f"{event.lon:.4f}",
                    "minradius": "0.0",
                    "maxradius": f"{max_rad:.2f}",
                    "channel": cha_query,
                    "starttime": win_start,
                    "endtime": win_end,
                }
            )
            try:
                status, raw = http_open(f"{sta_url}?{qs}", timeout)
            except Exception:
                continue
            if status == 200 and raw:
                discovered.extend(
                    parse_fdsn_station_text(
                        raw.decode("utf-8", errors="replace"),
                        dataselect_url=ds_url,
                        ev_lat=event.lat,
                        ev_lon=event.lon,
                    )
                )
        # Optional EIDA routing refinement for European networks when multiple stations found
        if discovered and http_open is not None:
            nets = ",".join(sorted({d.network for d in discovered if d.network != "*"}))
            stas = ",".join(sorted({d.station for d in discovered[:40]}))
            if nets and stas:
                rqs = urllib.parse.urlencode(
                    {
                        "service": "dataselect",
                        "format": "post",
                        "network": nets,
                        "station": stas,
                        "starttime": win_start,
                        "endtime": win_end,
                    }
                )
                try:
                    rst, rraw = http_open(f"{EIDA_ROUTING_URL}?{rqs}", timeout)
                    if rst == 200 and rraw:
                        routing = parse_eida_routing_post(rraw.decode("utf-8", errors="replace"))
                        if routing:
                            discovered = [
                                DiscoveredStation(
                                    network=d.network,
                                    station=d.station,
                                    lat=d.lat,
                                    lon=d.lon,
                                    dataselect_url=routing.get((d.network, d.station), d.dataselect_url),
                                    location=d.location,
                                    dist_deg=d.dist_deg,
                                    az_from_ev=d.az_from_ev,
                                    baz_to_ev=d.baz_to_ev,
                                )
                                for d in discovered
                            ]
                except Exception:
                    pass

    if not discovered:
        # Stepwise local database fallback (configured radius -> widened 45° -> empty -> observatory)
        for rad in (max_rad, max(max_rad, 45.0)):
            discovered = _discover_epicenter_stations_local(event, shm_cfg, rad)
            if discovered:
                note = f"Offline station DB fallback within {rad:.0f}°"
                break

    if not discovered:
        return [], "No epicenter-near stations found"

    chosen = select_azimuthally_balanced_stations(discovered, max_n)
    targets = [
        StationFetchTarget(
            station=c.station,
            network=c.network,
            location=c.location,
            channels=channels,
            base_url=c.dataselect_url,
            distance_deg=c.dist_deg,
            azimuth_deg=c.baz_to_ev,
        )
        for c in chosen
    ]
    return targets, note


def format_event_tag(event: LiveEvent) -> str:
    """Create a sortable, filesystem-safe folder name for an event's Mini-SEED cache."""
    dt = datetime.datetime.fromtimestamp(event.origin_epoch, tz=datetime.timezone.utc)
    reg = re.sub(r"[^A-Za-z0-9]+", "_", (event.region or event.event_id)).strip("_")[:28]
    if not reg:
        reg = "EVENT"
    return f"{dt.strftime('%Y%m%d_%H%M%S')}_M{event.magnitude:.1f}_{reg}"


def plan_event_fetch_window(
    event: LiveEvent,
    evaluation: EventEvaluation,
    live_cfg: LiveConfig,
    shm_cfg: Optional[SHMConfig] = None,
    ttt: Optional[TravelTimeManager] = None,
    opener: Optional[HttpOpener] = None,
    allow_network: bool = True,
    now_epoch: Optional[float] = None,
) -> EventFetchPlan:
    """Build complete station list, Mini-SEED UTC time window, and archive readiness schedule."""
    cfg = shm_cfg or get_config()
    tm = ttt or get_ttt_manager(cfg.inputs_dir)
    now_ep = float(now_epoch) if now_epoch is not None else time.time()

    note = ""
    strategy_used = live_cfg.station_strategy
    if live_cfg.station_strategy == "epicenter":
        targets, note = select_epicenter_targets(
            event,
            live_cfg,
            cfg,
            opener=opener,
            allow_network=allow_network,
        )
        if not targets:
            strategy_used = "observatory_fallback"
            note = "Epicenter lookup empty; fell back to BGR observatory stations"
            targets = select_observatory_targets(event, evaluation, live_cfg, cfg)
    else:
        targets = select_observatory_targets(event, evaluation, live_cfg, cfg)

    # Compute earliest arrival & coda window across selected stations (or evaluation fallback)
    arrival_secs: List[float] = []
    coda_secs: List[float] = []
    for tgt in targets:
        d = tgt.distance_deg if tgt.distance_deg > 0.0 else evaluation.dist_deg
        _ph, t_first, _slo = estimate_first_arrival(tm, d, event.depth_km)
        t_coda = estimate_coda_end_travel_time(tm, d, event.depth_km, t_first)
        arrival_secs.append(t_first)
        coda_secs.append(t_coda)

    if not arrival_secs:
        arrival_secs = [evaluation.first_arrival_sec]
        coda_secs = [estimate_coda_end_travel_time(tm, evaluation.dist_deg, event.depth_km, evaluation.first_arrival_sec)]

    earliest_arr_ep = float(event.origin_epoch) + min(arrival_secs)
    start_ep = earliest_arr_ep - max(10.0, float(live_cfg.pre_p_sec))

    # Desired end covers up to latest S + post_s buffer, clamped to [180s, max_window_sec]
    coda_end_ep = float(event.origin_epoch) + max(coda_secs) + max(30.0, float(live_cfg.post_s_sec))
    max_win = max(120.0, float(live_cfg.max_window_sec))
    duration = min(max_win, max(180.0, coda_end_ep - start_ep))
    end_ep = start_ep + duration

    # Archive readiness timestamp: wait until wave end + grace period (immediate if historical)
    ready_at_ep = max(now_ep, end_ep + max(0.0, float(live_cfg.archive_grace_sec)))
    base_dir = os.path.expanduser(live_cfg.output_base_dir or DEFAULT_LIVE_CACHE_DIR)
    out_dir = os.path.join(base_dir, format_event_tag(event))

    return EventFetchPlan(
        event=event,
        evaluation=evaluation,
        targets=tuple(targets),
        start_epoch=start_ep,
        end_epoch=end_ep,
        earliest_arrival_epoch=earliest_arr_ep,
        ready_at_epoch=ready_at_ep,
        output_dir=out_dir,
        strategy_used=strategy_used,
        fallback_note=note,
    )
