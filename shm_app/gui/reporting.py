"""Publication-grade vector observatory bulletin generation (M-20).

Generates multi-page vector PDF and self-contained vector SVG bulletins
headlessly using pure Matplotlib (`Figure`, `PdfPages`, `savefig(..., format="svg")`)
and NumPy.
"""

from __future__ import annotations

import datetime
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np

from shm_app.core.fit_models import ConfidenceEllipse, FitDiagnosticResult
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.gui.map_helpers import (
    distance_ring_segments,
    draw_country_basemap,
    great_circle_segments,
)

try:
    from PyQt6.QtCore import QThread, pyqtSignal

    _HAS_PYQT6 = True
except ImportError:  # pragma: no cover - optional for headless environments
    QThread = object  # type: ignore[misc, assignment]
    pyqtSignal = None  # type: ignore[assignment]
    _HAS_PYQT6 = False


def _extract_focal_mech(
    focal_mech: Any,
    fit_result: Optional[FitDiagnosticResult] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize focal mechanism input (dict, dataclass, or fit_result context) to a dict."""
    raw = focal_mech
    if raw is None and fit_result is not None and isinstance(fit_result.context_data, dict):
        raw = fit_result.context_data.get("focal_mechanism")
    if raw is None:
        return None

    if isinstance(raw, dict):
        strike = raw.get("strike", raw.get("strike_deg"))
        dip = raw.get("dip", raw.get("dip_deg"))
        rake = raw.get("rake", raw.get("rake_deg"))
        if strike is None or dip is None or rake is None:
            return None
        return {
            "strike": float(strike),
            "dip": float(dip),
            "rake": float(rake),
            "misfit": raw.get("misfit"),
            "polarities": raw.get("polarities"),
        }

    strike = getattr(raw, "strike", getattr(raw, "strike_deg", None))
    dip = getattr(raw, "dip", getattr(raw, "dip_deg", None))
    rake = getattr(raw, "rake", getattr(raw, "rake_deg", None))
    if strike is None or dip is None or rake is None:
        return None
    return {
        "strike": float(strike),
        "dip": float(dip),
        "rake": float(rake),
        "misfit": getattr(raw, "misfit", None),
        "polarities": getattr(raw, "polarities", None),
    }


def _collect_station_coords(
    traces: Sequence[Trace],
    fit_result: Optional[FitDiagnosticResult] = None,
    config: Any = None,
) -> Dict[str, Tuple[float, float]]:
    """Collect unique (lat, lon) coordinates per station from traces, fit_result, or config."""
    coords: Dict[str, Tuple[float, float]] = {}
    for tr in traces:
        sta = (tr.station or "").strip()
        if not sta or sta in coords:
            continue
        if tr.lat != 0.0 or tr.lon != 0.0:
            coords[sta] = (float(tr.lat), float(tr.lon))
        elif config is not None and hasattr(config, "get_station"):
            try:
                st = config.get_station(sta)
                if st.lat != 0.0 or st.lon != 0.0:
                    coords[sta] = (float(st.lat), float(st.lon))
            except Exception:
                pass

    if fit_result is not None and config is not None and hasattr(config, "get_station"):
        for entry in fit_result.residuals:
            sta = (entry.station or "").strip()
            if sta and sta not in coords:
                try:
                    st = config.get_station(sta)
                    if st.lat != 0.0 or st.lon != 0.0:
                        coords[sta] = (float(st.lat), float(st.lon))
                except Exception:
                    pass
    return coords


def _resolve_ellipse(
    params: AnalysisParamSet,
    fit_result: Optional[FitDiagnosticResult],
) -> Optional[Tuple[float, float, float]]:
    """Return (semi_major_km, semi_minor_km, strike_deg) if available."""
    if fit_result is not None and fit_result.confidence_ellipse is not None:
        el = fit_result.confidence_ellipse
        if el.semi_major_km > 0.0 or el.semi_minor_km > 0.0:
            return float(el.semi_major_km), float(el.semi_minor_km), float(el.strike_deg)
    if params.error_ellipse is not None and len(params.error_ellipse) == 3:
        a_km, b_km, strike_deg = params.error_ellipse
        if a_km > 0.0 or b_km > 0.0:
            return float(a_km), float(b_km), float(strike_deg)
    return None


def _ellipse_lonlat(
    lat0: float,
    lon0: float,
    semi_major_km: float,
    semi_minor_km: float,
    strike_deg: float,
    n_pts: int = 121,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a local horizontal confidence ellipse (km) onto geographic (lon, lat) degrees."""
    alpha_rad = math.radians(90.0 - strike_deg)
    phi = np.linspace(0.0, 2.0 * math.pi, max(36, n_pts))
    x_raw = semi_major_km * np.cos(phi)
    y_raw = semi_minor_km * np.sin(phi)
    east_km = x_raw * math.cos(alpha_rad) - y_raw * math.sin(alpha_rad)
    north_km = x_raw * math.sin(alpha_rad) + y_raw * math.cos(alpha_rad)

    km_per_deg_lat = 111.1949
    cos_lat = max(0.05, abs(math.cos(math.radians(lat0))))
    km_per_deg_lon = km_per_deg_lat * cos_lat

    el_lons = lon0 + east_km / km_per_deg_lon
    el_lats = lat0 + north_km / km_per_deg_lat
    return el_lons, el_lats


def render_header_and_metadata(
    ax: Any,
    params: AnalysisParamSet,
    traces: Sequence[Trace],
    fit_result: Optional[FitDiagnosticResult] = None,
    *,
    title: str = "SeismicHandler Observatory Bulletin",
    agency: str = "SHM-Modern Observatory",
) -> None:
    """Render element 1: observatory header banner and event metadata table."""
    ax.axis("off")
    ts_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ax.text(
        0.0,
        0.96,
        title,
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )
    ax.text(
        1.0,
        0.96,
        f"Agency: {agency}\nGenerated: {ts_utc}",
        fontsize=8.5,
        color="#334155",
        va="top",
        ha="right",
        transform=ax.transAxes,
    )

    event_id_str = str(params.event_id) if params.event_id else "N/A"
    region_str = (
        f"[{params.region_id}] {params.region_name}"
        if params.region_name
        else (str(params.region_id) if params.region_id else "Unassigned Region")
    )
    origin_str = params.origin or (
        traces[0].start_time_str if traces else "N/A"
    )
    lat_str = f"{params.lat:+.4f}° (±{params.d_lat:.2f} km)" if params.d_lat else f"{params.lat:+.4f}°"
    lon_str = f"{params.lon:+.4f}° (±{params.d_lon:.2f} km)" if params.d_lon else f"{params.lon:+.4f}°"
    depth_str = f"{params.depth:.1f} km (±{params.d_depth:.1f})" if params.d_depth else f"{params.depth:.1f} km"

    mags = []
    if params.mag_ml:
        mags.append(f"ML {params.mag_ml:.2f}")
    if params.mag_mb:
        mags.append(f"mb {params.mag_mb:.2f}")
    if params.mag_ms:
        mags.append(f"MS {params.mag_ms:.2f}")
    if params.mag_mbb:
        mags.append(f"mB {params.mag_mbb:.2f}")
    if params.mag_mw:
        mags.append(f"Mw {params.mag_mw:.2f}")
    mag_str = " | ".join(mags) if mags else "Not computed"

    rmse_val = fit_result.rmse if fit_result is not None else params.rmse
    wrmse_val = fit_result.wrmse if fit_result is not None else params.wrmse
    chi2_val = fit_result.chi2_reduced if fit_result is not None else params.chi2_reduced
    n_picks = sum(len(tr.picks) for tr in traces)

    rows = [
        ["Event ID", event_id_str, "Origin Time", origin_str],
        ["Region", region_str, "Event Type", f"{params.event_type} ({params.source})"],
        ["Epicenter Lat", lat_str, "Epicenter Lon", lon_str],
        ["Focal Depth", depth_str, "Phase / Slowness", f"{params.phase} | {params.slowness:.2f} s/° @ BAZ {params.azimuth:.1f}°"],
        ["Magnitudes", mag_str, "Stations / Picks", f"{len({tr.station for tr in traces})} traces / {n_picks} picks"],
        ["Fit Metrics", f"RMSE={rmse_val:.3f} s  WRMSE={wrmse_val:.3f} s", "Reduced Chi²", f"{chi2_val:.3f}  (κ={params.condition_number:.2f})"],
    ]

    table = ax.table(
        cellText=rows,
        loc="lower center",
        cellLoc="left",
        bbox=[0.0, 0.0, 1.0, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    col_widths = [0.16, 0.34, 0.16, 0.34]
    for (r_idx, c_idx), cell in table.get_celld().items():
        cell.set_width(col_widths[c_idx])
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if c_idx in (0, 2):
            cell.set_facecolor("#f1f5f9")
            cell.set_text_props(fontweight="bold", color="#1e293b")
        else:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(color="#0f172a")


def render_geographic_map(
    ax: Any,
    traces: Sequence[Trace],
    params: AnalysisParamSet,
    fit_result: Optional[FitDiagnosticResult] = None,
    config: Any = None,
) -> None:
    """Render element 2: geographic station/epicenter map with ray paths and 95% error ellipse."""
    ax.set_facecolor("#0f172a")
    draw_country_basemap(
        ax,
        edge_linewidth=0.55,
        grid_color="#1e293b",
        axis_color="#334155",
        axis_lw=0.7,
        axis_alpha=0.9,
    )

    sta_coords = _collect_station_coords(traces, fit_result=fit_result, config=config)
    has_epi = (params.lat != 0.0 or params.lon != 0.0)
    ellipse = _resolve_ellipse(params, fit_result)

    all_lons: List[float] = []
    all_lats: List[float] = []
    if has_epi:
        all_lons.append(float(params.lon))
        all_lats.append(float(params.lat))

    ray_segs: List[np.ndarray] = []
    sta_lons: List[float] = []
    sta_lats: List[float] = []
    for sta, (slat, slon) in sta_coords.items():
        sta_lons.append(slon)
        sta_lats.append(slat)
        all_lons.append(slon)
        all_lats.append(slat)
        if has_epi:
            ray_segs.extend(great_circle_segments(params.lat, params.lon, slat, slon, n_pts=48))

    if ray_segs:
        ax.add_collection(
            LineCollection(ray_segs, colors="#38bdf8", linewidths=0.85, alpha=0.55, zorder=2)
        )

    if sta_lons:
        ax.scatter(
            sta_lons,
            sta_lats,
            marker="^",
            s=42,
            c="#22c55e",
            edgecolors="#ffffff",
            linewidths=0.6,
            zorder=4,
            label="Stations",
        )
        for sta, (slat, slon) in sta_coords.items():
            ax.annotate(
                sta,
                (slon, slat),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="#e2e8f0",
                zorder=5,
            )

    if has_epi:
        for ring_deg in (30.0, 60.0, 90.0):
            for seg in distance_ring_segments(params.lat, params.lon, ring_deg, n_pts=121):
                ax.plot(seg[:, 0], seg[:, 1], color="#475569", lw=0.5, ls="--", alpha=0.5, zorder=1)

        ax.scatter(
            [params.lon],
            [params.lat],
            marker="*",
            s=140,
            c="#facc15",
            edgecolors="#dc2626",
            linewidths=0.8,
            zorder=6,
            label="Epicenter",
        )

        if ellipse is not None:
            a_km, b_km, strike_deg = ellipse
            el_lons, el_lats = _ellipse_lonlat(params.lat, params.lon, a_km, b_km, strike_deg)
            ax.plot(
                el_lons,
                el_lats,
                color="#f97316",
                lw=1.6,
                zorder=5,
                label=f"95% CI ({a_km:.1f}×{b_km:.1f} km)",
            )
            ax.fill(el_lons, el_lats, color="#f97316", alpha=0.22, zorder=4)
            all_lons.extend(el_lons.tolist())
            all_lats.extend(el_lats.tolist())

    if all_lons and all_lats:
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        span_lon = lon_max - lon_min
        span_lat = lat_max - lat_min
        if span_lon > 120.0 or span_lat > 75.0:
            ax.set_xlim(-180.0, 180.0)
            ax.set_ylim(-85.0, 85.0)
        else:
            pad_lon = max(2.0, span_lon * 0.25)
            pad_lat = max(1.5, span_lat * 0.25)
            ax.set_xlim(max(-180.0, lon_min - pad_lon), min(180.0, lon_max + pad_lon))
            ax.set_ylim(max(-85.0, lat_min - pad_lat), min(85.0, lat_max + pad_lat))
    else:
        ax.set_xlim(-180.0, 180.0)
        ax.set_ylim(-85.0, 85.0)

    ax.set_xlabel("Longitude (°)", fontsize=8, color="#334155")
    ax.set_ylabel("Latitude (°)", fontsize=8, color="#334155")
    ax.tick_params(colors="#334155", labelsize=7.5)
    ax.set_title("Station Network, Epicenter & 95% Confidence Ellipse", fontsize=9.5, fontweight="bold", color="#0f172a")
    if sta_lons or has_epi:
        ax.legend(loc="upper right", fontsize=7, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")


def render_beachball(
    ax: Any,
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
    polarities: Optional[Sequence[Tuple[float, float, int]]] = None,
    title: Optional[str] = None,
) -> None:
    """Render element 5: lower-hemisphere Schmidt equal-area focal mechanism beachball."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")

    n_grid = 180
    gx = np.linspace(-1.0, 1.0, n_grid)
    gy = np.linspace(-1.0, 1.0, n_grid)
    xx, yy = np.meshgrid(gx, gy)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    mask = rr <= 1.0

    # Schmidt equal-area projection: r = sqrt(2) * sin(ih / 2)
    r_clip = np.clip(rr, 0.0, 1.0)
    ih = 2.0 * np.arcsin(np.clip(r_clip / math.sqrt(2.0), 0.0, 1.0))
    az = np.arctan2(xx, yy)  # clockwise from North (+y)

    # Ray unit vector in NED (North, East, Down)
    gn = np.sin(ih) * np.cos(az)
    ge = np.sin(ih) * np.sin(az)
    gd = np.cos(ih)

    phi_s = math.radians(strike_deg)
    delta = math.radians(dip_deg)
    lam = math.radians(rake_deg)

    # Fault normal n and slip vector d in NED (Aki & Richards convention)
    n_vec = (
        -math.sin(delta) * math.sin(phi_s),
        math.sin(delta) * math.cos(phi_s),
        -math.cos(delta),
    )
    d_vec = (
        math.cos(lam) * math.cos(phi_s) + math.cos(delta) * math.sin(lam) * math.sin(phi_s),
        math.cos(lam) * math.sin(phi_s) - math.cos(delta) * math.sin(lam) * math.cos(phi_s),
        -math.sin(delta) * math.sin(lam),
    )

    gn_dot = gn * n_vec[0] + ge * n_vec[1] + gd * n_vec[2]
    gd_dot = gn * d_vec[0] + ge * d_vec[1] + gd * d_vec[2]
    rad_p = np.where(mask, 2.0 * gn_dot * gd_dot, np.nan)

    ax.contourf(
        xx,
        yy,
        rad_p,
        levels=[-1e6, 0.0, 1e6],
        colors=["#f8fafc", "#1e3a5f"],
        zorder=1,
    )
    ax.contour(
        xx,
        yy,
        rad_p,
        levels=[0.0],
        colors=["#0f172a"],
        linewidths=1.4,
        zorder=2,
    )

    theta_rim = np.linspace(0.0, 2.0 * math.pi, 240)
    ax.plot(np.cos(theta_rim), np.sin(theta_rim), color="#0f172a", lw=1.6, zorder=3)
    for label, lx, ly in [("N", 0.0, 1.09), ("E", 1.09, 0.0), ("S", 0.0, -1.11), ("W", -1.11, 0.0)]:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#334155")

    if polarities:
        for item in polarities:
            if len(item) < 3:
                continue
            p_az, p_ih, p_pol = float(item[0]), float(item[1]), int(item[2])
            if p_ih > 90.0:
                p_az = (p_az + 180.0) % 360.0
                p_ih = 180.0 - p_ih
            p_ih = max(0.0, min(90.0, p_ih))
            r_p = math.sqrt(2.0) * math.sin(math.radians(p_ih) / 2.0)
            px = r_p * math.sin(math.radians(p_az))
            py = r_p * math.cos(math.radians(p_az))
            if p_pol > 0:
                ax.scatter([px], [py], marker="o", s=26, c="#ef4444", edgecolors="#ffffff", lw=0.6, zorder=4)
            elif p_pol < 0:
                ax.scatter([px], [py], marker="o", s=26, facecolors="#ffffff", edgecolors="#0f172a", lw=1.0, zorder=4)

    hdr = title or f"Focal Mechanism\nS={strike_deg:.0f}° / D={dip_deg:.0f}° / R={rake_deg:.0f}°"
    ax.set_title(hdr, fontsize=9.0, fontweight="bold", color="#0f172a", pad=4)


def render_waveform_panels(
    ax: Any,
    traces: Sequence[Trace],
    params: Optional[AnalysisParamSet] = None,
    *,
    title: str = "Aligned Waveform Panels & Phase Picks",
) -> None:
    """Render element 3: multi-trace waveforms with phase picks, uncertainty bars, and theoretical arrivals."""
    ax.set_facecolor("#ffffff")
    if not traces:
        ax.text(0.5, 0.5, "No waveform traces available", ha="center", va="center", fontsize=10, color="#64748b")
        ax.set_title(title, fontsize=9.5, fontweight="bold", color="#0f172a")
        return

    valid_starts = [tr.start_epoch for tr in traces if tr.length > 0]
    ref_epoch = min(valid_starts) if valid_starts else 0.0

    n_tr = len(traces)
    t_min = float("inf")
    t_max = float("-inf")
    y_ticks: List[float] = []
    y_labels: List[str] = []

    for idx, tr in enumerate(traces):
        y_base = float(n_tr - 1 - idx)
        y_ticks.append(y_base)
        flt_tag = ""
        if tr.filter_name and tr.filter_name != "NONE":
            flt_tag += f" [{tr.filter_name}]"
        if tr.overlay_filter_name and tr.overlay_filter_name != "NONE":
            flt_tag += f" (+{tr.overlay_filter_name})"
        dist_tag = f" {tr.distance:.1f}°" if tr.distance > 0 else ""
        y_labels.append(f"{tr.station} {tr.channel3}{dist_tag}{flt_tag}")

        if tr.length > 0:
            t_ax = tr.time_axis(ref_epoch=ref_epoch)
            arr = np.asarray(tr.data, dtype=np.float64)
            arr_dm = arr - float(np.mean(arr))
            peak = float(np.max(np.abs(arr_dm)))
            norm_arr = (arr_dm / peak) * 0.40 if peak > 1e-15 else np.zeros_like(arr_dm)
            ax.plot(t_ax, y_base + norm_arr, color="#0f172a", lw=0.7, zorder=2)
            t_min = min(t_min, float(t_ax[0]))
            t_max = max(t_max, float(t_ax[-1]))

        # Theoretical arrivals
        for ph_name, ep in tr.theo_picks.items():
            t_rel = float(ep) - ref_epoch + tr.display_offset
            ax.plot([t_rel, t_rel], [y_base - 0.36, y_base + 0.36], color="#7c3aed", ls="--", lw=0.9, alpha=0.8, zorder=3)
            ax.text(t_rel, y_base - 0.38, f"th:{ph_name}", fontsize=6.0, color="#7c3aed", ha="center", va="top", zorder=4)
            t_min = min(t_min, t_rel)
            t_max = max(t_max, t_rel)

        # Phase picks + sigma_t uncertainty bars
        for pk in tr.picks:
            if not pk.onset and pk.onset_epoch == 0.0:
                continue
            t_rel = pk.onset_epoch - ref_epoch + tr.display_offset
            sig = pk.sigma_t(tr.delta)
            ph_up = (pk.name or "").upper()
            pcol = "#dc2626" if ph_up.startswith("P") else ("#2563eb" if ph_up.startswith("S") else "#059669")
            y_span = float(n_tr) + 0.2
            ax.axvspan(
                t_rel - sig,
                t_rel + sig,
                ymin=(y_base - 0.38 - (-0.6)) / y_span,
                ymax=(y_base + 0.38 - (-0.6)) / y_span,
                color=pcol,
                alpha=0.14,
                zorder=1,
            )
            ax.plot([t_rel, t_rel], [y_base - 0.42, y_base + 0.42], color=pcol, lw=1.25, zorder=4)
            pol = pk.sign_char
            lbl = f"{pk.name}{pol}"
            ax.text(t_rel, y_base + 0.43, lbl, fontsize=6.5, fontweight="bold", color=pcol, ha="center", va="bottom", zorder=5)
            t_min = min(t_min, t_rel - sig)
            t_max = max(t_max, t_rel + sig)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7.5, color="#1e293b")
    ax.set_ylim(-0.6, float(n_tr) - 0.4)
    if math.isfinite(t_min) and math.isfinite(t_max) and t_max > t_min:
        pad = 0.03 * (t_max - t_min)
        ax.set_xlim(t_min - pad, t_max + pad)
    ax.grid(True, axis="x", color="#e2e8f0", ls=":", lw=0.6, zorder=0)
    ax.set_xlabel("Time relative to earliest trace start (s)", fontsize=8.0, color="#334155")
    ax.tick_params(axis="x", colors="#334155", labelsize=7.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", color="#0f172a")


def render_diagnostic_summary(
    ax_scatter: Any,
    ax_corr: Any,
    ax_kpi: Optional[Any],
    traces: Sequence[Trace],
    params: AnalysisParamSet,
    fit_result: Optional[FitDiagnosticResult] = None,
) -> None:
    """Render element 4: residual scatter, correlation matrix (or residual bars), and KPI table."""
    # 1. Residual scatter
    ax_scatter.set_facecolor("#ffffff")
    xs: List[float] = []
    ys: List[float] = []
    errs: List[float] = []
    cols: List[str] = []
    xlabel = "Epicentral Distance (°)"

    if fit_result is not None and fit_result.residuals:
        has_dist = any(e.distance_deg != 0.0 for e in fit_result.residuals)
        if not has_dist:
            xlabel = "Observation Index"
        for i, entry in enumerate(fit_result.residuals):
            xs.append(entry.distance_deg if has_dist else float(i + 1))
            ys.append(entry.residual)
            errs.append(max(0.0, entry.uncertainty))
            cols.append("#dc2626" if entry.is_outlier else "#16a34a")
    else:
        idx = 0
        for tr in traces:
            for pk in tr.picks:
                if pk.resid != 0.0 or pk.onset:
                    idx += 1
                    xs.append(tr.distance if tr.distance > 0 else float(idx))
                    ys.append(pk.resid)
                    errs.append(pk.sigma_t(tr.delta))
                    cols.append("#16a34a")
        if all(tr.distance == 0.0 for tr in traces):
            xlabel = "Pick Index"

    ax_scatter.axhline(0.0, color="#64748b", ls="--", lw=0.9, zorder=1)
    if xs:
        ax_scatter.errorbar(
            xs,
            ys,
            yerr=errs if any(e > 0 for e in errs) else None,
            fmt="none",
            ecolor="#94a3b8",
            elinewidth=0.8,
            capsize=2.0,
            zorder=2,
        )
        ax_scatter.scatter(xs, ys, c=cols, edgecolors="#0f172a", s=34, lw=0.5, zorder=3)
    else:
        ax_scatter.text(0.5, 0.5, "No residual data", ha="center", va="center", color="#64748b", transform=ax_scatter.transAxes)
    ax_scatter.grid(True, color="#e2e8f0", ls=":", lw=0.6)
    ax_scatter.set_xlabel(xlabel, fontsize=8.0, color="#334155")
    ax_scatter.set_ylabel("Residual (s)", fontsize=8.0, color="#334155")
    ax_scatter.tick_params(colors="#334155", labelsize=7.5)
    ax_scatter.set_title("Arrival / Fit Residuals", fontsize=9.0, fontweight="bold", color="#0f172a")

    # 2. Correlation matrix or station residual bar summary
    ax_corr.set_facecolor("#ffffff")
    if fit_result is not None and fit_result.correlation_matrix is not None and fit_result.correlation_matrix.size > 0:
        corr = np.asarray(fit_result.correlation_matrix, dtype=np.float64)
        pnames = list(fit_result.parameters.keys())
        if len(pnames) != corr.shape[0]:
            pnames = [f"P{i + 1}" for i in range(corr.shape[0])]
        ax_corr.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax_corr.set_xticks(range(len(pnames)))
        ax_corr.set_yticks(range(len(pnames)))
        ax_corr.set_xticklabels(pnames, fontsize=7.0, rotation=25, ha="right", color="#1e293b")
        ax_corr.set_yticklabels(pnames, fontsize=7.0, color="#1e293b")
        for r_i in range(corr.shape[0]):
            for c_i in range(corr.shape[1]):
                val = corr[r_i, c_i]
                txt_col = "#ffffff" if abs(val) > 0.55 else "#0f172a"
                ax_corr.text(c_i, r_i, f"{val:.2f}", ha="center", va="center", fontsize=7.0, color=txt_col)
        ax_corr.set_title("Parameter Correlation Matrix", fontsize=9.0, fontweight="bold", color="#0f172a")
    elif xs:
        labels = (
            [e.station or e.identifier for e in fit_result.residuals]
            if (fit_result is not None and fit_result.residuals)
            else [f"{tr.station}.{pk.name}" for tr in traces for pk in tr.picks if pk.resid != 0.0 or pk.onset]
        )
        bar_x = np.arange(len(ys))
        ax_corr.bar(bar_x, ys, color=cols, edgecolor="#0f172a", linewidth=0.5, zorder=2)
        ax_corr.axhline(0.0, color="#64748b", ls="--", lw=0.8)
        if len(labels) <= 16:
            ax_corr.set_xticks(bar_x)
            ax_corr.set_xticklabels(labels, fontsize=6.5, rotation=35, ha="right", color="#1e293b")
        ax_corr.set_ylabel("Residual (s)", fontsize=8.0, color="#334155")
        ax_corr.tick_params(axis="y", colors="#334155", labelsize=7.5)
        ax_corr.grid(True, axis="y", color="#e2e8f0", ls=":", lw=0.6)
        ax_corr.set_title("Per-Station Residual Distribution", fontsize=9.0, fontweight="bold", color="#0f172a")
    else:
        ax_corr.text(0.5, 0.5, "No correlation matrix available", ha="center", va="center", color="#64748b", transform=ax_corr.transAxes)
        ax_corr.set_title("Parameter Correlation Matrix", fontsize=9.0, fontweight="bold", color="#0f172a")

    # 3. KPI & Uncertainty Table
    if ax_kpi is not None:
        ax_kpi.axis("off")
        if fit_result is not None:
            kpi_rows = [
                ["Fit Domain", fit_result.fit_type.value, "Converged", "Yes" if fit_result.converged else "No"],
                ["Iterations / DOF", f"{fit_result.iterations} / {fit_result.dof}", "Outliers", str(fit_result.outlier_count)],
                ["RMSE / WRMSE", f"{fit_result.rmse:.4f} / {fit_result.wrmse:.4f}", "MAD / R²", f"{fit_result.mad:.4f} / {fit_result.r_squared:.4f}"],
                ["Reduced Chi²", f"{fit_result.chi2_reduced:.4f}", "Condition κ", f"{fit_result.condition_number:.2f}"],
            ]
            for pname, pinfo in list(fit_result.parameters.items())[:4]:
                ci_lo, ci_hi = pinfo.confidence_interval_95
                unit = f" {pinfo.unit}" if pinfo.unit else ""
                kpi_rows.append(
                    [
                        pname,
                        f"{pinfo.value:.4f}{unit}",
                        "±1σ / 95% CI",
                        f"±{pinfo.standard_error:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]",
                    ]
                )
        else:
            kpi_rows = [
                ["RMSE / WRMSE", f"{params.rmse:.4f} / {params.wrmse:.4f}", "Reduced Chi²", f"{params.chi2_reduced:.4f}"],
                ["Condition κ", f"{params.condition_number:.2f}", "Depth / Type", f"{params.depth:.1f} km (type {params.depth_type})"],
            ]

        tbl = ax_kpi.table(cellText=kpi_rows, loc="center", cellLoc="left", bbox=[0.0, 0.0, 1.0, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.8)
        for (_, c_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if c_idx in (0, 2):
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(fontweight="bold", color="#1e293b")
            else:
                cell.set_facecolor("#ffffff")
                cell.set_text_props(color="#0f172a")


def generate_bulletin_pdf(
    output_path: Union[str, os.PathLike],
    traces: Sequence[Trace],
    params: Optional[AnalysisParamSet] = None,
    fit_result: Optional[FitDiagnosticResult] = None,
    focal_mech: Any = None,
    title: str = "SeismicHandler Observatory Bulletin",
    agency: str = "SHM-Modern Observatory",
    config: Any = None,
) -> str:
    """Generate a multi-page vector PDF observatory bulletin using Matplotlib PdfPages."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = params if params is not None else AnalysisParamSet()
    fm = _extract_focal_mech(focal_mech, fit_result)

    with PdfPages(str(out)) as pdf:
        # Page 1: Header + Metadata Table + Geographic Map (+ Focal Mechanism Beachball if present)
        fig1 = Figure(figsize=(8.27, 11.69), facecolor="#ffffff")
        gs1 = GridSpec(2, 2 if fm else 1, figure=fig1, height_ratios=[0.95, 1.45], hspace=0.28, wspace=0.22)
        ax_hdr = fig1.add_subplot(gs1[0, :])
        render_header_and_metadata(ax_hdr, p, traces, fit_result=fit_result, title=title, agency=agency)

        if fm:
            ax_map = fig1.add_subplot(gs1[1, 0])
            ax_fm = fig1.add_subplot(gs1[1, 1])
            render_geographic_map(ax_map, traces, p, fit_result=fit_result, config=config)
            render_beachball(
                ax_fm,
                fm["strike"],
                fm["dip"],
                fm["rake"],
                polarities=fm.get("polarities"),
            )
        else:
            ax_map = fig1.add_subplot(gs1[1, 0])
            render_geographic_map(ax_map, traces, p, fit_result=fit_result, config=config)

        fig1.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.07)
        pdf.savefig(fig1)

        # Page 2+: Waveform Panels (paginated at 12 traces per page)
        per_page = 12
        chunks = [traces[i : i + per_page] for i in range(0, len(traces), per_page)] or [[]]
        for page_idx, chunk in enumerate(chunks, start=1):
            fig_w = Figure(figsize=(8.27, 11.69), facecolor="#ffffff")
            ax_w = fig_w.add_subplot(111)
            suffix = f" (Page {page_idx}/{len(chunks)})" if len(chunks) > 1 else ""
            render_waveform_panels(ax_w, chunk, p, title=f"Aligned Waveform Panels & Phase Picks{suffix}")
            fig_w.subplots_adjust(left=0.22, right=0.96, top=0.93, bottom=0.07)
            pdf.savefig(fig_w)

        # Final Page: Diagnostic Summary (KPI table, residual scatter, correlation matrix)
        fig_d = Figure(figsize=(8.27, 11.69), facecolor="#ffffff")
        gs_d = GridSpec(2, 2, figure=fig_d, height_ratios=[0.85, 1.35], hspace=0.32, wspace=0.28)
        ax_kpi = fig_d.add_subplot(gs_d[0, :])
        ax_scat = fig_d.add_subplot(gs_d[1, 0])
        ax_corr = fig_d.add_subplot(gs_d[1, 1])
        render_diagnostic_summary(ax_scat, ax_corr, ax_kpi, traces, p, fit_result=fit_result)
        fig_d.suptitle("Fit Diagnostics & Residual Analysis", fontsize=12, fontweight="bold", color="#0f172a", y=0.97)
        fig_d.subplots_adjust(left=0.09, right=0.95, top=0.92, bottom=0.08)
        pdf.savefig(fig_d)

    return str(out)


def generate_bulletin_svg(
    output_path: Union[str, os.PathLike],
    traces: Sequence[Trace],
    params: Optional[AnalysisParamSet] = None,
    fit_result: Optional[FitDiagnosticResult] = None,
    focal_mech: Any = None,
    title: str = "SeismicHandler Observatory Bulletin",
    agency: str = "SHM-Modern Observatory",
    config: Any = None,
) -> str:
    """Generate a single self-contained multi-panel vector SVG observatory bulletin."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = params if params is not None else AnalysisParamSet()
    fm = _extract_focal_mech(focal_mech, fit_result)

    fig = Figure(figsize=(11.5, 16.5), facecolor="#ffffff")
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.85, 1.25, 1.45, 1.15], hspace=0.34, wspace=0.26)

    # Row 0: Header + Event Metadata Table
    ax_hdr = fig.add_subplot(gs[0, :])
    render_header_and_metadata(ax_hdr, p, traces, fit_result=fit_result, title=title, agency=agency)

    # Row 1: Geographic Map + optional Beachball
    if fm:
        ax_map = fig.add_subplot(gs[1, 0])
        ax_fm = fig.add_subplot(gs[1, 1])
        render_geographic_map(ax_map, traces, p, fit_result=fit_result, config=config)
        render_beachball(ax_fm, fm["strike"], fm["dip"], fm["rake"], polarities=fm.get("polarities"))
    else:
        ax_map = fig.add_subplot(gs[1, :])
        render_geographic_map(ax_map, traces, p, fit_result=fit_result, config=config)

    # Row 2: Aligned Waveform Panels
    ax_wav = fig.add_subplot(gs[2, :])
    render_waveform_panels(ax_wav, list(traces[:12]), p)

    # Row 3: Residual Scatter + Correlation Matrix / KPI summary
    ax_scat = fig.add_subplot(gs[3, 0])
    ax_corr = fig.add_subplot(gs[3, 1])
    render_diagnostic_summary(ax_scat, ax_corr, None, traces, p, fit_result=fit_result)

    fig.subplots_adjust(left=0.11, right=0.96, top=0.97, bottom=0.05)
    fig.savefig(str(out), format="svg", bbox_inches="tight")
    return str(out)


def generate_observatory_bulletin(
    output_path: Union[str, os.PathLike],
    traces: Sequence[Trace],
    params: Optional[AnalysisParamSet] = None,
    fit_result: Optional[FitDiagnosticResult] = None,
    focal_mech: Any = None,
    title: str = "SeismicHandler Observatory Bulletin",
    format: Optional[str] = None,
    agency: str = "SHM-Modern Observatory",
    config: Any = None,
) -> str:
    """Generate a publication-grade vector PDF or SVG observatory bulletin."""
    if format is not None:
        fmt = format.strip().lower().lstrip(".")
    else:
        suffix = Path(output_path).suffix.lower().lstrip(".")
        fmt = suffix if suffix in ("pdf", "svg") else "pdf"

    if fmt == "pdf":
        return generate_bulletin_pdf(
            output_path=output_path,
            traces=traces,
            params=params,
            fit_result=fit_result,
            focal_mech=focal_mech,
            title=title,
            agency=agency,
            config=config,
        )
    if fmt == "svg":
        return generate_bulletin_svg(
            output_path=output_path,
            traces=traces,
            params=params,
            fit_result=fit_result,
            focal_mech=focal_mech,
            title=title,
            agency=agency,
            config=config,
        )
    raise ValueError(f"Unsupported report format {fmt!r}; expected 'pdf' or 'svg'.")


if _HAS_PYQT6:

    class BulletinWorker(QThread):
        """Asynchronous QThread worker for generating vector PDF/SVG observatory bulletins."""

        finishedReport = pyqtSignal(str)
        failed = pyqtSignal(str)

        def __init__(
            self,
            output_path: Union[str, os.PathLike],
            traces: Sequence[Trace],
            params: Optional[AnalysisParamSet] = None,
            fit_result: Optional[FitDiagnosticResult] = None,
            focal_mech: Any = None,
            title: str = "SeismicHandler Observatory Bulletin",
            format: Optional[str] = None,
            agency: str = "SHM-Modern Observatory",
            config: Any = None,
            parent: Any = None,
        ) -> None:
            super().__init__(parent)
            self.output_path = output_path
            self.traces = list(traces)
            self.params = params
            self.fit_result = fit_result
            self.focal_mech = focal_mech
            self.title = title
            self.format = format
            self.agency = agency
            self.config = config

        def run(self) -> None:
            try:
                out = generate_observatory_bulletin(
                    output_path=self.output_path,
                    traces=self.traces,
                    params=self.params,
                    fit_result=self.fit_result,
                    focal_mech=self.focal_mech,
                    title=self.title,
                    format=self.format,
                    agency=self.agency,
                    config=self.config,
                )
                self.finishedReport.emit(out)
            except Exception as exc:
                self.failed.emit(str(exc))
