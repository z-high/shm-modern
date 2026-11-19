"""Unit tests for M-20 publication-grade vector PDF and SVG observatory reporting."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/shm_mpl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from shm_app.core.fit_models import (
    ConfidenceEllipse,
    FitDiagnosticResult,
    FitType,
    ParameterUncertainty,
    ResidualEntry,
)
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.gui.reporting import (
    generate_bulletin_pdf,
    generate_bulletin_svg,
    generate_observatory_bulletin,
)


@dataclass
class _DummyFocalMech:
    strike: float
    dip: float
    rake: float
    polarities: list


class TestVectorReporting(unittest.TestCase):
    """Tests for headless pure-Matplotlib vector PDF and SVG bulletin generation."""

    def _build_fixture(self, n_stations: int = 4):
        t0 = 1700000000.0
        dt = 0.05
        n_pts = 800
        t_rel = np.arange(n_pts) * dt
        stations = [
            ("GRA1", 49.6919, 11.2217, 14.1, 42.0),
            ("GRB1", 49.3910, 11.6520, 14.3, 44.0),
            ("GRC1", 48.9960, 11.5210, 14.5, 45.0),
            ("BFO", 48.3301, 8.3296, 15.8, 52.0),
            ("WET", 49.1440, 12.8780, 13.7, 39.0),
            ("FUR", 48.1629, 11.2752, 14.9, 47.0),
            ("CLZ", 51.8430, 10.3740, 16.2, 33.0),
            ("BUG", 51.4410, 7.2690, 17.0, 56.0),
            ("TNS", 50.2220, 8.4470, 15.6, 51.0),
            ("STU", 48.7719, 9.1950, 15.2, 48.0),
            ("BRG", 50.8732, 13.9428, 14.0, 31.0),
            ("CLL", 51.3077, 13.0026, 14.6, 30.0),
            ("MOX", 50.6447, 11.6156, 14.8, 37.0),
            ("RUE", 52.4759, 13.7800, 15.4, 24.0),
        ]

        traces = []
        residuals = []
        for idx in range(n_stations):
            sta, slat, slon, dist, baz = stations[idx % len(stations)]
            sta_name = sta if idx < len(stations) else f"{sta}_{idx}"
            # Synthetic P + S wavepacket
            p_arr = 10.0 + 0.5 * idx
            s_arr = 22.0 + 0.8 * idx
            sig = (
                100.0 * np.exp(-((t_rel - p_arr) ** 2) / 0.3) * np.sin(2.0 * np.pi * 1.8 * (t_rel - p_arr))
                + 180.0 * np.exp(-((t_rel - s_arr) ** 2) / 0.8) * np.sin(2.0 * np.pi * 0.9 * (t_rel - s_arr))
            )
            pk_p = PhasePick(
                name="P",
                onset=format_sh_time(t0 + p_arr),
                spec="i",
                sign=1 if idx % 2 == 0 else -1,
                quality=2 if idx % 3 else 4,
                resid=(-0.12 if idx != 2 else 0.85),
                station=sta_name,
            )
            pk_s = PhasePick(
                name="S",
                onset=format_sh_time(t0 + s_arr),
                spec="e",
                quality=4,
                resid=0.18,
                station=sta_name,
            )
            tr = Trace(
                data=sig,
                delta=dt,
                station=sta_name,
                comp="Z",
                lat=slat,
                lon=slon,
                distance=dist,
                azimuth=baz,
                start_epoch=t0,
                filter_name="WWSSN_SP",
                overlay_filter_name="BP_1_4HZ" if idx == 0 else "NONE",
                picks=[pk_p, pk_s],
                theo_picks={"P": t0 + p_arr + 0.12, "pP": t0 + p_arr + 5.2},
            )
            traces.append(tr)
            residuals.append(
                ResidualEntry(
                    identifier=f"{sta_name}.BHZ.P",
                    observed=t0 + p_arr,
                    calculated=t0 + p_arr + 0.12,
                    residual=pk_p.resid,
                    normalized_residual=pk_p.resid / pk_p.sigma_t(dt),
                    weight=1.0 if idx != 2 else 0.25,
                    uncertainty=pk_p.sigma_t(dt),
                    distance_deg=dist,
                    azimuth_deg=baz,
                    station=sta_name,
                    phase="P",
                    channel="BHZ",
                    is_outlier=(idx == 2),
                )
            )

        params = AnalysisParamSet(
            distance=14.5,
            azimuth=42.5,
            slowness=13.2,
            depth=15.0,
            d_depth=1.8,
            origin=format_sh_time(t0 - 195.0),
            lat=38.42,
            d_lat=0.04,
            lon=21.75,
            d_lon=0.06,
            phase="P",
            mag_ml=5.2,
            mag_mb=5.5,
            mag_ms=5.1,
            mag_mw=5.4,
            event_id=2026111801,
            event_type="r",
            region_id=364,
            region_name="GREECE",
            error_ellipse=(12.5, 6.2, 40.0),
            condition_number=3.4,
            chi2_reduced=1.12,
            rmse=0.24,
            wrmse=0.18,
        )

        corr = np.array(
            [
                [1.0, -0.35, 0.22],
                [-0.35, 1.0, -0.18],
                [0.22, -0.18, 1.0],
            ],
            dtype=np.float64,
        )
        fit_result = FitDiagnosticResult(
            fit_type=FitType.HYPOCENTER_GEIGER,
            title="Geiger Network Location",
            converged=True,
            iterations=7,
            dof=max(1, n_stations - 3),
            rmse=0.24,
            wrmse=0.18,
            nrmse=0.05,
            mad=0.14,
            chi2_reduced=1.12,
            r_squared=0.96,
            condition_number=3.4,
            parameters={
                "Latitude": ParameterUncertainty("Latitude", 38.42, 0.04, (38.34, 38.50), "deg"),
                "Longitude": ParameterUncertainty("Longitude", 21.75, 0.06, (21.63, 21.87), "deg"),
                "Depth": ParameterUncertainty("Depth", 15.0, 1.8, (11.4, 18.6), "km"),
            },
            correlation_matrix=corr,
            confidence_ellipse=ConfidenceEllipse(
                semi_major_km=12.5,
                semi_minor_km=6.2,
                strike_deg=40.0,
                center_x=21.75,
                center_y=38.42,
            ),
            residuals=residuals,
            outlier_count=1,
        )

        focal_mech = {
            "strike": 125.0,
            "dip": 55.0,
            "rake": -80.0,
            "polarities": [(42.0, 28.0, 1), (135.0, 40.0, -1), (225.0, 35.0, 1), (315.0, 45.0, -1)],
        }
        return traces, params, fit_result, focal_mech

    def test_generate_multipage_vector_pdf(self) -> None:
        traces, params, fit_result, focal_mech = self._build_fixture(n_stations=14)
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "observatory_bulletin.pdf"
            out = generate_observatory_bulletin(
                pdf_path,
                traces=traces,
                params=params,
                fit_result=fit_result,
                focal_mech=focal_mech,
                title="BGR / SHM Observatory Test Bulletin",
            )
            self.assertEqual(Path(out), pdf_path)
            self.assertTrue(pdf_path.is_file())
            raw = pdf_path.read_bytes()
            self.assertTrue(raw.startswith(b"%PDF-"))
            self.assertGreater(len(raw), 15000)
            self.assertIn(b"%%EOF", raw)
            # 14 traces -> Page 1 (map+fm) + 2 waveform pages (12+2) + Page 4 (diagnostics) = 4 pages
            page_markers = raw.count(b"/Type /Page\n") + raw.count(b"/Type /Page ")
            self.assertGreaterEqual(page_markers, 4)

    def test_generate_vector_svg_with_all_elements(self) -> None:
        traces, params, fit_result, _ = self._build_fixture(n_stations=5)
        fm_obj = _DummyFocalMech(
            strike=210.0,
            dip=68.0,
            rake=15.0,
            polarities=[(30.0, 25.0, 1), (120.0, 35.0, -1)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = Path(tmpdir) / "bulletin.svg"
            out = generate_observatory_bulletin(
                svg_path,
                traces=traces,
                params=params,
                fit_result=fit_result,
                focal_mech=fm_obj,
                format="svg",
            )
            self.assertEqual(Path(out), svg_path)
            svg_text = svg_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg_text)
            self.assertIn("</svg>", svg_text)
            self.assertGreater(len(svg_text), 10000)
            # Verify well-formed XML
            root = ET.fromstring(svg_text)
            self.assertTrue(root.tag.endswith("svg"))

    def test_generate_bulletin_edge_cases_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty traces, no fit_result, no focal_mech
            pdf_empty = Path(tmpdir) / "empty_report"
            out_pdf = generate_bulletin_pdf(pdf_empty, traces=[], params=None)
            self.assertTrue(Path(out_pdf).read_bytes().startswith(b"%PDF-"))

            svg_empty = Path(tmpdir) / "empty.svg"
            out_svg = generate_bulletin_svg(svg_empty, traces=[], params=None)
            self.assertIn("<svg", Path(out_svg).read_text(encoding="utf-8"))

            # Unsupported explicit format raises ValueError
            with self.assertRaises(ValueError):
                generate_observatory_bulletin(Path(tmpdir) / "bad.out", traces=[], format="docx")


if __name__ == "__main__":
    unittest.main()
