"""Unit tests for M-19: QuakeML 1.2 XML & GeoJSON RFC 7946 export/import validation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.geojson import export_geojson, parse_geojson, write_geojson
from shm_app.io.quakeml import BED_NS, QML_NS, export_quakeml, parse_quakeml, write_quakeml
from shm_app.seismology.traveltime import sphere_diff


def _sample_event_data():
    t0 = parse_sh_time("02-NOV-2003_02:15:10.500")
    params = AnalysisParamSet(
        event_id=20031102,
        origin="02-NOV-2003_02:15:10.500",
        lat=35.25,
        lon=139.85,
        d_lat=4.5,
        d_lon=3.2,
        depth=42.5,
        d_depth=3.1,
        depth_type=3,
        distance=81.4,
        azimuth=41.2,
        slowness=5.12,
        mag_ml=5.4,
        mag_mb=5.8,
        mag_ms=5.6,
        mag_mw=5.9,
        mags_count=2,
        source="SHM-Geiger",
        region_name="Near East Coast of Honshu",
        event_type="t",
        error_ellipse=(12.5, 6.2, 35.0),
        chi2_reduced=1.15,
        rmse=0.32,
        wrmse=0.28,
        condition_number=4.8,
        residuals={"GRA1": -0.25, "GRB1": 0.38},
    )

    tr1 = Trace(
        data=np.zeros(100, dtype=np.float64),
        delta=0.05,
        station="GRA1",
        comp="Z",
        chan1="B",
        chan2="H",
        network="GR",
        locid="00",
        start_epoch=t0 + 700.0,
        lat=49.6919,
        lon=11.2217,
        distance=81.4,
        azimuth=325.0,
        picks=[
            PhasePick(
                name="P",
                onset="02-NOV-2003_02:27:25.120",
                spec="i",
                sign=1,
                source="m",
                quality=2,
                weight=4,
                resid=-0.25,
                station="GRA1",
            )
        ],
    )
    tr2 = Trace(
        data=np.zeros(100, dtype=np.float64),
        delta=0.05,
        station="GRB1",
        comp="Z",
        chan1="B",
        chan2="H",
        network="GR",
        locid="00",
        start_epoch=t0 + 700.0,
        lat=49.3900,
        lon=11.6500,
        distance=81.8,
        azimuth=326.5,
        picks=[
            PhasePick(
                name="S",
                onset="02-NOV-2003_02:37:40.200",
                spec="e",
                sign=-1,
                source="a",
                quality=4,
                weight=2,
                resid=0.38,
                station="GRB1",
            )
        ],
    )
    fm = {
        "nodal_plane1": {"strike": 120.0, "dip": 45.0, "rake": -80.0},
        "nodal_plane2": {"strike": 286.0, "dip": 46.0, "rake": -100.0},
    }
    return [tr1, tr2], params, fm


def _shoelace_signed_area(ring: list) -> float:
    area2 = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[i + 1][0], ring[i + 1][1]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * area2


class TestQuakeMLAndGeoJSON(unittest.TestCase):
    def test_quakeml_12_structure_namespaces_and_roundtrip(self) -> None:
        traces, params, fm = _sample_event_data()
        xml_str = export_quakeml(traces, params, public_id_prefix="smi:test-obs", focal_mechanism=fm)

        self.assertIn(QML_NS, xml_str)
        self.assertIn(BED_NS, xml_str)

        root = ET.fromstring(xml_str)
        self.assertEqual(root.tag, f"{{{QML_NS}}}quakeml")
        ep = root.find(f"{{{BED_NS}}}eventParameters")
        self.assertIsNotNone(ep)
        assert ep is not None
        ev = ep.find(f"{{{BED_NS}}}event")
        self.assertIsNotNone(ev)
        assert ev is not None

        # Verify mandatory QuakeML 1.2 elements exist
        org = ev.find(f"{{{BED_NS}}}origin")
        self.assertIsNotNone(org)
        assert org is not None
        self.assertIsNotNone(org.find(f"{{{BED_NS}}}originUncertainty"))
        ou = org.find(f"{{{BED_NS}}}originUncertainty")
        assert ou is not None
        self.assertIsNotNone(ou.find(f"{{{BED_NS}}}confidenceEllipsoid"))
        self.assertEqual(len(org.findall(f"{{{BED_NS}}}arrival")), 2)
        self.assertGreaterEqual(len(ev.findall(f"{{{BED_NS}}}magnitude")), 4)
        self.assertEqual(len(ev.findall(f"{{{BED_NS}}}pick")), 2)
        self.assertIsNotNone(ev.find(f"{{{BED_NS}}}focalMechanism"))

        # Roundtrip parse check
        with tempfile.TemporaryDirectory() as tmpdir:
            qml_path = os.path.join(tmpdir, "event.xml")
            write_quakeml(qml_path, traces, params, public_id_prefix="smi:test-obs", focal_mechanism=fm)
            parsed = parse_quakeml(qml_path)

        self.assertEqual(parsed["schema_version"], "1.2")
        self.assertEqual(len(parsed["events"]), 1)
        p_out = parsed["params"]
        self.assertAlmostEqual(p_out.lat, 35.25, places=4)
        self.assertAlmostEqual(p_out.lon, 139.85, places=4)
        self.assertAlmostEqual(p_out.depth, 42.5, places=3)
        self.assertAlmostEqual(p_out.d_depth, 3.1, places=3)
        self.assertAlmostEqual(p_out.mag_mw, 5.9, places=3)
        self.assertAlmostEqual(p_out.mag_mb, 5.8, places=3)
        self.assertAlmostEqual(p_out.mag_ms, 5.6, places=3)
        self.assertAlmostEqual(p_out.mag_ml, 5.4, places=3)
        self.assertIsNotNone(p_out.error_ellipse)
        assert p_out.error_ellipse is not None
        self.assertAlmostEqual(p_out.error_ellipse[0], 12.5, places=2)
        self.assertAlmostEqual(p_out.error_ellipse[1], 6.2, places=2)
        self.assertAlmostEqual(p_out.error_ellipse[2], 35.0, places=2)

        self.assertEqual(len(parsed["picks"]), 2)
        self.assertEqual(len(parsed["arrivals"]), 2)
        self.assertEqual(parsed["picks"][0]["polarity"], "positive")
        self.assertEqual(parsed["picks"][1]["polarity"], "negative")
        self.assertAlmostEqual(parsed["arrivals"][0]["time_residual"], -0.25, places=3)
        self.assertAlmostEqual(parsed["arrivals"][1]["time_residual"], 0.38, places=3)

        self.assertEqual(len(parsed["focal_mechanisms"]), 1)
        np1 = parsed["focal_mechanisms"][0]["nodalPlane1"]
        self.assertAlmostEqual(np1["strike"], 120.0, places=2)
        self.assertAlmostEqual(np1["dip"], 45.0, places=2)
        self.assertAlmostEqual(np1["rake"], -80.0, places=2)

    def test_parse_legacy_quakeml_example(self) -> None:
        legacy_qml = "/home/zhigh/Documents/Antigravity/SHM/sh/util/quakeml/example.xml"
        if not os.path.isfile(legacy_qml):
            self.skipTest("Bundled legacy QuakeML example.xml not found")
        parsed = parse_quakeml(legacy_qml)
        self.assertEqual(len(parsed["events"]), 1)
        self.assertGreater(len(parsed["events"][0]["origins"]), 1)
        self.assertAlmostEqual(parsed["origin"]["latitude"], 38.948, places=3)
        self.assertAlmostEqual(parsed["origin"]["longitude"], 70.544, places=3)

    def test_geojson_rfc7946_export_and_roundtrip(self) -> None:
        traces, params, fm = _sample_event_data()
        gj = export_geojson(traces, params, include_stations=True, include_ellipse=True, focal_mechanism=fm)

        self.assertEqual(gj["type"], "FeatureCollection")
        self.assertNotIn("crs", gj)  # RFC 7946 prohibits non-standard crs member
        # 1 epicenter + 1 confidence ellipse + 2 stations = 4 features
        self.assertEqual(len(gj["features"]), 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            gj_path = os.path.join(tmpdir, "event.geojson")
            write_geojson(gj_path, traces, params, focal_mechanism=fm)
            with open(gj_path, "r", encoding="utf-8") as f:
                raw_txt = f.read()
            self.assertEqual(json.loads(raw_txt)["type"], "FeatureCollection")
            parsed = parse_geojson(gj_path)

        epi = parsed["epicenter"]
        self.assertIsNotNone(epi)
        assert epi is not None
        self.assertEqual(epi["geometry"]["type"], "Point")
        self.assertEqual(epi["geometry"]["coordinates"], [139.85, 35.25])
        self.assertAlmostEqual(epi["properties"]["mag_mw"], 5.9)
        self.assertAlmostEqual(epi["properties"]["depth_km"], 42.5)

        ell = parsed["confidence_ellipse"]
        self.assertIsNotNone(ell)
        assert ell is not None
        self.assertEqual(ell["geometry"]["type"], "Polygon")
        rings = ell["geometry"]["coordinates"]
        self.assertEqual(len(rings), 1)
        ring = rings[0]
        self.assertGreaterEqual(len(ring), 4)
        self.assertEqual(ring[0], ring[-1])  # Closed ring
        self.assertGreater(_shoelace_signed_area(ring), 0.0)  # Counterclockwise RFC 7946 §3.1.6

        # Verify projected ellipse semi-major/semi-minor distances from epicenter match 12.5 km and 6.2 km
        dists_km = []
        for pt_lon, pt_lat in ring[:-1]:
            d_deg, _, _ = sphere_diff(params.lat, params.lon, pt_lat, pt_lon)
            dists_km.append(d_deg * 111.1949266)
        self.assertAlmostEqual(max(dists_km), 12.5, delta=0.15)
        self.assertAlmostEqual(min(dists_km), 6.2, delta=0.15)

        stations = parsed["stations"]
        self.assertEqual(len(stations), 2)
        by_code = {s["properties"]["station"]: s for s in stations}
        self.assertEqual(by_code["GRA1"]["geometry"]["coordinates"], [11.2217, 49.6919])
        self.assertAlmostEqual(by_code["GRA1"]["properties"]["residual_s"], -0.25)
        self.assertAlmostEqual(by_code["GRA1"]["properties"]["distance_deg"], 81.4)
        self.assertAlmostEqual(by_code["GRB1"]["properties"]["residual_s"], 0.38)

        # Verify optional exclusions
        gj_min = export_geojson(traces, params, include_stations=False, include_ellipse=False)
        self.assertEqual(len(gj_min["features"]), 1)
        self.assertEqual(gj_min["features"][0]["properties"]["feature_type"], "epicenter")


if __name__ == "__main__":
    unittest.main()
