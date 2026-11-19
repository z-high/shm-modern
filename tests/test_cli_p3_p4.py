"""Headless CLI / SHCommandEngine tests for Phase 3 & Phase 4 commands."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import SHMConfig, get_config
from shm_app.core.fit_models import FitDiagnosticResult, FitType
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.processing.spectrum import source_spectrum_model
from shm_app.seismology.focal_mechanism import (
    estimate_takeoff_angle,
    p_wave_radiation_pattern,
)
from shm_app.seismology.traveltime import get_ttt_manager, sphere_add, sphere_diff


def _make_brune_waveform_trace(
    omega0: float = 1200.0,
    fc: float = 2.0,
    t_star: float = 0.05,
    model: str = "brune",
    dt: float = 0.05,
    duration_sec: float = 20.0,
    station: str = "GRA1",
) -> Trace:
    """Synthesize a causal time-domain displacement trace whose FFT matches a Brune/Boatwright spectrum."""
    n = int(round(duration_sec / dt))
    nfft = 1
    while nfft < n:
        nfft <<= 1
    freqs = np.fft.rfftfreq(nfft, d=dt)
    amp = source_spectrum_model(freqs, omega0=omega0, fc=fc, t_star=t_star, model=model)
    amp[0] = 0.0
    # Minimum/causal phase delay of 5 s into the window
    phase = -2.0 * math.pi * freqs * 5.0
    spec = (amp / dt) * np.exp(1j * phase)
    sig = np.fft.irfft(spec, n=nfft)[:n]
    return Trace(
        data=sig,
        delta=dt,
        station=station,
        comp="Z",
        chan1="B",
        chan2="H",
        start_epoch=1700000000.0,
        distance=5.0,
    )


class TestSHCommandEngineP3P4(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = get_config()
        self.engine = SHCommandEngine(config=self.cfg)

    def test_fit_brune_and_boatwright_and_callback(self) -> None:
        emitted: list[FitDiagnosticResult] = []
        self.engine.fit_diagnostic_callback = emitted.append

        tr = _make_brune_waveform_trace(omega0=1500.0, fc=2.2, t_star=0.04, model="brune")
        self.engine.traces.append(tr)

        out = self.engine.execute_line("FIT /BRUNE 1 /F_LO=0.2 /F_HI=8.0")
        self.assertIn("Brune Fit", out)
        self.assertIsNotNone(self.engine.last_fit_diagnostic)
        self.assertEqual(self.engine.last_fit_diagnostic.fit_type, FitType.BRUNE_SPECTRUM)
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(self.engine.params.corner_freq, 2.2, delta=0.5)
        self.assertGreater(self.engine.params.moment_m0, 0.0)
        self.assertGreater(self.engine.params.mag_mw, 0.0)
        self.assertIn("misfit_grid", self.engine.last_fit_diagnostic.overlay_curves)

        # Boatwright model fit
        tr_bw = _make_brune_waveform_trace(omega0=1000.0, fc=1.8, t_star=0.03, model="boatwright")
        self.engine.traces = [tr_bw]
        out_bw = self.engine.execute_line("FIT /BOATWRIGHT 1 /F_LO=0.2 /F_HI=8.0")
        self.assertIn("Boatwright Fit", out_bw)
        self.assertEqual(self.engine.last_fit_diagnostic.context_data["model_type"], "boatwright")
        self.assertAlmostEqual(self.engine.params.corner_freq, 1.8, delta=0.5)

    def test_fit_pol_and_fit_locate(self) -> None:
        # 1. FIT /POL on synthetic rectilinear 3C P-wave arriving from BAZ=120 deg, inc=30 deg
        dt = 0.05
        t = np.arange(400) * dt
        pulse = np.exp(-0.5 * ((t - 5.0) / 0.3) ** 2) * np.sin(2.0 * math.pi * 1.5 * (t - 5.0))
        baz_rad = math.radians(120.0)
        inc_rad = math.radians(30.0)
        # Wave vector pointing FROM source (upgoing ray moves away from source: +Z up, -sin(inc)*cos(baz) N, -sin(inc)*sin(baz) E)
        # Using SHM compute_mdir convention: L = (cos(inc), -sin(inc)*cos(baz), -sin(inc)*sin(baz))
        z_data = pulse * math.cos(inc_rad)
        n_data = -pulse * math.sin(inc_rad) * math.cos(baz_rad)
        e_data = -pulse * math.sin(inc_rad) * math.sin(baz_rad)
        for comp, arr in (("Z", z_data), ("N", n_data), ("E", e_data)):
            self.engine.traces.append(
                Trace(data=arr, delta=dt, station="TRI1", comp=comp, start_epoch=1700000000.0)
            )

        pol_msg = self.engine.execute_line("FIT /POL 3.0 7.0")
        self.assertIn("Polarization Fit", pol_msg)
        self.assertIsNotNone(self.engine.last_fit_diagnostic)
        self.assertEqual(self.engine.last_fit_diagnostic.fit_type, FitType.POLARIZATION_3C)
        self.assertAlmostEqual(self.engine.params.incidence, 30.0, delta=2.0)
        # Rectilinear polarization azimuth has 180-deg ambiguity; check modulo 180
        az_mod = min(
            abs(self.engine.params.azimuth - 120.0),
            abs(((self.engine.params.azimuth + 180.0) % 360.0) - 120.0),
        )
        self.assertLess(az_mod, 3.0)

        # 2. FIT /LOCATE with synthetic travel-time picks from known Greece hypocenter
        self.engine.traces = []
        ttt = get_ttt_manager(self.cfg.inputs_dir)
        true_lat, true_lon, true_depth = 38.5, 22.5, 15.0
        t0 = 1700001000.0
        for sta in ("GRA1", "BFO", "CLZ", "FUR", "MOX", "WET"):
            st = self.cfg.get_station(sta)
            dist, _, _ = sphere_diff(st.lat, st.lon, true_lat, true_lon)
            tt_p = ttt.travel_time("P", dist, true_depth)
            self.assertIsNotNone(tt_p)
            tr = Trace(data=np.zeros(100), delta=0.05, station=sta, comp="Z", start_epoch=t0)
            tr.add_or_replace_pick(
                PhasePick(name="P", onset=format_sh_time(t0 + float(tt_p)), station=sta)
            )
            self.engine.traces.append(tr)

        loc_msg = self.engine.execute_line("FIT /LOCATE /DEPTH=15.0 /LAT=39.0 /LON=22.0")
        self.assertIn("Geiger Location Fit", loc_msg)
        self.assertIsNotNone(self.engine.last_fit_diagnostic)
        self.assertEqual(self.engine.last_fit_diagnostic.fit_type, FitType.HYPOCENTER_GEIGER)
        self.assertAlmostEqual(self.engine.params.lat, true_lat, delta=0.5)
        self.assertAlmostEqual(self.engine.params.lon, true_lon, delta=0.5)
        self.assertIsNotNone(self.engine.params.error_ellipse)

    def test_fit_focal_mechanism(self) -> None:
        # True pure strike-slip fault: strike=30, dip=85, rake=10
        true_s, true_d, true_r = 30.0, 85.0, 10.0
        ev_lat, ev_lon, depth_km = 45.0, 10.0, 15.0
        self.engine.params.lat = ev_lat
        self.engine.params.lon = ev_lon
        self.engine.params.depth = depth_km

        ttt = get_ttt_manager(self.cfg.inputs_dir)
        # Create a well-distributed ring of stations around the epicenter
        for idx, az_deg in enumerate(range(0, 360, 20)):
            dist_deg = 2.0 + (idx % 4) * 3.5
            st_lat, st_lon = sphere_add(ev_lat, ev_lon, dist_deg, float(az_deg))
            ih = estimate_takeoff_angle(dist_deg, depth_km=depth_km, phase="P", ttt=ttt)
            rad = p_wave_radiation_pattern(true_s, true_d, true_r, ih, float(az_deg))
            if abs(rad) < 0.05:
                continue
            pol = 1 if rad > 0 else -1
            sta_code = f"SYN{idx:02d}"
            tr = Trace(
                data=np.zeros(40),
                delta=0.05,
                station=sta_code,
                comp="Z",
                lat=st_lat,
                lon=st_lon,
                start_epoch=1700000000.0,
            )
            tr.extra_headers["station_lat"] = st_lat
            tr.extra_headers["station_lon"] = st_lon
            tr.add_or_replace_pick(
                PhasePick(
                    name="P",
                    onset=format_sh_time(1700000010.0),
                    sign=pol,
                    station=sta_code,
                )
            )
            self.engine.traces.append(tr)

        msg = self.engine.execute_line("FIT /FOCAL")
        self.assertIn("Focal Mechanism Fit", msg)
        self.assertIsNotNone(self.engine.last_focal_mechanism)
        self.assertEqual(self.engine.last_focal_mechanism["n_misfit"], 0)
        self.assertIsNotNone(self.engine.last_fit_diagnostic)
        self.assertIn("focal_mechanism", self.engine.last_fit_diagnostic.context_data)

    def test_calib_array_and_file_application(self) -> None:
        # 1. Load GRF calibration via CALIB /ARRAY
        for sta in ("GRA1", "GRA2", "GRB1"):
            self.engine.traces.append(
                Trace(data=np.zeros(100), delta=0.05, station=sta, comp="Z", start_epoch=1700000000.0)
            )

        msg = self.engine.execute_line("CALIB /ARRAY")
        self.assertIn("Loaded slowness calibration table", msg)
        self.assertIsNotNone(self.engine.calib_table)
        self.assertGreater(len(self.engine.calib_table), 100)

        # 2. Custom delta calibration file with explicit shift: slo=6.0, baz=45.0 -> d_slo=+0.5, d_az=+10.0
        with tempfile.TemporaryDirectory() as td:
            cal_path = os.path.join(td, "custom_calib.dat")
            with open(cal_path, "w", encoding="utf-8") as fh:
                fh.write("# slowness_in azimuth_in d_slowness d_azimuth\n")
                fh.write("6.0  45.0  +0.5  +10.0\n")

            self.engine.params.slowness = 6.0
            self.engine.params.azimuth = 45.0
            self.engine.params.phase = "P"
            out = self.engine.execute_line(f"CALIB {cal_path}")
            self.assertIn("Applied calibration", out)
            self.assertAlmostEqual(self.engine.params.slowness, 6.5, places=3)
            self.assertAlmostEqual(self.engine.params.azimuth, 55.0, places=3)
            self.assertIsNotNone(self.engine.params.residuals)
            self.assertEqual(self.engine.params.residuals["calib_applied"], 1.0)

            # Verify LOCATE /SLOWNESS uses active calib_table
            self.engine.params.slowness = 6.0
            self.engine.params.azimuth = 45.0
            loc_out = self.engine.execute_line("LOCATE /SLOWNESS P")
            self.assertIn("Slowness Location", loc_out)
            self.assertAlmostEqual(self.engine.params.slowness, 6.5, places=3)
            self.assertAlmostEqual(self.engine.params.azimuth, 55.0, places=3)

            # Verify CALIB /OFF disables it
            off_out = self.engine.execute_line("CALIB /OFF")
            self.assertIn("cleared", off_out)
            self.assertIsNone(self.engine.calib_table)

    def test_identify_command(self) -> None:
        ttt = get_ttt_manager(self.cfg.inputs_dir)
        dist_deg = 65.0
        depth_km = 20.0
        t0 = 1700000000.0
        # Compute theoretical PP arrival time & slowness
        tt_pp = ttt.travel_time("PP", dist_deg, depth_km)
        slo_pp = ttt.slowness("PP", dist_deg, depth_km)
        self.assertIsNotNone(tt_pp)
        self.assertIsNotNone(slo_pp)

        tr = Trace(data=np.zeros(100), delta=0.05, station="GRA1", comp="Z", start_epoch=t0, distance=dist_deg)
        tr.add_or_replace_pick(
            PhasePick(
                name="?",
                onset=format_sh_time(t0 + float(tt_pp)),
                slowness=float(slo_pp),
                station="GRA1",
            )
        )
        self.engine.traces.append(tr)
        self.engine.params.origin = format_sh_time(t0)
        self.engine.params.distance = dist_deg
        self.engine.params.depth = depth_km

        msg = self.engine.execute_line("IDENTIFY")
        self.assertIn("PP", msg)
        self.assertEqual(tr.picks[0].name, "PP")
        self.assertGreater(len(self.engine.last_identify_results), 0)
        self.assertEqual(self.engine.last_identify_results[0]["identified_phase"], "PP")

        # Also test direct IDENTIFY /TIME=... mode
        tt_s = ttt.travel_time("S", dist_deg, depth_km)
        slo_s = ttt.slowness("S", dist_deg, depth_km)
        msg_dir = self.engine.execute_line(
            f"IDENTIFY /TIME={float(tt_s):.3f} /SLO={float(slo_s):.3f} /DIST={dist_deg} /DEPTH={depth_km}"
        )
        self.assertIn("S", msg_dir)
        self.assertEqual(self.engine.last_identify_results[0]["phase"], "S")

    def test_autocut_and_filter_autocut(self) -> None:
        dt = 0.05
        n = 1200  # 60 s
        t = np.arange(n) * dt
        sig = np.sin(2.0 * math.pi * 1.5 * t)
        # Add startup step at sample 0 that causes Butterworth filter ringing
        sig[:10] += 500.0
        tr0 = Trace(data=sig.copy(), delta=dt, station="GRA1", comp="Z", start_epoch=1700000000.0)
        tr1 = Trace(data=sig.copy(), delta=dt, station="GRA2", comp="Z", start_epoch=1700000000.0)
        self.engine.traces = [tr0, tr1]

        # 1. BUTTER /AUTOCUT on trace 1
        msg_b = self.engine.execute_line("BUTTER BP 0.5 3.0 4 /LIST=1 /AUTOCUT")
        self.assertIn("with autocut", msg_b)
        self.assertLess(self.engine.traces[0].length, n)
        self.assertGreater(self.engine.traces[0].start_epoch, 1700000000.0)
        # Trace 2 untouched
        self.assertEqual(self.engine.traces[1].length, n)

        # 2. Standalone AUTOCUT on trace 2 after BUTTER
        self.engine.execute_line("BUTTER BP 0.5 3.0 4 /LIST=2")
        self.assertEqual(self.engine.traces[1].length, n)
        msg_ac = self.engine.execute_line("AUTOCUT 2")
        self.assertIn("Autocut trimmed", msg_ac)
        self.assertLess(self.engine.traces[1].length, n)
        self.assertEqual(self.engine.traces[1].length, self.engine.traces[0].length)

    def test_surfmag_command(self) -> None:
        dt = 0.2
        t = np.arange(1500) * dt  # 300 s duration
        # 20.0 s Rayleigh wave packet with 5000 nm zero-to-peak amplitude (5 um)
        env = np.exp(-0.5 * ((t - 150.0) / 35.0) ** 2)
        rayleigh = 5000.0 * env * np.sin(2.0 * math.pi * (1.0 / 20.0) * t)

        dist_deg = 40.0
        tr_z = Trace(data=rayleigh, delta=dt, station="GRA1", comp="Z", start_epoch=1700000000.0, distance=dist_deg)
        tr_n = Trace(data=np.zeros_like(rayleigh), delta=dt, station="GRA1", comp="N", start_epoch=1700000000.0, distance=dist_deg)
        tr_z.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(1700000020.0), station="GRA1"))
        self.engine.traces = [tr_z, tr_n]

        msg = self.engine.execute_line("SURFMAG")
        self.assertIn("Surface Wave Magnitude", msg)
        # Expected Ms = log10(5.0 / 20.0) + 1.66 * log10(40.0) + 3.3 ~= -0.602 + 2.659 + 3.3 = 5.357
        expected_ms = math.log10(5.0 / 20.0) + 1.66 * math.log10(dist_deg) + 3.3
        self.assertAlmostEqual(self.engine.params.mag_ms, expected_ms, delta=0.15)
        self.assertAlmostEqual(tr_z.picks[0].ampl, 5000.0, delta=400.0)
        self.assertAlmostEqual(tr_z.picks[0].period, 20.0, delta=1.5)
        self.assertAlmostEqual(tr_z.picks[0].mag_ms, expected_ms, delta=0.15)
        # N trace should not have had picks created
        self.assertEqual(len(tr_n.picks), 0)

    def test_addfil_and_colour_addfil(self) -> None:
        self.engine.execute_line("CREATE KUEPPER 30.0 0.05 10.0 100.0 2.0")
        self.engine.execute_line("CREATE KUEPPER 30.0 0.05 10.0 100.0 2.0")
        self.engine.execute_line("CREATE KUEPPER 30.0 0.05 10.0 100.0 2.0")
        self.engine.traces[0].filter_name = "NONE"
        self.engine.traces[1].filter_name = "NONE"
        self.engine.traces[2].filter_name = "NONE"
        unselected_orig = self.engine.traces[1].data.copy()

        # Select traces 1 and 3 only
        self.engine.execute_line("SEL 1,3")
        msg = self.engine.execute_line("COLOUR_ADDFIL WWSSN_SP #00aa55")
        self.assertIn("Assigned overlay filter 'WWSSN_SP'", msg)

        # Selected traces 0 and 2 get overlay filter + color + history, while preserving global filter_name
        for idx in (0, 2):
            tr = self.engine.traces[idx]
            self.assertEqual(tr.filter_name, "NONE")
            self.assertEqual(tr.overlay_filter_name, "WWSSN_SP")
            self.assertEqual(tr.overlay_filter_color, "#00aa55")
            self.assertEqual(tr.filter_history, ["WWSSN_SP"])

        # Unselected trace 1 must remain completely untouched
        tr_unsel = self.engine.traces[1]
        self.assertEqual(tr_unsel.overlay_filter_name, "NONE")
        self.assertEqual(tr_unsel.overlay_filter_color, "")
        self.assertEqual(tr_unsel.filter_history, [])
        np.testing.assert_array_equal(tr_unsel.data, unselected_orig)

        # Apply second overlay filter via ADDFIL alias and verify history appends
        self.engine.execute_line("SEL 1")
        self.engine.execute_line("ADDFIL GRSN blue")
        self.assertEqual(self.engine.traces[0].overlay_filter_name, "GRSN")
        self.assertEqual(self.engine.traces[0].overlay_filter_color, "blue")
        self.assertEqual(self.engine.traces[0].filter_history, ["WWSSN_SP", "GRSN"])
        # Trace 2 unchanged
        self.assertEqual(self.engine.traces[2].overlay_filter_name, "WWSSN_SP")

    def test_readsxml_writeqml_readqml_writegeojson_report(self) -> None:
        stationxml_doc = """<?xml version="1.0" encoding="UTF-8"?>
<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.1">
  <Source>SHM-Test</Source>
  <Created>2024-01-01T00:00:00Z</Created>
  <Network code="GR">
    <Station code="SYNT" startDate="2020-01-01T00:00:00Z">
      <Latitude>49.6919</Latitude>
      <Longitude>11.2217</Longitude>
      <Elevation>500.0</Elevation>
      <Site><Name>Synthetic Test Station</Name></Site>
      <Channel code="BHZ" locationCode="" startDate="2020-01-01T00:00:00Z">
        <Latitude>49.6919</Latitude>
        <Longitude>11.2217</Longitude>
        <Elevation>500.0</Elevation>
        <Depth>0.0</Depth>
        <Azimuth>0.0</Azimuth>
        <Dip>-90.0</Dip>
        <SampleRate>20.0</SampleRate>
        <Sensor><Description>STS-2</Description></Sensor>
        <Response>
          <InstrumentSensitivity>
            <Value>1.0e9</Value>
            <Frequency>1.0</Frequency>
            <InputUnits><Name>M/S</Name></InputUnits>
            <OutputUnits><Name>COUNTS</Name></OutputUnits>
          </InstrumentSensitivity>
        </Response>
      </Channel>
    </Station>
  </Network>
</FDSNStationXML>
"""
        with tempfile.TemporaryDirectory() as td:
            # 1. READSXML
            sxml_path = os.path.join(td, "stations.xml")
            with open(sxml_path, "w", encoding="utf-8") as fh:
                fh.write(stationxml_doc)

            sxml_msg = self.engine.execute_line(f"READSXML {sxml_path}")
            self.assertIn("Loaded StationXML", sxml_msg)
            self.assertIn("SYNT", self.engine.config.stations)
            self.assertAlmostEqual(self.engine.config.get_station("SYNT").lat, 49.6919, places=4)

            # Populate workspace trace, params, pick, error ellipse, and focal mechanism
            t0 = 1700000000.0
            tr = Trace(
                data=np.sin(2.0 * math.pi * 1.0 * (np.arange(400) * 0.05)) * 200.0,
                delta=0.05,
                network="GR",
                station="SYNT",
                comp="Z",
                lat=49.6919,
                lon=11.2217,
                distance=12.4,
                azimuth=140.0,
                start_epoch=t0,
            )
            tr.add_or_replace_pick(
                PhasePick(name="P", onset=format_sh_time(t0 + 5.0), sign=1, spec="i", station="SYNT", resid=0.12)
            )
            self.engine.traces.append(tr)
            p = self.engine.params
            p.lat = 38.2
            p.lon = 22.1
            p.depth = 14.5
            p.origin = format_sh_time(t0 - 150.0)
            p.mag_mb = 5.1
            p.mag_ms = 4.9
            p.error_ellipse = (8.5, 4.2, 35.0)
            p.region_name = "Greece"
            self.engine.last_focal_mechanism = {"strike": 110.0, "dip": 65.0, "rake": -40.0}

            # 2. WRITEQML & READQML roundtrip
            qml_path = os.path.join(td, "event.qml")
            wq_msg = self.engine.execute_line(f"WRITEQML {qml_path}")
            self.assertIn("Wrote QuakeML 1.2", wq_msg)
            self.assertTrue(os.path.isfile(qml_path))
            root = ET.parse(qml_path).getroot()
            self.assertIn("quakeml", root.tag.lower())

            # Mutate params and reload via READQML
            self.engine.params.lat = 0.0
            self.engine.params.lon = 0.0
            self.engine.last_focal_mechanism = None
            rq_msg = self.engine.execute_line(f"READQML {qml_path}")
            self.assertIn("Read QuakeML 1.2", rq_msg)
            self.assertAlmostEqual(self.engine.params.lat, 38.2, places=3)
            self.assertAlmostEqual(self.engine.params.lon, 22.1, places=3)
            self.assertAlmostEqual(self.engine.params.depth, 14.5, places=2)
            self.assertIsNotNone(self.engine.last_focal_mechanism)
            self.assertAlmostEqual(self.engine.last_focal_mechanism["strike"], 110.0, places=1)

            # 3. WRITEGEOJSON
            gj_path = os.path.join(td, "event.geojson")
            gj_msg = self.engine.execute_line(f"WRITEGEOJSON {gj_path}")
            self.assertIn("Wrote GeoJSON", gj_msg)
            self.assertTrue(os.path.isfile(gj_path))
            with open(gj_path, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
            self.assertEqual(gj["type"], "FeatureCollection")
            geom_types = {feat["geometry"]["type"] for feat in gj["features"]}
            self.assertIn("Point", geom_types)
            self.assertIn("Polygon", geom_types)

            # 4. REPORT (SVG and PDF)
            svg_path = os.path.join(td, "bulletin.svg")
            pdf_path = os.path.join(td, "bulletin.pdf")
            svg_msg = self.engine.execute_line(f"REPORT {svg_path}")
            pdf_msg = self.engine.execute_line(f"REPORT {pdf_path}")
            self.assertIn("Generated observatory bulletin report", svg_msg)
            self.assertIn("Generated observatory bulletin report", pdf_msg)
            self.assertTrue(os.path.isfile(svg_path))
            self.assertTrue(os.path.isfile(pdf_path))
            self.assertGreater(os.path.getsize(svg_path), 1000)
            self.assertGreater(os.path.getsize(pdf_path), 1000)

    def test_calib_with_plane_wave_locate_and_fk(self) -> None:
        gse_file = os.path.join(SHM_DEFAULT_ROOT, "data-examples", "gse", "GRF_031102_0225.GSE")
        self.engine.execute_line(f"READGSE {gse_file}")
        self.engine.execute_line("BUTTER BP 0.7 2.5 3")
        self.engine.execute_line("PICK ALL")
        self.engine.execute_line("CALIB /ARRAY")

        fk_msg = self.engine.execute_line("FK 0.7 2.5 12.0")
        self.assertIn("[calibrated]", fk_msg)
        self.assertTrue(self.engine.last_fk_result["calib_applied"])

        loc_msg = self.engine.execute_line("LOCATE P")
        self.assertIn("[calibrated]", loc_msg)
        self.assertIsNotNone(self.engine.params.residuals)
        self.assertEqual(self.engine.params.residuals.get("calib_applied"), 1.0)
        p_picks = [tr.get_pick("P") for tr in self.engine.traces if tr.get_pick("P")]
        self.assertTrue(any(pk.l_slowness > 0.0 for pk in p_picks))

    def test_filter_autocut_addfil_clear_and_error_paths(self) -> None:
        self.engine.execute_line("CREATE KUEPPER 40.0 0.05 15.0 100.0 2.0")
        orig_len = self.engine.traces[0].length
        msg_f = self.engine.execute_line("FILTER WWSSN_SP /AUTOCUT")
        self.assertIn("with autocut", msg_f)
        self.assertLess(self.engine.traces[0].length, orig_len)

        # ADDFIL without selection raises SH Error
        self.engine.execute_line("SEL NONE")
        self.assertIn("SH Error in ADDFIL", self.engine.execute_line("ADDFIL WWSSN_SP"))

        # ADDFIL /CLEAR clears overlay metadata on selected trace
        self.engine.execute_line("SEL 1")
        self.engine.execute_line("ADDFIL WWSSN_SP red")
        self.assertEqual(self.engine.traces[0].overlay_filter_name, "WWSSN_SP")
        self.engine.execute_line("ADDFIL /CLEAR")
        self.assertEqual(self.engine.traces[0].overlay_filter_name, "NONE")
        self.assertEqual(self.engine.traces[0].overlay_filter_color, "")

        # Missing args error handling
        for cmd in ("FIT", "READSXML", "WRITEQML", "READQML", "WRITEGEOJSON", "REPORT"):
            self.assertIn(f"SH Error in {cmd}", self.engine.execute_line(cmd))

    def test_sh_engine_has_zero_forbidden_imports(self) -> None:
        import ast

        engine_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "shm_app",
            "interpreter",
            "sh_engine.py",
        )
        with open(engine_path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=engine_path)

        forbidden = {"PyQt6", "scipy", "obspy", "reportlab", "lxml"}
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in forbidden:
                        found.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".", 1)[0]
                if top in forbidden:
                    found.append(node.module)
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()

