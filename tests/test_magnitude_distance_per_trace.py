"""Unit tests for per-station magnitude distance calculations in interpreter/sh_engine.py (M-02)."""

import unittest
import numpy as np

from shm_app.core.config_station import get_config
from shm_app.core.models import PhasePick, Trace
from shm_app.core.time_utils import format_sh_time
from shm_app.interpreter.sh_engine import SHCommandEngine
from shm_app.seismology.traveltime import sphere_diff


class TestMagnitudeDistancePerTrace(unittest.TestCase):
    """Test individual station epicentral distance calculations and magnitude standard deviations."""

    def setUp(self) -> None:
        self.cfg = get_config()
        self.engine = SHCommandEngine(config=self.cfg)

    def test_magnitude_per_station_distance(self) -> None:
        """Verify _cmd_mag calculates distinct great-circle distances per station and computes stddev."""
        # Set earthquake hypocenter (e.g. Greece: 38.0 N, 22.0 E, depth 15 km)
        p = self.engine.current_params
        p.lat = 38.0
        p.lon = 22.0
        p.depth = 15.0

        # Create synthetic traces with realistic sinusoids at 3 different German stations:
        # GRA1 (lat ~49.69, lon ~11.22), BFO (lat ~48.33, lon ~8.31), CLZ (lat ~51.81, lon ~10.33)
        stations = ["GRA1", "BFO", "CLZ"]
        ref_epoch = 1000000.0
        dt = 0.05
        t_arr = np.arange(400) * dt

        for st_code in stations:
            # Synthetic 1 Hz sine burst representing a P arrival
            sig = np.sin(2.0 * np.pi * 1.0 * t_arr) * 5000.0
            tr = Trace(
                data=sig,
                delta=dt,
                station=st_code,
                comp="Z",
                chan1="B",
                chan2="H",
                start_epoch=ref_epoch,
                calib=1.0,
            )
            tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + 2.0), station=st_code))
            self.engine.traces.append(tr)

        # Run MAG command
        msg = self.engine.execute_line("MAG P")
        self.assertIn("Mean Magnitudes", msg)

        # 1. Verify each trace has received its own great-circle distance
        distances = []
        for tr in self.engine.traces:
            st = self.cfg.get_station(tr.station)
            expected_dist, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
            self.assertAlmostEqual(tr.distance, expected_dist, places=2)
            distances.append(tr.distance)

        # Distances must differ between different stations
        self.assertEqual(len(distances), 3)
        self.assertNotEqual(distances[0], distances[1])
        self.assertNotEqual(distances[1], distances[2])

        # 2. Verify mags_count and standard deviations are computed
        self.assertEqual(p.mags_count, 3)
        self.assertGreater(p.mag_mb, 0.0)
        self.assertGreaterEqual(p.d_mb, 0.0)
        self.assertGreater(p.mag_ml, 0.0)
        self.assertGreaterEqual(p.d_ml, 0.0)

    def test_magnitude_fallback_global_distance(self) -> None:
        """Verify fallback to parameter box distance when event epicenter is unset (lat=0, lon=0)."""
        p = self.engine.current_params
        p.lat = 0.0
        p.lon = 0.0
        p.distance = 42.5
        p.depth = 20.0

        ref_epoch = 1000000.0
        sig = np.sin(2.0 * np.pi * 1.0 * (np.arange(300) * 0.05)) * 3000.0
        tr = Trace(
            data=sig,
            delta=0.05,
            station="UNKNOWN_STAT",
            comp="Z",
            start_epoch=ref_epoch,
        )
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + 1.0), station="UNKNOWN_STAT"))
        self.engine.traces.append(tr)

        msg = self.engine.execute_line("MAG P")
        self.assertIn("Dist=42.5 deg", msg)
        self.assertEqual(p.mags_count, 1)
        # Single station: stddev must be 0.0
        self.assertEqual(p.d_mb, 0.0)
        self.assertEqual(p.d_ml, 0.0)

    def test_measure_amplitude_magnitudes_uses_station_distance(self) -> None:
        """Verify measure_amplitude_magnitudes assigns and uses per-station distance."""
        p = self.engine.current_params
        p.lat = 45.0
        p.lon = 15.0
        p.depth = 10.0

        ref_epoch = 1000000.0
        sig = np.sin(2.0 * np.pi * 1.0 * (np.arange(400) * 0.05)) * 4000.0
        tr = Trace(
            data=sig,
            delta=0.05,
            station="FUR",
            comp="Z",
            start_epoch=ref_epoch,
            distance=0.0,
        )
        tr.add_or_replace_pick(PhasePick(name="P", onset=format_sh_time(ref_epoch + 1.0), station="FUR"))
        self.engine.traces.append(tr)

        res = self.engine.measure_amplitude_magnitudes(0, 1.0, 5.0, prefer_phase="P")
        self.assertIsNotNone(res)
        self.assertIn("mb", res)
        self.assertIn("ml", res)

        # Verify trace distance was updated
        st = self.cfg.get_station("FUR")
        expected_dist, _, _ = sphere_diff(st.lat, st.lon, p.lat, p.lon)
        self.assertAlmostEqual(tr.distance, expected_dist, places=2)


if __name__ == "__main__":
    unittest.main()
