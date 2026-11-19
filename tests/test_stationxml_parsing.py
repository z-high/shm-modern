"""Unit tests for M-12: FDSN StationXML 1.x parser, frequency response evaluator, and SHMConfig integration."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import numpy as np

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import SHMConfig
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.stationxml import eval_stationxml_response, parse_iso_epoch, parse_stationxml


def _make_sample_stationxml(namespace: str = "http://www.fdsn.org/xml/station/1") -> str:
    ns_attr = f' xmlns="{namespace}"' if namespace else ""
    # Simple 1-pole velocity sensor normalized at 1 Hz:
    # H_pz(s) = A0 * s / (s - p0), with p0 = -2*pi*0.00833 (120 s corner) -> at 1 Hz, |s/(s-p0)| ~ 1
    p0_re = -2.0 * math.pi * (1.0 / 120.0)
    # Exact normalization factor A0 at f = 1.0 Hz:
    w_norm = 2.0 * math.pi * 1.0
    a0 = abs((1j * w_norm - p0_re) / (1j * w_norm))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<FDSNStationXML{ns_attr} schemaVersion="1.1">
  <Source>SHM-Modern Test</Source>
  <Sender>BGR</Sender>
  <Created>2023-05-01T12:00:00Z</Created>
  <Network code="GR" startDate="1990-01-01T00:00:00Z">
    <Station code="GRA1" startDate="1990-01-01T00:00:00Z">
      <Latitude>49.6919</Latitude>
      <Longitude>11.2217</Longitude>
      <Elevation>500.0</Elevation>
      <Site><Name>Graefenberg A1</Name></Site>
      <Channel code="BHZ" locationCode="00" startDate="1995-01-01T00:00:00Z" endDate="2010-01-01T00:00:00Z">
        <Latitude>49.6919</Latitude>
        <Longitude>11.2217</Longitude>
        <Elevation>500.0</Elevation>
        <Depth>0.0</Depth>
        <Azimuth>0.0</Azimuth>
        <Dip>-90.0</Dip>
        <SampleRate>20.0</SampleRate>
        <Sensor><Type>STS-2</Type></Sensor>
        <Response>
          <InstrumentSensitivity>
            <Value>600000000.0</Value>
            <Frequency>1.0</Frequency>
            <InputUnits><Name>M/S</Name><Description>Velocity in Meters per Second</Description></InputUnits>
            <OutputUnits><Name>COUNTS</Name><Description>Digital Counts</Description></OutputUnits>
          </InstrumentSensitivity>
          <Stage number="1">
            <PolesZeros>
              <InputUnits><Name>M/S</Name></InputUnits>
              <OutputUnits><Name>V</Name></OutputUnits>
              <PzTransferFunctionType>LAPLACE (RADIANS/SECOND)</PzTransferFunctionType>
              <NormalizationFactor>{a0:.10f}</NormalizationFactor>
              <NormalizationFrequency>1.0</NormalizationFrequency>
              <Zero number="0"><Real>0.0</Real><Imaginary>0.0</Imaginary></Zero>
              <Pole number="0"><Real>{p0_re:.10f}</Real><Imaginary>0.0</Imaginary></Pole>
            </PolesZeros>
            <StageGain><Value>1500.0</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
          <Stage number="2">
            <Coefficients>
              <InputUnits><Name>V</Name></InputUnits>
              <OutputUnits><Name>COUNTS</Name></OutputUnits>
              <CfTransferFunctionType>DIGITAL</CfTransferFunctionType>
              <Numerator>0.25</Numerator>
              <Numerator>0.5</Numerator>
              <Numerator>0.25</Numerator>
            </Coefficients>
            <Decimation>
              <InputSampleRate>20.0</InputSampleRate>
              <Factor>1</Factor>
              <Offset>0</Offset>
              <Delay>0.0</Delay>
              <Correction>0.0</Correction>
            </Decimation>
            <StageGain><Value>400000.0</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
        </Response>
      </Channel>
      <Channel code="BHZ" locationCode="00" startDate="2010-01-01T00:00:00Z">
        <Latitude>49.6920</Latitude>
        <Longitude>11.2218</Longitude>
        <Elevation>505.0</Elevation>
        <Depth>10.0</Depth>
        <Azimuth>0.0</Azimuth>
        <Dip>-90.0</Dip>
        <SampleRate>40.0</SampleRate>
        <Sensor><Type>STS-2.5</Type></Sensor>
        <Response>
          <InstrumentSensitivity>
            <Value>1200000000.0</Value>
            <Frequency>1.0</Frequency>
            <InputUnits><Name>M/S</Name></InputUnits>
            <OutputUnits><Name>COUNTS</Name></OutputUnits>
          </InstrumentSensitivity>
          <Stage number="1">
            <PolesZeros>
              <PzTransferFunctionType>LAPLACE (RADIANS/SECOND)</PzTransferFunctionType>
              <NormalizationFactor>{a0:.10f}</NormalizationFactor>
              <NormalizationFrequency>1.0</NormalizationFrequency>
              <Zero number="0"><Real>0.0</Real><Imaginary>0.0</Imaginary></Zero>
              <Pole number="0"><Real>{p0_re:.10f}</Real><Imaginary>0.0</Imaginary></Pole>
            </PolesZeros>
            <StageGain><Value>1500.0</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
          <Stage number="2">
            <StageGain><Value>800000.0</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
        </Response>
      </Channel>
      <Channel code="BHN" locationCode="00" startDate="2010-01-01T00:00:00Z">
        <Latitude>49.6920</Latitude>
        <Longitude>11.2218</Longitude>
        <Elevation>505.0</Elevation>
        <Depth>10.0</Depth>
        <Azimuth>0.0</Azimuth>
        <Dip>0.0</Dip>
        <SampleRate>40.0</SampleRate>
        <Sensor><Type>STS-2.5</Type></Sensor>
        <Response>
          <InstrumentSensitivity>
            <Value>1200000000.0</Value>
            <Frequency>1.0</Frequency>
            <InputUnits><Name>M/S</Name></InputUnits>
            <OutputUnits><Name>COUNTS</Name></OutputUnits>
          </InstrumentSensitivity>
        </Response>
      </Channel>
    </Station>
    <Station code="NEWSTA" startDate="2020-01-01T00:00:00Z">
      <Latitude>48.1234</Latitude>
      <Longitude>12.5678</Longitude>
      <Elevation>610.0</Elevation>
      <Site><Name>New Alpine Station</Name></Site>
      <Channel code="HHZ" locationCode="" startDate="2020-01-01T00:00:00Z">
        <Azimuth>0.0</Azimuth>
        <Dip>-90.0</Dip>
        <SampleRate>100.0</SampleRate>
        <Sensor><Type>Trillium-120</Type></Sensor>
        <Response>
          <InstrumentSensitivity>
            <Value>0.5</Value>
            <Frequency>1.0</Frequency>
            <InputUnits><Name>NM/S</Name></InputUnits>
            <OutputUnits><Name>COUNTS</Name></OutputUnits>
          </InstrumentSensitivity>
        </Response>
      </Channel>
    </Station>
  </Network>
</FDSNStationXML>
"""


class TestStationXMLParsing(unittest.TestCase):
    def test_parse_namespace_variations_and_epochs(self) -> None:
        for ns in (
            "http://www.fdsn.org/xml/station/1",
            "http://www.fdsn.org/xml/station/1.1",
            "",
        ):
            inv = parse_stationxml(_make_sample_stationxml(ns))
            self.assertEqual(inv.source, "SHM-Modern Test")
            self.assertEqual(len(inv.stations), 2)
            self.assertEqual(len(inv.channels), 4)

            ep_2005 = parse_iso_epoch("2005-06-15T00:00:00Z")
            ep_2021 = parse_iso_epoch("2021-06-15T00:00:00Z")

            ch_old = inv.get_channel("GR", "GRA1", "00", "BHZ", epoch=ep_2005)
            ch_new = inv.get_channel("GR", "GRA1", "00", "BHZ", epoch=ep_2021)
            self.assertIsNotNone(ch_old)
            self.assertIsNotNone(ch_new)
            assert ch_old is not None and ch_new is not None
            self.assertAlmostEqual(ch_old.sample_rate, 20.0)
            self.assertAlmostEqual(ch_new.sample_rate, 40.0)
            self.assertAlmostEqual(ch_old.instrument_sensitivity.value, 6.0e8)
            self.assertAlmostEqual(ch_new.instrument_sensitivity.value, 1.2e9)
            self.assertEqual(ch_old.dip, -90.0)
            self.assertEqual(len(ch_old.stages), 2)
            self.assertIsNotNone(ch_old.poles_zeros)
            self.assertEqual(len(ch_old.stages[1].coefficients.numerator), 3)

    def test_eval_stationxml_response_multistage_and_pz_types(self) -> None:
        inv = parse_stationxml(_make_sample_stationxml())
        ep_2021 = parse_iso_epoch("2021-06-15T00:00:00Z")
        ch_new = inv.get_channel("GR", "GRA1", "00", "BHZ", epoch=ep_2021)
        assert ch_new is not None

        # Evaluate at normalization frequency 1.0 Hz -> should match InstrumentSensitivity (1.2e9)
        resp_1hz = eval_stationxml_response(ch_new, np.array([1.0]))
        self.assertAlmostEqual(float(np.abs(resp_1hz[0])), 1.2e9, delta=1e3)

        # Normalized response without sensitivity should have magnitude 1.0 at 1.0 Hz
        resp_norm = eval_stationxml_response(ch_new, np.array([1.0]), include_sensitivity=False)
        self.assertAlmostEqual(float(np.abs(resp_norm[0])), 1.0, places=6)

        # FIR stage in old epoch: [0.25, 0.5, 0.25] has DC gain 1.0 and Nyquist gain 0.0
        ep_2005 = parse_iso_epoch("2005-06-15T00:00:00Z")
        ch_old = inv.get_channel("GR", "GRA1", "00", "BHZ", epoch=ep_2005)
        assert ch_old is not None
        resp_nyq = eval_stationxml_response(ch_old, np.array([10.0]))  # Nyquist for 20 Hz = 10 Hz
        self.assertAlmostEqual(float(np.abs(resp_nyq[0])), 0.0, places=6)

        # Test LAPLACE (HERTZ) and DIGITAL (Z-TRANSFORM) PZ types
        xml_types = """<?xml version="1.0" encoding="UTF-8"?>
<FDSNStationXML xmlns="http://www.fdsn.org/xml/station/1">
  <Network code="XX">
    <Station code="HZST">
      <Channel code="BHZ" locationCode="">
        <SampleRate>20.0</SampleRate>
        <Response>
          <InstrumentSensitivity>
            <Value>1000.0</Value>
            <Frequency>1.0</Frequency>
          </InstrumentSensitivity>
          <Stage number="1">
            <PolesZeros>
              <PzTransferFunctionType>LAPLACE (HERTZ)</PzTransferFunctionType>
              <NormalizationFactor>1.0</NormalizationFactor>
              <NormalizationFrequency>1.0</NormalizationFrequency>
              <Zero><Real>0.0</Real><Imaginary>0.0</Imaginary></Zero>
              <Pole><Real>-0.01</Real><Imaginary>0.0</Imaginary></Pole>
            </PolesZeros>
            <StageGain><Value>1000.0</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
        </Response>
      </Channel>
      <Channel code="HHZ" locationCode="">
        <SampleRate>100.0</SampleRate>
        <Response>
          <Stage number="1">
            <PolesZeros>
              <PzTransferFunctionType>DIGITAL (Z-TRANSFORM)</PzTransferFunctionType>
              <NormalizationFactor>1.0</NormalizationFactor>
              <NormalizationFrequency>1.0</NormalizationFrequency>
              <Zero><Real>1.0</Real><Imaginary>0.0</Imaginary></Zero>
              <Pole><Real>0.5</Real><Imaginary>0.0</Imaginary></Pole>
            </PolesZeros>
            <StageGain><Value>500.0</Value><Frequency>1.0</Frequency></StageGain>
          </Stage>
        </Response>
      </Channel>
    </Station>
  </Network>
</FDSNStationXML>
"""
        inv_t = parse_stationxml(xml_types)
        ch_hz = inv_t.get_channel("XX", "HZST", "", "BHZ")
        ch_dig = inv_t.get_channel("XX", "HZST", "", "HHZ")
        assert ch_hz is not None and ch_dig is not None

        # LAPLACE (HERTZ) at f=0 -> 0, at f=5 Hz -> ~1000 * |5i / (5i + 0.01)|
        r_hz = eval_stationxml_response(ch_hz, np.array([0.0, 5.0]))
        self.assertAlmostEqual(float(np.abs(r_hz[0])), 0.0, places=9)
        expected_5 = 1000.0 * abs((5j) / (5j + 0.01))
        self.assertAlmostEqual(float(np.abs(r_hz[1])), expected_5, places=4)

        # DIGITAL (Z-TRANSFORM) at f=0 (z=1) -> zero at z=1 gives 0; at Nyquist f=50 Hz (z=-1) -> 500 * |-1 - 1| / |-1 - 0.5| = 500 * 2 / 1.5
        r_dig = eval_stationxml_response(ch_dig, np.array([0.0, 50.0]))
        self.assertAlmostEqual(float(np.abs(r_dig[0])), 0.0, places=9)
        self.assertAlmostEqual(float(np.abs(r_dig[1])), 500.0 * (2.0 / 1.5), places=4)

    def test_shm_config_load_stationxml_file_and_string(self) -> None:
        cfg = SHMConfig(SHM_DEFAULT_ROOT)
        with tempfile.TemporaryDirectory() as tmpdir:
            sxml_path = os.path.join(tmpdir, "stations.xml")
            with open(sxml_path, "w", encoding="utf-8") as f:
                f.write(_make_sample_stationxml())

            inv = cfg.load_stationxml(sxml_path)
            self.assertEqual(len(inv.stations), 2)

        # New station merged into cfg.stations
        new_sta = cfg.get_station("NEWSTA")
        self.assertAlmostEqual(new_sta.lat, 48.1234, places=4)
        self.assertAlmostEqual(new_sta.lon, 12.5678, places=4)
        self.assertEqual(new_sta.instrument, "Trillium-120")

        # Existing station GRA1 preserved & updated
        gra1 = cfg.get_station("GRA1")
        self.assertAlmostEqual(gra1.lat, 49.6919, places=4)

        # Epoch-aware sensitivity query (nm/s per count)
        # 2005 epoch: 6e8 counts/(m/s) -> 1e9 / 6e8 = 1.6666667 nm/s per count
        ep_2005 = parse_sh_time("15-JUN-2005_00:00:00.000")
        ep_2021 = parse_sh_time("15-JUN-2021_00:00:00.000")
        self.assertAlmostEqual(cfg.get_sensitivity("gra1-bh-z", ep_2005), 1e9 / 6.0e8, places=6)
        self.assertAlmostEqual(cfg.get_sensitivity("gra1-bh-z", ep_2021), 1e9 / 1.2e9, places=6)
        # NM/S input units: 0.5 counts/(nm/s) -> 2.0 nm/s per count
        self.assertAlmostEqual(cfg.get_sensitivity("newsta-hh-z", ep_2021), 2.0, places=6)

        # Channel metadata query by epoch (float and ISO string)
        meta_old = cfg.get_channel_metadata("GR", "GRA1", "00", "BHZ", epoch=ep_2005)
        meta_new = cfg.get_channel_metadata("GR", "GRA1", "00", "BHZ", epoch="2021-06-15T00:00:00Z")
        meta_hor = cfg.get_channel_metadata("GR", "GRA1", "00", "BHN")
        self.assertIsNotNone(meta_old)
        self.assertIsNotNone(meta_new)
        self.assertIsNotNone(meta_hor)
        assert meta_old is not None and meta_new is not None and meta_hor is not None
        self.assertEqual(meta_old.sample_rate, 20.0)
        self.assertEqual(meta_new.sample_rate, 40.0)
        self.assertAlmostEqual(meta_new.latitude, 49.6920, places=4)
        self.assertEqual(meta_hor.dip, 0.0)
        self.assertEqual(meta_hor.azimuth, 0.0)

    def test_invalid_inputs_raise_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            parse_stationxml("/nonexistent/path/to/station.xml")
        with self.assertRaises(ValueError):
            parse_stationxml("<NotStationXML/>")
        with self.assertRaises(ValueError):
            parse_stationxml("<FDSNStationXML><broken")


if __name__ == "__main__":
    unittest.main()
