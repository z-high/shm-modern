"""Intensive tests for all SHM data formats and I/O modules (Mini-SEED, Steim-1/2, SFD, GSE1/2, CM6, Q-files, ASCII, .evt)."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
import numpy as np

from shm_app import SHM_DEFAULT_ROOT
from shm_app.core.config_station import get_config
from shm_app.core.models import AnalysisParamSet, PhasePick, Trace
from shm_app.core.time_utils import format_sh_time, parse_sh_time, tc_tadd, tc_tdiff
from shm_app.io.evt import format_ev2view_summary, read_evt_file, write_evt_file
from shm_app.io.gse import compress_cm6, decompress_cm6, gse_checksum, read_gse_file, write_gse2_file
from shm_app.io.mseed import (
    _decode_record_samples,
    compress_steim1_record,
    decompress_steim,
    generate_sfd_file,
    parse_seed_header,
    parse_sfd_file,
    read_from_sfd,
    read_mseed_bytes,
    read_mseed_file,
    write_mseed_file,
)
from shm_app.io.qfile import read_ascii_file, read_qfile, write_ascii_file, write_qfile


class TestTimeUtilsAndConfig(unittest.TestCase):
    def test_sh_time_roundtrip_and_arithmetic(self) -> None:
        t_str = "02-NOV-2003_02:23:54.239"
        ep = parse_sh_time(t_str)
        self.assertEqual(format_sh_time(ep), t_str)

        t_plus = tc_tadd(t_str, 65.5)
        self.assertAlmostEqual(tc_tdiff(t_plus, t_str), 65.5, places=3)

        # GSE2 format parsing
        ep_gse = parse_sh_time("2003/11/02 02:23:54.239")
        self.assertAlmostEqual(ep, ep_gse, places=3)

    def test_station_sensitivity_filter_and_fe_region_lookup(self) -> None:
        cfg = get_config()
        self.assertIn("GRA1", cfg.stations)
        self.assertIn("BFO", cfg.stations)
        gra1 = cfg.get_station("GRA1")
        self.assertAlmostEqual(gra1.lat, 49.6919, places=3)
        self.assertAlmostEqual(gra1.lon, 11.2217, places=3)

        calib = cfg.get_sensitivity("bfo-bh-z", parse_sh_time("02-NOV-2003_02:25:00.000"))
        self.assertGreater(calib, 0.0)

        tf = cfg.get_transfer_prefix("gra1-bh-z", parse_sh_time("02-NOV-2003_02:25:00.000"))
        self.assertEqual(tf, "GRF")

        # Flinn-Engdahl lookup for Germany (~49.7N, 11.2E) -> Region 543 (Germany)
        igreg, grname, isreg, srname = cfg.fer.lookup(49.7, 11.2)
        self.assertEqual(igreg, 543)
        self.assertIn("Germany", grname)
        self.assertEqual(isreg, 36)
        self.assertIn("EUROPE", srname.upper())


class TestMiniSEEDAndSFD(unittest.TestCase):
    def test_read_all_36_bundled_mseed_files_and_sfd(self) -> None:
        cfg = get_config()
        mseed_dir = os.path.join(SHM_DEFAULT_ROOT, "data-examples", "mseed")
        with tempfile.TemporaryDirectory() as tmpdir:
            sfd_path = os.path.join(tmpdir, "sfdlist.sfd")
            gen_sfd = generate_sfd_file(mseed_dir, sfd_path, config=cfg)
            self.assertEqual(len(gen_sfd), 36)

            recs = parse_sfd_file(sfd_path)
            self.assertEqual(len(recs), 36)

            # Read every single .bhz file and verify non-empty continuous trace
            for rec in recs:
                trs = read_mseed_file(rec.filepath, calib_override=rec.calib, config=cfg)
                self.assertEqual(len(trs), 1, f"Expected 1 merged trace from {rec.filepath}")
                tr = trs[0]
                self.assertGreater(tr.length, 5000)
                self.assertAlmostEqual(tr.delta, 0.05, places=4)
                self.assertFalse(np.any(np.isnan(tr.data)))
                self.assertGreater(float(np.std(tr.data)), 0.0)

            # Test windowed read via SFD
            t0 = parse_sh_time("02-NOV-2003_02:25:00.000")
            sub_trs = read_from_sfd(sfd_path, streams=["gra1-bh-z", "grb1-bh-z", "grc1-bh-z"], t_start=t0, duration=60.0)
            self.assertGreaterEqual(len(sub_trs), 3)
            for st in sub_trs:
                self.assertEqual(st.length, 1200)

    def test_mseed_steim1_write_and_sfd_generation_roundtrip(self) -> None:
        t = np.linspace(0, 20.0, 400)
        sig = (500.0 * np.sin(2.0 * np.pi * 1.2 * t) + 120.0 * np.cos(2.0 * np.pi * 3.5 * t)).astype(np.float64)
        tr = Trace(
            data=sig,
            delta=0.05,
            station="GRA1",
            comp="Z",
            chan1="B",
            chan2="H",
            network="GR",
            start_epoch=parse_sh_time("02-NOV-2003_02:25:00.000"),
            calib=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mseed_out = os.path.join(tmpdir, "test_gra1.bhz")
            write_mseed_file(mseed_out, [tr])
            read_back = read_mseed_file(mseed_out, calib_override=1.0)
            self.assertEqual(len(read_back), 1)
            rb = read_back[0]
            self.assertEqual(rb.station, "GRA1")
            self.assertEqual(rb.length, tr.length)
            np.testing.assert_allclose(rb.data, np.round(tr.data), atol=1.0)

            sfd_out = os.path.join(tmpdir, "sfdlist.sfd")
            gen_recs = generate_sfd_file(tmpdir, sfd_out)
            self.assertEqual(len(gen_recs), 1)
            self.assertEqual(gen_recs[0].stream, "gra1-bh-z")


class TestGSEFilesAndCM6(unittest.TestCase):
    def test_read_all_bundled_gse_files_and_verify_checksums(self) -> None:
        gse_dir = os.path.join(SHM_DEFAULT_ROOT, "data-examples", "gse")
        expected_counts = {
            "GRF_031102_0225.GSE": 19,
            "GRSN_031002_2334.GSE": 48,
            "GRSN_031102_0224.GSE": 21,
        }
        for fname, exp_n in expected_counts.items():
            fpath = os.path.join(gse_dir, fname)
            trs = read_gse_file(fpath)
            self.assertEqual(len(trs), exp_n, f"Trace count mismatch in {fname}")
            for tr in trs:
                self.assertTrue(
                    tr.extra_headers.get("checksum_ok", False),
                    f"GSE CHK2 checksum failed for {tr.station} in {fname}",
                )
                self.assertGreater(tr.length, 1000)

    def test_gse2_cm6_write_and_read_roundtrip(self) -> None:
        raw_ints = np.array([0, 15, -42, 1024, -35000, 123456, -98765, 500, 12, 0], dtype=np.int64)
        lines = compress_cm6(raw_ints, n_differences=2)
        dec = decompress_cm6(lines, expected_samples=raw_ints.size, n_integrations=2)
        np.testing.assert_array_equal(dec, raw_ints)

        tr = Trace(
            data=raw_ints.astype(np.float64),
            delta=0.05,
            station="BFO",
            comp="Z",
            chan1="B",
            chan2="H",
            start_epoch=parse_sh_time("02-NOV-2003_02:25:00.000"),
            calib=1.0,
            calper=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            gse_out = os.path.join(tmpdir, "roundtrip.GSE")
            write_gse2_file(gse_out, [tr])
            rb = read_gse_file(gse_out)
            self.assertEqual(len(rb), 1)
            self.assertTrue(rb[0].extra_headers.get("checksum_ok"))
            np.testing.assert_allclose(rb[0].data, tr.data, atol=1e-4)


class TestQFileASCIIAndEvt(unittest.TestCase):
    def test_qfile_and_ascii_roundtrip(self) -> None:
        tr1 = Trace(
            data=np.linspace(-100.0, 200.0, 250),
            delta=0.05,
            station="GRA1",
            comp="Z",
            chan1="B",
            chan2="H",
            start_epoch=parse_sh_time("02-NOV-2003_02:25:00.000"),
            distance=64.5,
            azimuth=48.2,
            depth=33.0,
            comment="QFile Test",
        )
        tr2 = Trace(
            data=np.cos(np.linspace(0.0, 10.0, 300)) * 50.0,
            delta=0.05,
            station="GRA2",
            comp="N",
            chan1="B",
            chan2="H",
            start_epoch=parse_sh_time("02-NOV-2003_02:25:00.000"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            qpath = os.path.join(tmpdir, "test_q")
            write_qfile(qpath, [tr1, tr2])
            q_back = read_qfile(qpath)
            self.assertEqual(len(q_back), 2)
            self.assertEqual(q_back[0].station, "GRA1")
            self.assertAlmostEqual(q_back[0].distance, 64.5, places=3)
            np.testing.assert_allclose(q_back[0].data, tr1.data, rtol=1e-5, atol=1e-4)
            np.testing.assert_allclose(q_back[1].data, tr2.data, rtol=1e-5, atol=1e-4)

            apath = os.path.join(tmpdir, "test.asc")
            write_ascii_file(apath, [tr1, tr2])
            a_back = read_ascii_file(apath)
            self.assertEqual(len(a_back), 2)
            np.testing.assert_allclose(a_back[0].data, tr1.data, rtol=1e-4, atol=1e-3)

    def test_evt_file_roundtrip_and_ev2view(self) -> None:
        param = AnalysisParamSet(
            event_id=20031102,
            origin="02-NOV-2003_02:15:10.500",
            lat=12.345,
            lon=142.678,
            depth=45.0,
            distance=82.4,
            azimuth=41.5,
            slowness=5.12,
            mag_mb=5.8,
            mag_ms=5.6,
            region_id=216,
            region_name="Mariana Islands",
            comment="Test Mariana Event",
        )
        pk = PhasePick(
            name="P",
            onset="02-NOV-2003_02:27:25.120",
            spec="i",
            sign=1,
            quality=6,
            weight=1,
            ampl=245.0,
            period=1.2,
            mag_mb=5.8,
            station="GRA1",
            comp="Z",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ev_path = os.path.join(tmpdir, "test.evt")
            write_evt_file(ev_path, param, [pk])
            p_back, pks_back = read_evt_file(ev_path)
            self.assertEqual(len(pks_back), 1)
            self.assertEqual(pks_back[0].station, "GRA1")
            self.assertEqual(pks_back[0].sign, 1)
            self.assertAlmostEqual(p_back.lat, 12.345, places=3)
            self.assertAlmostEqual(p_back.mag_mb, 5.8, places=2)
            self.assertEqual(p_back.comment, "Test Mariana Event")

            summary = format_ev2view_summary(p_back, pks_back)
            self.assertIn("Mariana Islands", summary)
            self.assertIn("GRA1", summary)


class TestMiniSEEDEdgeCasesAndSteim2(unittest.TestCase):
    """Edge-case tests for Steim-2 decompression, Little-Endian Mini-SEED records,
    Blockette 1001 timing corrections, and gap/overlap merging in read_mseed_bytes.
    """

    @staticmethod
    def _u32(val: int) -> int:
        return int(val) & 0xFFFFFFFF

    def test_steim2_decompression_all_nibbles_and_dnibs(self) -> None:
        """Verify Steim-2 decompression for all nibble and dnib combinations:
        nib=2, dnib=1 (1x30-bit)
        nib=2, dnib=2 (2x15-bit)
        nib=2, dnib=3 (3x10-bit)
        nib=3, dnib=0 (5x6-bit)
        nib=3, dnib=1 (6x5-bit)
        nib=3, dnib=2 (7x4-bit)
        and multi-frame continuity.
        """
        # 1. nib=2, dnib=1 (1x30-bit signed difference)
        diffs_30 = [0, 12345678, -12345678, 536870911, -536870912]
        x0 = 1000
        w0 = sum(2 << (30 - 2 * (3 + i)) for i in range(len(diffs_30)))
        words_30 = [w0, self._u32(x0), self._u32(x0 + sum(diffs_30[1:]))]
        for d in diffs_30:
            words_30.append((1 << 30) | (d & 0x3FFFFFFF))
        words_30 += [0] * (16 - len(words_30))
        frame_30 = struct.pack(">16I", *words_30)
        res_30 = decompress_steim(frame_30, len(diffs_30), steim_level=2)
        expected_30 = np.cumsum([x0] + diffs_30[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_30, expected_30)

        # 2. nib=2, dnib=2 (2x15-bit signed differences)
        pairs = [(0, 1250), (-850, 16383), (-16384, 42)]
        w0_15 = sum(2 << (30 - 2 * (3 + i)) for i in range(len(pairs)))
        words_15 = [w0_15, self._u32(500), 0]
        all_diffs_15 = []
        for d0, d1 in pairs:
            w = (2 << 30) | ((d0 & 0x7FFF) << 15) | (d1 & 0x7FFF)
            words_15.append(w)
            all_diffs_15.extend([d0, d1])
        words_15 += [0] * (16 - len(words_15))
        frame_15 = struct.pack(">16I", *words_15)
        res_15 = decompress_steim(frame_15, len(all_diffs_15), steim_level=2)
        expected_15 = np.cumsum([500] + all_diffs_15[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_15, expected_15)

        # 3. nib=2, dnib=3 (3x10-bit signed differences)
        triplets = [(0, 250, -180), (511, -512, -1)]
        w0_10 = sum(2 << (30 - 2 * (3 + i)) for i in range(len(triplets)))
        words_10 = [w0_10, self._u32(200), 0]
        all_diffs_10 = []
        for d0, d1, d2 in triplets:
            w = (3 << 30) | ((d0 & 0x3FF) << 20) | ((d1 & 0x3FF) << 10) | (d2 & 0x3FF)
            words_10.append(w)
            all_diffs_10.extend([d0, d1, d2])
        words_10 += [0] * (16 - len(words_10))
        frame_10 = struct.pack(">16I", *words_10)
        res_10 = decompress_steim(frame_10, len(all_diffs_10), steim_level=2)
        expected_10 = np.cumsum([200] + all_diffs_10[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_10, expected_10)

        # 4. nib=3, dnib=0 (5x6-bit signed differences)
        groups_6 = [(0, 15, -12, 31, -32), (7, -8, 0, -1, 1)]
        w0_6 = sum(3 << (30 - 2 * (3 + i)) for i in range(len(groups_6)))
        words_6 = [w0_6, self._u32(10), 0]
        all_diffs_6 = []
        for g in groups_6:
            w = (0 << 30) | ((g[0] & 0x3F) << 24) | ((g[1] & 0x3F) << 18) | ((g[2] & 0x3F) << 12) | ((g[3] & 0x3F) << 6) | (g[4] & 0x3F)
            words_6.append(w)
            all_diffs_6.extend(g)
        words_6 += [0] * (16 - len(words_6))
        frame_6 = struct.pack(">16I", *words_6)
        res_6 = decompress_steim(frame_6, len(all_diffs_6), steim_level=2)
        expected_6 = np.cumsum([10] + all_diffs_6[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_6, expected_6)

        # 5. nib=3, dnib=1 (6x5-bit signed differences)
        groups_5 = [(0, 7, -8, 15, -16, 3)]
        w0_5 = sum(3 << (30 - 2 * (3 + i)) for i in range(len(groups_5)))
        words_5 = [w0_5, self._u32(0), 0]
        all_diffs_5 = []
        for g in groups_5:
            w = (1 << 30) | ((g[0] & 0x1F) << 25) | ((g[1] & 0x1F) << 20) | ((g[2] & 0x1F) << 15) | ((g[3] & 0x1F) << 10) | ((g[4] & 0x1F) << 5) | (g[5] & 0x1F)
            words_5.append(w)
            all_diffs_5.extend(g)
        words_5 += [0] * (16 - len(words_5))
        frame_5 = struct.pack(">16I", *words_5)
        res_5 = decompress_steim(frame_5, len(all_diffs_5), steim_level=2)
        expected_5 = np.cumsum([0] + all_diffs_5[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_5, expected_5)

        # 6. nib=3, dnib=2 (7x4-bit signed differences)
        groups_4 = [(0, 5, -4, 7, -8, 1, -2)]
        w0_4 = sum(3 << (30 - 2 * (3 + i)) for i in range(len(groups_4)))
        words_4 = [w0_4, self._u32(5), 0]
        all_diffs_4 = []
        for g in groups_4:
            w = (2 << 30) | ((g[0] & 0x0F) << 24) | ((g[1] & 0x0F) << 20) | ((g[2] & 0x0F) << 16) | ((g[3] & 0x0F) << 12) | ((g[4] & 0x0F) << 8) | ((g[5] & 0x0F) << 4) | (g[6] & 0x0F)
            words_4.append(w)
            all_diffs_4.extend(g)
        words_4 += [0] * (16 - len(words_4))
        frame_4 = struct.pack(">16I", *words_4)
        res_4 = decompress_steim(frame_4, len(all_diffs_4), steim_level=2)
        expected_4 = np.cumsum([5] + all_diffs_4[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_4, expected_4)

        # 7. Multi-frame continuity (Frame 0 -> Frame 1)
        f0_words = [2 << 24, self._u32(100), 0, (2 << 30) | (0 << 15) | 10] + [0] * 12
        f1_w0 = 2 << (30 - 2 * 1)  # nibble 2 at word 1
        f1_w1 = (3 << 30) | ((20 & 0x3FF) << 20) | ((-5 & 0x3FF) << 10) | (15 & 0x3FF)
        f1_words = [f1_w0, f1_w1] + [0] * 14
        multi_frame = struct.pack(">16I", *f0_words) + struct.pack(">16I", *f1_words)
        res_multi = decompress_steim(multi_frame, 5, steim_level=2)
        expected_multi = np.array([100, 110, 130, 125, 140], dtype=np.int32)
        np.testing.assert_array_equal(res_multi, expected_multi)

        # 8. Edge cases: empty data, zero/negative sample count
        self.assertEqual(decompress_steim(b"", 10, steim_level=2).size, 0)
        self.assertEqual(decompress_steim(frame_30, 0, steim_level=2).size, 0)
        self.assertEqual(decompress_steim(frame_30, -5, steim_level=2).size, 0)
        self.assertEqual(decompress_steim(b"\x00" * 32, 10, steim_level=2).size, 0)

        # 9. nib=1 (four 8-bit signed differences) under steim_level=2
        diffs_8 = [0, 127, -128, 42, -50, 10, -1, 0]
        x0_8 = 500
        w0_8 = (1 << (30 - 2 * 3)) | (1 << (30 - 2 * 4))
        w3_8 = ((diffs_8[0] & 0xFF) << 24) | ((diffs_8[1] & 0xFF) << 16) | ((diffs_8[2] & 0xFF) << 8) | (diffs_8[3] & 0xFF)
        w4_8 = ((diffs_8[4] & 0xFF) << 24) | ((diffs_8[5] & 0xFF) << 16) | ((diffs_8[6] & 0xFF) << 8) | (diffs_8[7] & 0xFF)
        words_8 = [w0_8, self._u32(x0_8), 0, w3_8, w4_8] + [0] * 11
        frame_8 = struct.pack(">16I", *words_8)
        res_8 = decompress_steim(frame_8, len(diffs_8), steim_level=2)
        expected_8 = np.cumsum([x0_8] + diffs_8[1:]).astype(np.int32)
        np.testing.assert_array_equal(res_8, expected_8)

        # 10. Reserved dnib handling (nib=2, dnib=0 and nib=3, dnib=3) without unhandled crashes
        # A frame with only reserved words returns empty array without raising
        w0_res = (2 << (30 - 2 * 3)) | (3 << (30 - 2 * 4))
        w3_res = (0 << 30) | 0x1234567  # nib=2, dnib=0
        w4_res = (3 << 30) | 0x7654321  # nib=3, dnib=3
        frame_res = struct.pack(">16I", *([w0_res, 100, 0, w3_res, w4_res] + [0] * 11))
        res_res = decompress_steim(frame_res, 10, steim_level=2)
        self.assertEqual(res_res.size, 0)

        # Reserved words interspersed with valid difference words
        w0_mix = (2 << (30 - 2 * 3)) | (2 << (30 - 2 * 4)) | (3 << (30 - 2 * 5)) | (2 << (30 - 2 * 6))
        w3_mix = (1 << 30) | 0  # valid 1x30-bit diff = 0
        w4_mix = (0 << 30) | 0x3FFFFFF  # reserved nib=2, dnib=0
        w5_mix = (3 << 30) | 0x3FFFFFF  # reserved nib=3, dnib=3
        w6_mix = (1 << 30) | (250 & 0x3FFFFFFF)  # valid 1x30-bit diff = 250
        frame_mix = struct.pack(">16I", *([w0_mix, 1000, 0, w3_mix, w4_mix, w5_mix, w6_mix] + [0] * 9))
        res_mix = decompress_steim(frame_mix, 2, steim_level=2)
        np.testing.assert_array_equal(res_mix, np.array([1000, 1250], dtype=np.int32))

        # 11. Mid-word sample count cutoff
        # Requesting 4 samples from a 7x4-bit word (nib=3, dnib=2)
        res_cut4 = decompress_steim(frame_4, 4, steim_level=2)
        self.assertEqual(res_cut4.size, 4)
        np.testing.assert_array_equal(res_cut4, expected_4[:4])

        # Requesting 1 sample
        res_cut1 = decompress_steim(frame_4, 1, steim_level=2)
        self.assertEqual(res_cut1.size, 1)
        np.testing.assert_array_equal(res_cut1, expected_4[:1])

        # 12. Interspersed nib=0 words
        w0_zero = (2 << (30 - 2 * 3)) | (0 << (30 - 2 * 4)) | (2 << (30 - 2 * 5)) | (0 << (30 - 2 * 6)) | (1 << (30 - 2 * 7))
        w3_z = (1 << 30) | 0  # 1x30-bit diff = 0
        w4_z = 0xDEADBEEF  # nib=0 word (ignored)
        w5_z = (2 << 30) | ((150 & 0x7FFF) << 15) | (-80 & 0x7FFF)  # 2x15-bit diffs
        w6_z = 0xCAFEBABE  # nib=0 word (ignored)
        w7_z = ((20 & 0xFF) << 24) | ((-10 & 0xFF) << 16) | ((5 & 0xFF) << 8) | (-2 & 0xFF)  # 4x8-bit diffs
        words_zero = [w0_zero, 2000, 0, w3_z, w4_z, w5_z, w6_z, w7_z] + [0] * 8
        frame_zero = struct.pack(">16I", *words_zero)
        res_zero = decompress_steim(frame_zero, 7, steim_level=2)
        expected_zero = np.cumsum([2000, 150, -80, 20, -10, 5, -2]).astype(np.int32)
        np.testing.assert_array_equal(res_zero, expected_zero)

    def test_little_endian_miniseed_header_and_payload_decoding(self) -> None:
        """Verify parsing of Little-Endian Mini-SEED records:
        - Header byte-order auto-detection via year > 2050 in big-endian
        - Blockette 1000 word_order=0
        - Payload decoding for Steim-1, Steim-2, and numeric formats (int16, int32, float32)
        - Trace creation via read_mseed_bytes
        """
        reclth = 256
        endian = "<"
        seq_str = b"000001"
        indicator = b"D"
        reserved_a = 8  # 1 << 8 = 256
        stat_b = b"LEST "
        loc_b = b"  "
        chan_b = b"BHZ"
        net_b = b"GR"
        year, jday, hour, minute, sec, align, frac_0001 = 2003, 306, 2, 25, 0, 0, 0
        nsmp = 10
        fact, mult = 20, 1  # dt = 0.05
        activity, ioflags, quality = 0, 0, 0
        nblock = 1
        timecorr = 0
        databegin = 64
        first_blk = 48

        hdr_fixed = struct.pack(
            f"{endian}6scB5s2s3s2sHHBBBBHHhhBBBBiHH",
            seq_str, indicator, reserved_a, stat_b, loc_b, chan_b, net_b,
            year, jday, hour, minute, sec, align, frac_0001,
            nsmp, fact, mult, activity, ioflags, quality, nblock, timecorr, databegin, first_blk,
        )

        # 1. Little-Endian Steim-2 record
        btype = 1000
        next_blk = 0
        encoding_format = 11  # Steim-2
        word_order = 0  # Little-Endian
        recexp = 8  # 256 bytes
        blk1000 = struct.pack(f"{endian}HHBBBB", btype, next_blk, encoding_format, word_order, recexp, 0)
        pad_hdr = b"\x00" * (64 - len(hdr_fixed) - len(blk1000))

        # Pack 10 Steim-2 samples in little-endian words
        w0 = (2 << 24) | (2 << 22) | (2 << 20) | (2 << 18)
        x0 = 100
        w3 = (3 << 30) | ((0 & 0x3FF) << 20) | ((10 & 0x3FF) << 10) | (20 & 0x3FF)
        w4 = (3 << 30) | ((30 & 0x3FF) << 20) | ((40 & 0x3FF) << 10) | (50 & 0x3FF)
        w5 = (3 << 30) | ((60 & 0x3FF) << 20) | ((70 & 0x3FF) << 10) | (80 & 0x3FF)
        w6 = (1 << 30) | (90 & 0x3FFFFFFF)
        le_words = [w0, self._u32(x0), 0, w3, w4, w5, w6] + [0] * 9
        frame_le = struct.pack("<16I", *le_words)
        rec_steim2_le = hdr_fixed + blk1000 + pad_hdr + frame_le + b"\x00" * (reclth - 64 - len(frame_le))

        hdr_parsed = parse_seed_header(rec_steim2_le)
        self.assertFalse(hdr_parsed.big_endian_hdr)
        self.assertEqual(hdr_parsed.word_order, 0)
        self.assertEqual(hdr_parsed.encoding_format, 11)
        self.assertEqual(hdr_parsed.reclth, 256)
        self.assertEqual(hdr_parsed.station, "LEST")
        self.assertEqual(hdr_parsed.channel, "BHZ")

        decoded_s2 = _decode_record_samples(rec_steim2_le[hdr_parsed.databegin:], hdr_parsed)
        expected_s2 = np.cumsum([100, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        np.testing.assert_array_equal(decoded_s2, expected_s2)

        # 2. End-to-end read_mseed_bytes with LE Steim-2
        trs = read_mseed_bytes(rec_steim2_le)
        self.assertEqual(len(trs), 1)
        self.assertEqual(trs[0].station, "LEST")
        self.assertEqual(trs[0].channel3, "BHZ")
        self.assertEqual(trs[0].data.size, 10)
        np.testing.assert_allclose(trs[0].data, expected_s2)

        # 3. Numeric Little-Endian formats: int16 (fmt=1), int32 (fmt=3), float32 (fmt=4), float64 (fmt=5)
        for fmt, dt_str, test_vals in [
            (1, "<i2", np.array([-100, 0, 50, 12000], dtype=np.int16)),
            (3, "<i4", np.array([-100000, 0, 500000], dtype=np.int32)),
            (4, "<f4", np.array([-12.5, 0.0, 42.125], dtype=np.float32)),
            (5, "<f8", np.array([-12345.67890123, 0.0, 98765.43210987, 1.23456789e-4], dtype=np.float64)),
        ]:
            blk_fmt = struct.pack(f"{endian}HHBBBB", 1000, 0, fmt, 0, 8, 0)
            hdr_num = struct.pack(
                f"{endian}6scB5s2s3s2sHHBBBBHHhhBBBBiHH",
                seq_str, indicator, reserved_a, stat_b, loc_b, chan_b, net_b,
                year, jday, hour, minute, sec, align, frac_0001,
                len(test_vals), fact, mult, activity, ioflags, quality, nblock, timecorr, databegin, first_blk,
            )
            raw_payload = test_vals.tobytes()
            rec_num = hdr_num + blk_fmt + b"\x00" * (64 - len(hdr_num) - len(blk_fmt)) + raw_payload
            rec_num += b"\x00" * (reclth - len(rec_num))
            hdr_n = parse_seed_header(rec_num)
            self.assertEqual(hdr_n.word_order, 0)
            self.assertEqual(hdr_n.encoding_format, fmt)
            dec_n = _decode_record_samples(rec_num[hdr_n.databegin:], hdr_n)
            np.testing.assert_allclose(dec_n, test_vals, rtol=1e-12 if fmt == 5 else 1e-5)
            # End-to-end trace parsing via read_mseed_bytes
            trs_n = read_mseed_bytes(rec_num)
            self.assertEqual(len(trs_n), 1)
            np.testing.assert_allclose(trs_n[0].data, test_vals, rtol=1e-12 if fmt == 5 else 1e-5)

        # 4. Little-Endian Steim-1 record (fmt=10, word_order=0)
        blk1000_s1 = struct.pack(f"{endian}HHBBBB", 1000, 0, 10, 0, 8, 0)
        pad_s1 = b"\x00" * (64 - len(hdr_fixed) - len(blk1000_s1))
        # Word 3: nib=1 (four 8-bit: 0, 15, -20, 30)
        # Word 4: nib=2 (two 16-bit: -250, 400)
        # Word 5: nib=3 (one 32-bit: 50000)
        w0_s1 = (1 << 24) | (2 << 22) | (3 << 20)
        x0_s1 = 500
        w3_s1 = ((0 & 0xFF) << 24) | ((15 & 0xFF) << 16) | ((-20 & 0xFF) << 8) | (30 & 0xFF)
        w4_s1 = ((-250 & 0xFFFF) << 16) | (400 & 0xFFFF)
        w5_s1 = 50000 & 0xFFFFFFFF
        words_s1_le = [w0_s1, self._u32(x0_s1), 0, w3_s1, w4_s1, w5_s1] + [0] * 10
        frame_s1_le = struct.pack("<16I", *words_s1_le)
        rec_s1_le = hdr_fixed + blk1000_s1 + pad_s1 + frame_s1_le + b"\x00" * (reclth - 64 - len(frame_s1_le))

        hdr_s1 = parse_seed_header(rec_s1_le)
        self.assertFalse(hdr_s1.big_endian_hdr)
        self.assertEqual(hdr_s1.word_order, 0)
        self.assertEqual(hdr_s1.encoding_format, 10)

        expected_s1 = np.cumsum([500, 15, -20, 30, -250, 400, 50000])
        dec_s1 = _decode_record_samples(rec_s1_le[hdr_s1.databegin:], hdr_s1)
        np.testing.assert_array_equal(dec_s1, expected_s1)

        trs_s1 = read_mseed_bytes(rec_s1_le)
        self.assertEqual(len(trs_s1), 1)
        np.testing.assert_allclose(trs_s1[0].data, expected_s1)

    def test_blockette_1001_and_time_correction_logic(self) -> None:
        """Verify Blockette 1001 microsecond correction and activity flag timecorr:
        - activity & 0x02 == 0: timecorr * 0.0001 is added to start_epoch
        - activity & 0x02 != 0: timecorr is NOT added (already applied in header)
        - Blockette 1001 usec_corr (signed int8) is added as usec_corr * 1e-6
        - Timing quality is parsed properly
        """
        t0 = 1067739900.0  # 02-NOV-2003 02:25:00.000
        raw_base, _ = compress_steim1_record(
            np.arange(20, dtype=np.int32), "TEST", "BHZ", start_epoch=t0, reclth=256
        )

        # 1. Header timecorr without activity flag (timecorr not yet applied)
        ba1 = bytearray(raw_base)
        struct.pack_into(">i", ba1, 40, 5000)  # +0.5 s
        ba1[36] = 0  # activity bit 1 is 0
        hdr1 = parse_seed_header(bytes(ba1))
        self.assertAlmostEqual(hdr1.start_epoch, t0 + 0.5, places=6)

        # 2. Header timecorr WITH activity flag bit 1 (timecorr already applied)
        ba2 = bytearray(raw_base)
        struct.pack_into(">i", ba2, 40, 5000)
        ba2[36] = 2  # activity bit 1 is 1 (0x02)
        hdr2 = parse_seed_header(bytes(ba2))
        self.assertAlmostEqual(hdr2.start_epoch, t0, places=6)

        # 3. Blockette 1001 with negative microsecond correction (-75 us)
        ba3 = bytearray(raw_base)
        struct.pack_into(">H", ba3, 50, 56)  # B1000 next_blk = 56
        b1001_neg = struct.pack(">HHBbBB", 1001, 0, 92, -75, 0, 4)
        ba3[56:64] = b1001_neg
        ba3[39] = 2  # nblock = 2
        hdr3 = parse_seed_header(bytes(ba3))
        self.assertAlmostEqual(hdr3.start_epoch, t0 - 75e-6, places=6)

        # 4. Blockette 1001 with positive microsecond correction (+120 us)
        ba4 = bytearray(raw_base)
        struct.pack_into(">H", ba4, 50, 56)
        b1001_pos = struct.pack(">HHBbBB", 1001, 0, 98, 120, 0, 4)
        ba4[56:64] = b1001_pos
        ba4[39] = 2  # nblock = 2
        hdr4 = parse_seed_header(bytes(ba4))
        self.assertAlmostEqual(hdr4.start_epoch, t0 + 120e-6, places=6)

        # 5. Combined timecorr and Blockette 1001 usec_corr
        ba5 = bytearray(raw_base)
        struct.pack_into(">i", ba5, 40, 2000)  # +0.2 s
        ba5[36] = 0  # unapplied
        struct.pack_into(">H", ba5, 50, 56)
        ba5[56:64] = struct.pack(">HHBbBB", 1001, 0, 85, -45, 0, 4)
        ba5[39] = 2
        hdr5 = parse_seed_header(bytes(ba5))
        self.assertAlmostEqual(hdr5.start_epoch, t0 + 0.2 - 45e-6, places=6)

        # 6. Blockette 1001 signed byte boundary checks (-128 us and +127 us)
        ba_min = bytearray(raw_base)
        struct.pack_into(">H", ba_min, 50, 56)
        ba_min[56:64] = struct.pack(">HHBbBB", 1001, 0, 95, -128, 0, 4)
        ba_min[39] = 2
        hdr_min = parse_seed_header(bytes(ba_min))
        self.assertAlmostEqual(hdr_min.start_epoch, t0 - 128e-6, places=7)

        ba_max = bytearray(raw_base)
        struct.pack_into(">H", ba_max, 50, 56)
        ba_max[56:64] = struct.pack(">HHBbBB", 1001, 0, 95, 127, 0, 4)
        ba_max[39] = 2
        hdr_max = parse_seed_header(bytes(ba_max))
        self.assertAlmostEqual(hdr_max.start_epoch, t0 + 127e-6, places=7)

        # 7. Negative timecorr in fixed header (e.g. -3500 * 0.0001 s = -0.35 s)
        ba_neg = bytearray(raw_base)
        struct.pack_into(">i", ba_neg, 40, -3500)
        ba_neg[36] = 0  # activity bit 1 is 0 (unapplied)
        hdr_neg = parse_seed_header(bytes(ba_neg))
        self.assertAlmostEqual(hdr_neg.start_epoch, t0 - 0.35, places=6)

        # Negative timecorr when activity bit 1 IS set (already applied)
        ba_neg_app = bytearray(raw_base)
        struct.pack_into(">i", ba_neg_app, 40, -3500)
        ba_neg_app[36] = 2  # activity bit 1 is 1
        hdr_neg_app = parse_seed_header(bytes(ba_neg_app))
        self.assertAlmostEqual(hdr_neg_app.start_epoch, t0, places=6)

        # 8. Reversed blockette chain ordering (Blockette 1001 preceding Blockette 1000)
        ba_rev = bytearray(raw_base)
        # Put Blockette 1001 at offset 48, linking to offset 56
        ba_rev[48:56] = struct.pack(">HHBbBB", 1001, 56, 90, 65, 0, 2)
        # Put Blockette 1000 at offset 56, next_blk = 0
        ba_rev[56:64] = struct.pack(">HHBBBB", 1000, 0, 11, 1, 9, 0)
        struct.pack_into(">H", ba_rev, 46, 48)  # first_blk = 48
        ba_rev[39] = 2  # nblock = 2
        hdr_rev = parse_seed_header(bytes(ba_rev))
        self.assertEqual(hdr_rev.encoding_format, 11)
        self.assertEqual(hdr_rev.word_order, 1)
        self.assertEqual(hdr_rev.reclth, 512)
        self.assertAlmostEqual(hdr_rev.start_epoch, t0 + 65e-6, places=6)

    def test_read_mseed_bytes_gap_and_overlap_merging(self) -> None:
        """Verify segment merging in read_mseed_bytes:
        - Contiguous records (|gap| <= 1.5 * dt)
        - Positive gap (1.5 * dt < gap < 3600 s) padded with linear interpolation
        - Negative gap (gap < -1.5 * dt) overlap skipped
        - Time window trimming (t_start, duration)
        """
        t0 = 1067739900.0
        dt = 0.05
        # Record 1: 20 samples from t0
        data1 = np.arange(100, 120, dtype=np.int32)
        rec1, _ = compress_steim1_record(data1, "TEST", "BHZ", start_epoch=t0, dt=dt, reclth=256)

        # 1. Contiguous record: record 2 starts exactly at t0 + 20 * dt
        t1_cont = t0 + 20 * dt
        data2_cont = np.arange(120, 140, dtype=np.int32)
        rec2_cont, _ = compress_steim1_record(data2_cont, "TEST", "BHZ", start_epoch=t1_cont, dt=dt, reclth=256)
        trs_cont = read_mseed_bytes(rec1 + rec2_cont)
        self.assertEqual(len(trs_cont), 1)
        self.assertEqual(trs_cont[0].data.size, 40)
        np.testing.assert_array_equal(trs_cont[0].data, np.concatenate([data1, data2_cont]))
        self.assertAlmostEqual(trs_cont[0].start_epoch, t0, places=4)

        # 2. Positive gap: 4 missing samples (gap = 5 * dt = 0.25 s)
        t1_gap = t0 + 24 * dt
        data2_gap = np.arange(200, 220, dtype=np.int32)
        rec2_gap, _ = compress_steim1_record(data2_gap, "TEST", "BHZ", start_epoch=t1_gap, dt=dt, reclth=256)
        trs_gap = read_mseed_bytes(rec1 + rec2_gap)
        self.assertEqual(len(trs_gap), 1)
        self.assertEqual(trs_gap[0].data.size, 44)
        expected_pad = np.linspace(119.0, 200.0, 4 + 2)[1:-1]
        np.testing.assert_allclose(trs_gap[0].data[20:24], expected_pad, rtol=1e-5)
        np.testing.assert_allclose(trs_gap[0].data[:20], data1)
        np.testing.assert_allclose(trs_gap[0].data[24:], data2_gap)

        # 3. Negative gap (overlap): record 2 starts 5 samples earlier than record 1's end
        t1_ovl = t0 + 15 * dt  # overlap of 5 samples
        data2_ovl = np.arange(115, 135, dtype=np.int32)
        rec2_ovl, _ = compress_steim1_record(data2_ovl, "TEST", "BHZ", start_epoch=t1_ovl, dt=dt, reclth=256)
        trs_ovl = read_mseed_bytes(rec1 + rec2_ovl)
        self.assertEqual(len(trs_ovl), 1)
        self.assertEqual(trs_ovl[0].data.size, 35)
        np.testing.assert_allclose(trs_ovl[0].data[:20], data1)
        np.testing.assert_allclose(trs_ovl[0].data[20:], data2_ovl[5:])

        # 4. Time window trimming (t_start, duration)
        trs_win = read_mseed_bytes(rec1 + rec2_cont, t_start=t0 + 5 * dt, duration=10 * dt)
        self.assertEqual(len(trs_win), 1)
        self.assertEqual(trs_win[0].data.size, 10)
        self.assertAlmostEqual(trs_win[0].start_epoch, t0 + 5 * dt, places=4)
        np.testing.assert_allclose(trs_win[0].data, data1[5:15])


if __name__ == "__main__":
    unittest.main()

