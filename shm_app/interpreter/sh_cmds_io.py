"""File I/O and FDSN/BGR fetch command handlers for SHCommandEngine."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from shm_app.core.models import PhasePick
from shm_app.core.time_utils import parse_sh_time
from shm_app.io.evt import format_ev2view_summary, read_evt_file, write_evt_file
from shm_app.io.geojson import write_geojson
from shm_app.io.gse import read_gse_file, write_gse2_file
from shm_app.io.ascii_io import read_ascii_file, write_ascii_file
from shm_app.io.mseed import read_mseed_file, write_mseed_file
from shm_app.io.qfile import read_qfile, write_qfile
from shm_app.io.sfd import generate_sfd_file, read_from_sfd
from shm_app.io.quakeml import parse_quakeml, write_quakeml


class _IOCommandsMixin:
    """Waveform, event, StationXML/QuakeML/GeoJSON, bulletin, and BGR fetch SH commands."""

    def _cmd_read_qfile(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READ requires a Q-file path")
        fpath = args[0]
        rec_list: Optional[List[int]] = None
        if len(args) > 1 and args[1].upper() != "ALL":
            rec_list = []
            for part in args[1].split(","):
                if "-" in part:
                    a, b = part.split("-", 1)
                    rec_list.extend(range(int(a), int(b) + 1))
                elif part.isdigit():
                    rec_list.append(int(part))
        trs = read_qfile(fpath, rec_indices=rec_list, config=self.config)
        self.traces.extend(trs)
        msg = f"Read {len(trs)} trace(s) from Q-file {fpath}"
        self.emit(msg)
        return msg

    def _cmd_write_qfile(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITE requires a Q-file path")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        qhd, _qbn = write_qfile(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to {qhd}"
        self.emit(msg)
        return msg

    def _cmd_reads(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READS requires an SFD or Mini-SEED file/directory path")
        path = args[0]
        if path.upper() == "INQUIRE":
            path = os.path.join(self.config.data_examples_dir, "mseed")

        t_start: Optional[float] = None
        dur: Optional[float] = None
        streams: Optional[List[str]] = None

        if len(args) >= 2:
            try:
                t_start = parse_sh_time(args[1])
            except ValueError:
                streams = [s.strip() for s in args[1].split(",")]
        if len(args) >= 3:
            try:
                dur = float(args[2])
            except ValueError:
                streams = [s.strip() for s in args[2].split(",")]
        if len(args) >= 4:
            streams = [s.strip() for s in args[3].split(",")]
            if len(args) >= 5:
                comp_arg = args[4].strip().lower()
                chan_arg = (args[5].strip().lower() if len(args) >= 6 else "bh")
                streams = [f"{st.lower()}-{chan_arg}-{c}" if "-" not in st else st for st in streams for c in comp_arg]

        if os.path.isdir(path):
            sfd_cand = os.path.join(path, "sfdlist.sfd")
            if os.path.isfile(sfd_cand):
                trs = read_from_sfd(sfd_cand, streams=streams, t_start=t_start, duration=dur, config=self.config)
            else:
                trs = []
                for fn in sorted(os.listdir(path)):
                    fp = os.path.join(path, fn)
                    if os.path.isfile(fp) and not fn.endswith(".sfd"):
                        trs.extend(read_mseed_file(fp, t_start=t_start, duration=dur, config=self.config))
        elif path.lower().endswith(".sfd"):
            trs = read_from_sfd(path, streams=streams, t_start=t_start, duration=dur, config=self.config)
        else:
            trs = read_mseed_file(path, t_start=t_start, duration=dur, config=self.config)

        self.traces.extend(trs)
        msg = f"Read {len(trs)} Mini-SEED trace(s) from {path}"
        self.emit(msg)
        return msg

    def _cmd_writeseed(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITESEED requires an output filepath")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        write_mseed_file(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to Mini-SEED {fpath}"
        self.emit(msg)
        return msg

    def _cmd_readgse(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READGSE requires a GSE file path")
        fpath = args[0]
        trs = read_gse_file(fpath, config=self.config)
        self.traces.extend(trs)
        msg = f"Read {len(trs)} GSE trace(s) from {fpath}"
        self.emit(msg)
        return msg

    def _cmd_writegse(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEGSE requires a GSE output path")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        write_gse2_file(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to GSE2 file {fpath}"
        self.emit(msg)
        return msg

    def _cmd_reada(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READA requires an ASCII file path")
        fpath = args[0]
        trs = read_ascii_file(fpath, config=self.config)
        self.traces.extend(trs)
        msg = f"Read {len(trs)} ASCII trace(s) from {fpath}"
        self.emit(msg)
        return msg

    def _cmd_writea(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEA requires an ASCII output path")
        fpath = args[0]
        idxs = self.parse_trace_list(args[1] if len(args) > 1 else "ALL")
        trs = [self.traces[i] for i in idxs]
        write_ascii_file(fpath, trs)
        msg = f"Wrote {len(trs)} trace(s) to ASCII file {fpath}"
        self.emit(msg)
        return msg

    def _cmd_readevt(self, args: List[str]) -> str:
        if not args:
            raise ValueError("READEVT requires an .evt file path")
        param, picks = read_evt_file(args[0])
        self.param_sets[self.active_set_index] = param
        # Attach picks to matching traces
        for p in picks:
            for tr in self.traces:
                if tr.station.upper() == p.station.upper() and (not p.comp or tr.comp.upper() == p.comp.upper()):
                    tr.add_or_replace_pick(p)
                    break
        summary = format_ev2view_summary(param, picks)
        self.emit(summary)
        return summary

    def _cmd_writeevt(self, args: List[str]) -> str:
        if not args:
            raise ValueError("WRITEEVT requires an .evt output path")
        all_picks: List[PhasePick] = []
        for tr in self.traces:
            all_picks.extend(tr.picks)
        write_evt_file(args[0], self.current_params, all_picks)
        msg = f"Wrote event parameters ({len(all_picks)} picks) to {args[0]}"
        self.emit(msg)
        return msg

    def _cmd_readsxml(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("READSXML requires a StationXML file path")
        fpath = args[0]
        inv = self.config.load_stationxml(fpath)
        self.last_stationxml_inventory = inv
        n_sta = len(inv.stations)
        n_cha = sum(len(s.channels) for s in inv.stations)
        # Optionally enrich existing workspace traces that lack coordinates
        for tr in self.traces:
            st = self.config.get_station(tr.station)
            if st is not None and (st.lat != 0.0 or st.lon != 0.0) and tr.lat == 0.0 and tr.lon == 0.0:
                tr.lat = st.lat
                tr.lon = st.lon
        msg = f"Loaded StationXML from {fpath}: {n_sta} station(s), {n_cha} channel(s)."
        self.emit(msg)
        return msg

    def _cmd_writeqml(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEQML requires an output filepath")
        fpath = args[0]
        list_spec = args[1] if len(args) > 1 else quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        trs = [self.traces[i] for i in idxs]
        prefix = quals.get("PREFIX", "smi:shm-modern")
        write_quakeml(
            fpath,
            trs,
            self.current_params,
            public_id_prefix=prefix,
            focal_mechanism=self.last_focal_mechanism,
        )
        msg = f"Wrote QuakeML 1.2 ({len(trs)} trace(s)) to {fpath}"
        self.emit(msg)
        return msg

    def _cmd_readqml(self, args: List[str]) -> str:
        if not args:
            raise ValueError("READQML requires a QuakeML file path")
        fpath = args[0]
        qml = parse_quakeml(fpath)
        param = qml["params"]
        picks = qml.get("phase_picks", [])
        fms = qml.get("focal_mechanisms", [])
        if fms:
            fm0 = fms[0]
            np1 = fm0.get("nodalPlane1")
            if isinstance(np1, dict):
                fm_dict: Dict[str, Any] = {
                    "strike": float(np1.get("strike", 0.0)),
                    "dip": float(np1.get("dip", 0.0)),
                    "rake": float(np1.get("rake", 0.0)),
                }
                np2 = fm0.get("nodalPlane2")
                if isinstance(np2, dict):
                    fm_dict["strike2"] = float(np2.get("strike", 0.0))
                    fm_dict["dip2"] = float(np2.get("dip", 0.0))
                    fm_dict["rake2"] = float(np2.get("rake", 0.0))
                self.last_focal_mechanism = fm_dict
                setattr(param, "focal_mechanism", fm_dict)

        self.param_sets[self.active_set_index] = param
        for p in picks:
            for tr in self.traces:
                if tr.station.upper() == p.station.upper() and (not p.comp or tr.comp.upper() == p.comp.upper()):
                    tr.add_or_replace_pick(p)
                    break
            else:
                for tr in self.traces:
                    if tr.station.upper() == p.station.upper():
                        tr.add_or_replace_pick(p)
                        break

        msg = (
            f"Read QuakeML {qml.get('schema_version', '1.2')} from {fpath}: "
            f"Lat={param.lat:+.3f}, Lon={param.lon:+.3f}, Depth={param.depth:.1f} km, "
            f"{len(picks)} pick(s)."
        )
        self.emit(msg)
        return msg

    def _cmd_writegeojson(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("WRITEGEOJSON requires an output filepath")
        fpath = args[0]
        list_spec = args[1] if len(args) > 1 else quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        trs = [self.traces[i] for i in idxs]
        write_geojson(
            fpath,
            trs,
            self.current_params,
            include_stations=("NOSTATIONS" not in quals),
            include_ellipse=("NOELLIPSE" not in quals),
            focal_mechanism=self.last_focal_mechanism,
        )
        msg = f"Wrote GeoJSON ({len(trs)} trace(s)) to {fpath}"
        self.emit(msg)
        return msg

    def _cmd_report(self, args: List[str], quals: Dict[str, str]) -> str:
        if not args:
            raise ValueError("REPORT requires an output filepath (.pdf or .svg)")
        fpath = args[0]
        list_spec = args[1] if len(args) > 1 else quals.get("LIST", "ALL")
        idxs = self.parse_trace_list(list_spec)
        trs = [self.traces[i] for i in idxs]
        fmt = quals.get("FORMAT", quals.get("FMT"))
        if fmt is None:
            if "SVG" in quals:
                fmt = "svg"
            elif "PDF" in quals:
                fmt = "pdf"
        title = quals.get("TITLE", self.title)
        agency = quals.get("AGENCY", "SHM-Modern Observatory")

        from shm_app.gui.reporting import generate_observatory_bulletin

        out_path = generate_observatory_bulletin(
            output_path=fpath,
            traces=trs,
            params=self.current_params,
            fit_result=self.last_fit_diagnostic,
            focal_mech=self.last_focal_mechanism,
            title=title,
            format=fmt,
            agency=agency,
            config=self.config,
        )
        msg = f"Generated observatory bulletin report: {out_path}"
        self.emit(msg)
        return msg

    def _cmd_sfd(self, args: List[str]) -> str:
        dpath = args[0] if args else "."
        out_sfd = args[1] if len(args) > 1 else os.path.join(dpath, "sfdlist.sfd")
        recs = generate_sfd_file(dpath, out_sfd, config=self.config)
        msg = f"Generated {len(recs)} SFD index entries in {out_sfd}"
        self.emit(msg)
        return msg

    def _cmd_fetch(self, args: List[str], quals: Dict[str, str]) -> str:
        from shm_app.plugins.bgr_fetch import (
            BGR_FDSN_DATASELECT_URL,
            DEFAULT_DOWNLOAD_DIR,
            expand_channels,
            fetch_bgr_waveforms,
            parse_station_list,
        )

        if len(args) >= 5:
            network = args[0]
            stations = parse_station_list(args[1])
            channels = expand_channels(args[2])
            start_str = args[3]
            dur_s = float(args[4])
        elif len(args) == 4:
            network = quals.get("NET", quals.get("NETWORK", "GR"))
            stations = parse_station_list(args[0])
            channels = expand_channels(args[1])
            start_str = args[2]
            dur_s = float(args[3])
        elif len(args) == 3:
            network = quals.get("NET", quals.get("NETWORK", "GR"))
            stations = parse_station_list(args[0])
            channels = expand_channels(quals.get("CHA", quals.get("CHAN", "BH")))
            start_str = args[1]
            dur_s = float(args[2])
        else:
            raise ValueError(
                "FETCH usage: FETCH /BGR <network> <stations> <channels> <start_time> <duration>"
            )

        out_dir = quals.get("OUT", DEFAULT_DOWNLOAD_DIR)
        loc = quals.get("LOC", quals.get("LOCATION", "*"))
        base_url = quals.get("URL", BGR_FDSN_DATASELECT_URL)
        append = "APPEND" in quals
        build_sfd = "NOSFD" not in quals

        summary = fetch_bgr_waveforms(
            stations=stations,
            start_time=start_str,
            duration=dur_s,
            output_dir=out_dir,
            network=network,
            channels=channels,
            location=loc,
            base_url=base_url,
            build_sfd=build_sfd,
            config=self.config,
        )
        if append:
            self.traces.extend(summary.traces)
        elif summary.traces:
            self.traces = list(summary.traces)
        ok_cnt = sum(1 for r in summary.results if r.status == "ok")
        msg = (
            f"Fetched {len(summary.traces)} Mini-SEED trace(s) from {ok_cnt}/{len(summary.results)} "
            f"BGR station(s) into {summary.output_dir}"
        )
        self.emit(msg)
        return msg

