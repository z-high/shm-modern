"""Workspace session autosave and crash-recovery restore mixin for SHMMainWindow."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np

from PyQt6.QtWidgets import QMessageBox

from shm_app.core.models import Trace
from shm_app.core.session import SessionManager
from shm_app.io.ascii_io import read_ascii_file
from shm_app.io.gse import read_gse_file
from shm_app.io.mseed import read_mseed_file
from shm_app.io.qfile import read_qfile
from shm_app.processing.filters import filter_trace


class _MainWindowSessionMixin:
    """Session autosave and crash recovery."""

    def autosave_session(self, clean_exit: bool = False) -> Optional[str]:
        disp = {
            "norm_mode": self.main_canvas.norm_mode,
            "global_zoom": self.main_canvas.global_zoom,
            "view_t_lo": self.main_canvas.view_t_lo,
            "view_t_hi": self.main_canvas.view_t_hi,
            "mag_t_lo": self.main_canvas.mag_t_lo,
            "mag_t_hi": self.main_canvas.mag_t_hi,
            "min_slot_h": self.main_canvas.min_slot_h,
            "group_by_station": self.main_canvas.group_by_station,
            "show_pick_error_bars": self.main_canvas.show_pick_error_bars,
            "cursor_overlay_mode": self.mag_canvas.cursor_overlay_mode,
            "time_window": list(self.engine.time_window) if self.engine.time_window is not None else None,
        }
        sel_idxs = [i for i, t in enumerate(self.engine.traces) if t.selected]
        saved = self.session_manager.save(
            traces=self.engine.traces,
            param_sets=self.engine.param_sets,
            active_set_index=self.engine.active_set_index,
            selected_trace_indices=sel_idxs,
            active_trace_index=self.main_canvas.mag_trace_idx,
            display_settings=disp,
            clean_exit=clean_exit,
        )
        self._has_autosaved = True
        return saved

    def _on_autosave_timer(self) -> None:
        if self.engine.traces or self._explicit_session_path or self._has_autosaved:
            try:
                self.autosave_session(clean_exit=False)
            except Exception:
                pass

    def _load_trace_data_for_session(self, raw_tr: Dict[str, Any]) -> Optional[np.ndarray]:
        src = str(raw_tr.get("source_path") or "")
        if not src or not os.path.isfile(src):
            return None
        sta = str(raw_tr.get("station") or "").upper()
        ch1 = str(raw_tr.get("chan1") or "B").upper()
        ch2 = str(raw_tr.get("chan2") or "H").upper()
        cp = str(raw_tr.get("comp") or "Z").upper()
        ch3 = f"{ch1}{ch2}{cp}"
        low = src.lower()
        try:
            if low.endswith(".gse"):
                trs = read_gse_file(src, config=self.config)
            elif low.endswith((".qhd", ".qbn")):
                trs = read_qfile(src, config=self.config)
            elif low.endswith((".asc", ".txt")):
                trs = read_ascii_file(src, config=self.config)
            else:
                trs = read_mseed_file(src, config=self.config)
        except Exception:
            return None
        for cand in trs:
            if cand.station.upper() == sta and cand.channel3.upper() == ch3:
                return cand.data
        for cand in trs:
            if cand.station.upper() == sta:
                return cand.data
        return trs[0].data if trs else None

    def restore_session(self, session_path: Optional[Union[str, os.PathLike]] = None) -> bool:
        mgr = SessionManager(session_path) if session_path is not None else self.session_manager
        self._last_session_restore_error = ""
        self._last_session_restore_warnings: List[str] = []
        disk_raw_cache: Dict[Tuple[str, str, str, float], np.ndarray] = {}
        missing_sources: List[str] = []

        def _session_trace_loader(raw_tr: Dict[str, Any]) -> Optional[np.ndarray]:
            arr = self._load_trace_data_for_session(raw_tr)
            src = str(raw_tr.get("source_path") or "")
            sta = str(raw_tr.get("station") or "").upper()
            ch3 = (
                f"{str(raw_tr.get('chan1') or 'B').upper()}"
                f"{str(raw_tr.get('chan2') or 'H').upper()}"
                f"{str(raw_tr.get('comp') or 'Z').upper()}"
            )
            ep = float(raw_tr.get("start_epoch") or 0.0)
            if arr is not None and len(arr) > 0:
                disk_raw_cache[(src, sta, ch3, ep)] = np.asarray(arr, dtype=np.float64)
            elif src:
                missing_sources.append(f"{sta}.{ch3} ({src})")
            return arr

        res = mgr.load(trace_loader=_session_trace_loader, raise_on_error=False)
        if not res.get("ok", False):
            err = res.get("error") or "Unable to restore session."
            self._last_session_restore_error = str(err)
            self._append_console_log(f"Session restore skipped: {err}")
            return False

        raw_loaded_traces = list(res.get("traces") or [])
        kept_traces: List[Trace] = []
        pruned_traces: List[str] = []
        for tr in raw_loaded_traces:
            if tr.length == 0:
                label = f"{tr.station}.{tr.channel3}"
                if tr.source_path:
                    label = f"{label} ({tr.source_path})"
                pruned_traces.append(label)
                continue
            kept_traces.append(tr)

        if pruned_traces:
            warn_msg = (
                f"Pruned {len(pruned_traces)} trace(s) with missing or unreadable source files: "
                + ", ".join(pruned_traces[:5])
            )
            self._last_session_restore_warnings.append(warn_msg)
            self._append_console_log(f"Session restore warning: {warn_msg}")
            self.status_bar.showMessage(warn_msg, 6000)

        self._unfiltered_by_uid.clear()
        replayed_traces: List[Trace] = []
        for tr in kept_traces:
            uid = str(tr.extra_headers.get("_raw_uid") or "")
            if not uid:
                uid = uuid.uuid4().hex
                tr.extra_headers["_raw_uid"] = uid

            key = (str(tr.source_path or ""), tr.station.upper(), tr.channel3.upper(), float(tr.start_epoch))
            from_disk = key in disk_raw_cache
            if from_disk:
                raw_copy = tr.copy()
                raw_copy.data = disk_raw_cache[key].copy()
                raw_copy.filter_name = "NONE"
                raw_copy.overlay_filter_name = "NONE"
                raw_copy.overlay_filter_color = ""
                self._unfiltered_by_uid[uid] = raw_copy

                cur = tr
                g_flt = cur.filter_name or "NONE"
                ov_flt = cur.overlay_filter_name or "NONE"
                ov_col = cur.overlay_filter_color or ""
                hist = list(cur.filter_history)
                if g_flt not in ("", "NONE"):
                    try:
                        cur = filter_trace(cur, g_flt, config=self.config)
                        cur.filter_name = g_flt
                        cur.overlay_filter_name = ov_flt
                        cur.overlay_filter_color = ov_col
                        cur.filter_history = hist
                    except Exception:
                        pass
                if ov_flt not in ("", "NONE"):
                    try:
                        cur = filter_trace(cur, ov_flt, config=self.config)
                        cur.filter_name = g_flt
                        cur.overlay_filter_name = ov_flt
                        cur.overlay_filter_color = ov_col
                        cur.filter_history = hist
                    except Exception:
                        pass
                replayed_traces.append(cur)
            else:
                # Inline waveform (e.g. filtered/modified before autosave): register raw disk copy if still available
                if tr.source_path and tr.filter_name not in ("", "NONE"):
                    raw_disk = self._load_trace_data_for_session(
                        {
                            "source_path": tr.source_path,
                            "station": tr.station,
                            "chan1": tr.chan1,
                            "chan2": tr.chan2,
                            "comp": tr.comp,
                        }
                    )
                    if raw_disk is not None and len(raw_disk) == tr.length:
                        raw_copy = tr.copy()
                        raw_copy.data = np.asarray(raw_disk, dtype=np.float64).copy()
                        raw_copy.filter_name = "NONE"
                        raw_copy.overlay_filter_name = "NONE"
                        raw_copy.overlay_filter_color = ""
                        self._unfiltered_by_uid[uid] = raw_copy
                replayed_traces.append(tr)

        self.engine.traces = replayed_traces
        ps_list = list(res.get("param_sets") or [])
        for idx, ps in enumerate(ps_list):
            if idx < len(self.engine.param_sets):
                self.engine.param_sets[idx] = ps
            else:
                self.engine.param_sets.append(ps)
        if self.engine.param_sets:
            self.engine.active_set_index = max(
                0, min(len(self.engine.param_sets) - 1, int(res.get("active_set_index", 0)))
            )

        disp = res.get("display_settings") or {}
        if disp.get("norm_mode") in ("A", "C", "AW", "SW"):
            self.engine.norm_mode = disp["norm_mode"]
            self.main_canvas.norm_mode = disp["norm_mode"]
        if disp.get("global_zoom") is not None:
            gz = max(0.05, min(200.0, float(disp["global_zoom"])))
            self.engine.global_zoom = gz
            self.main_canvas.global_zoom = gz
        tw = disp.get("time_window")
        if isinstance(tw, (list, tuple)) and len(tw) == 2 and tw[0] is not None and tw[1] is not None:
            self.engine.time_window = (float(tw[0]), float(tw[1]))
        else:
            self.engine.time_window = None
        v_lo = disp.get("view_t_lo")
        v_hi = disp.get("view_t_hi")
        if v_lo is not None and v_hi is not None:
            self.engine.view_window = (float(v_lo), float(v_hi))
            self.main_canvas.view_t_lo = float(v_lo)
            self.main_canvas.view_t_hi = float(v_hi)
            self._synced_view_window_version = self.engine._view_window_version
        if disp.get("mag_t_lo") is not None and disp.get("mag_t_hi") is not None:
            self.main_canvas.mag_t_lo = float(disp["mag_t_lo"])
            self.main_canvas.mag_t_hi = float(disp["mag_t_hi"])
        if disp.get("min_slot_h") is not None:
            self._set_row_height(int(disp["min_slot_h"]))
        if "group_by_station" in disp:
            grp_on = bool(disp["group_by_station"])
            if hasattr(self, "act_group"):
                self.act_group.setChecked(grp_on)
            self.main_canvas.set_group_by_station(grp_on)
        if "show_pick_error_bars" in disp:
            eb_on = bool(disp["show_pick_error_bars"])
            if hasattr(self, "act_pick_error_bars"):
                self.act_pick_error_bars.setChecked(eb_on)
            self.main_canvas.set_show_pick_error_bars(eb_on)
            self.mag_canvas.set_show_pick_error_bars(eb_on)
        if disp.get("cursor_overlay_mode"):
            self.mag_canvas.set_cursor_overlay_mode(str(disp["cursor_overlay_mode"]))
        self.main_canvas.mag_trace_idx = int(res.get("active_trace_index", 0))

        self._register_unfiltered_traces(self.engine.traces, replace=False, overwrite_existing=False)
        self._sync_unfiltered_backup_view()
        self.refresh_views(reset_time=False)
        msg = f"Restored workspace session ({len(self.engine.traces)} trace(s)) from {mgr.session_path}"
        self._append_console_log(msg)
        self.status_bar.showMessage(msg, 5000)
        return True

    def action_restore_session(self) -> None:
        if not self.restore_session():
            err = getattr(self, "_last_session_restore_error", "") or "No valid saved session found to restore."
            QMessageBox.warning(self, "Restore Session", err)
            self.status_bar.showMessage("No valid saved session found to restore.", 4000)
        elif getattr(self, "_last_session_restore_warnings", None):
            QMessageBox.warning(
                self,
                "Restore Session Warnings",
                "\n".join(self._last_session_restore_warnings),
            )
