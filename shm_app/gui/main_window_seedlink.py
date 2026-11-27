"""SeedLink live waveform streaming actions and handlers for SHMMainWindow."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shm_app.core.models import Trace
from shm_app.gui.dialog_seedlink import SeedLinkStreamDialog


class _MainWindowSeedLinkMixin:
    """SeedLink v3.1 streaming actions and trace synchronization."""

    def action_open_seedlink_dialog(self) -> None:
        existing = self._find_open_dialog(SeedLinkStreamDialog)
        if existing is not None:
            self._raise_nonmodal(existing)
            return
        dlg = SeedLinkStreamDialog(self.live_coordinator, parent=self)
        self._show_nonmodal(dlg)

    def start_seedlink_stream(
        self,
        host: str,
        port: int = 18000,
        streams: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ):
        norm_streams: Optional[List[Any]] = None
        if streams is not None:
            norm_streams = []
            for item in streams:
                if isinstance(item, str):
                    tok = item.strip()
                    if ":" in tok:
                        left, sel = tok.split(":", 1)
                        parts = [p for p in left.replace(".", "_").split("_") if p]
                        if len(parts) >= 2:
                            sel_norm = sel.strip().upper()
                            if "." not in sel_norm:
                                sel_norm = f"{sel_norm}.D"
                            norm_streams.append((parts[0].upper(), parts[1].upper(), sel_norm))
                            continue
                    if "." in tok:
                        parts = tok.split(".")
                        if len(parts) >= 4:
                            loc = parts[2].strip().upper()
                            cha = parts[3].strip().upper() or "BHZ"
                            sel = f"{loc}{cha}.D" if loc else f"{cha}.D"
                            norm_streams.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                            continue
                        if len(parts) >= 2:
                            sel = f"{parts[2].strip().upper()}.D" if len(parts) == 3 and parts[2].strip() else "BHZ.D"
                            norm_streams.append((parts[0].strip().upper(), parts[1].strip().upper(), sel))
                            continue
                    if "_" in tok:
                        parts = [p for p in tok.split("_") if p]
                        if len(parts) >= 2:
                            sel = f"{parts[2].upper()}.D" if len(parts) >= 3 else "BHZ.D"
                            norm_streams.append((parts[0].upper(), parts[1].upper(), sel))
                            continue
                norm_streams.append(item)

        worker = self.live_coordinator.start_seedlink(
            host=host,
            port=port,
            streams=norm_streams,
            **kwargs,
        )
        if hasattr(self, "act_seedlink_stop"):
            self.act_seedlink_stop.setEnabled(True)
        return worker

    def action_stop_seedlink(self) -> None:
        self.live_coordinator.stop_seedlink()
        if hasattr(self, "act_seedlink_stop"):
            self.act_seedlink_stop.setEnabled(False)
        self.status_bar.showMessage("SeedLink stream stopped.", 4000)

    def _on_seedlink_traces_updated(self, stream_traces: list) -> None:
        if not stream_traces:
            return
        was_empty = len(self.engine.traces) == 0
        existing_picks: Dict[Tuple[str, str, str, str], List[Any]] = {}
        kept_traces: List[Trace] = []
        for tr in self.engine.traces:
            key = (tr.network.upper(), tr.station.upper(), tr.locid.upper(), tr.channel3.upper())
            if tr.picks:
                existing_picks[key] = list(tr.picks)
            if not tr.extra_headers.get("seedlink_stream"):
                kept_traces.append(tr)

        sl_traces: List[Trace] = []
        for tr in stream_traces:
            tr_copy = tr.copy()
            tr_copy.extra_headers["seedlink_stream"] = True
            key = (tr_copy.network.upper(), tr_copy.station.upper(), tr_copy.locid.upper(), tr_copy.channel3.upper())
            if not tr_copy.picks and key in existing_picks:
                tr_copy.picks = copy.deepcopy(existing_picks[key])
            sl_traces.append(tr_copy)

        self.engine.traces = kept_traces + sl_traces
        self._register_unfiltered_traces(sl_traces, replace=False, overwrite_existing=False)
        self.refresh_views(reset_time=was_empty)

    def _on_seedlink_status_changed(self, status: str) -> None:
        running = bool(self.live_coordinator.is_seedlink_running)
        if hasattr(self, "act_seedlink_stop"):
            self.act_seedlink_stop.setEnabled(running)
        self.status_bar.showMessage(f"SeedLink: {status}", 3000)

    def _on_seedlink_error(self, err: str) -> None:
        self.status_bar.showMessage(f"SeedLink warning: {err}", 4000)
