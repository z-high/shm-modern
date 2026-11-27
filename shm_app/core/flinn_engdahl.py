"""Flinn-Engdahl (1995) geographical and seismic region lookup tables."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from shm_app import SHM_DEFAULT_ROOT


class FlinnEngdahlLookup:
    """Flinn-Engdahl (1995) Geographical and Seismic Region Lookup matching earthloc.c & fereg.dat/fername.dat."""

    def __init__(self, inputs_dir: Optional[str] = None, shm_root: Optional[str] = None) -> None:
        self.shm_root = shm_root or (
            os.path.dirname(os.path.normpath(inputs_dir)) if inputs_dir else SHM_DEFAULT_ROOT
        )
        self.inputs_dir = inputs_dir or os.path.join(self.shm_root, "inputs")
        self.greg_names: Dict[int, str] = {}
        self.sreg_names: Dict[int, str] = {}
        self.nsreg: List[int] = []
        self.quad_ind: Dict[int, int] = {}
        self.quad_num: Dict[int, int] = {}
        self.lln: List[Tuple[int, int, int]] = []
        self._loaded = False
        self._load()

    def _load(self) -> None:
        for cand_fn in ("FERNAME.DAT", "fername.dat"):
            fername_path = os.path.join(self.inputs_dir, cand_fn)
            if os.path.isfile(fername_path):
                with open(fername_path, "r", encoding="latin-1", errors="ignore") as f:
                    for line in f:
                        line_s = line.strip()
                        if not line_s or line_s.startswith("!"):
                            continue
                        parts = line_s.split(None, 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            self.greg_names[int(parts[0])] = parts[1].strip()
                break

        # Parse earthloc.c tables directly so all 757 geographical regions, 50 seismic regions,
        # and the exact 1995 quadrant lat/lon grid are always available.
        earthloc_candidates = [
            os.path.join(self.shm_root, "source", "earthloc.c"),
            os.path.join(SHM_DEFAULT_ROOT, "source", "earthloc.c"),
        ]
        earthloc_path = next((p for p in earthloc_candidates if os.path.isfile(p)), "")
        if earthloc_path:
            with open(earthloc_path, "r", encoding="latin-1", errors="ignore") as f:
                src = f.read()

            # Extract lln[6060][3]
            m_lln = re.search(r"lln\[\d+\]\[3\]\s*=\s*\{(.*?)\};", src, re.DOTALL)
            if m_lln:
                nums = [int(x) for x in re.findall(r"-?\d+", m_lln.group(1))]
                self.lln = [
                    (nums[i], nums[i + 1], nums[i + 2])
                    for i in range(0, len(nums) - 2, 3)
                ]

            # Extract nsreg[759]
            m_ns = re.search(r"nsreg\[\d+\]\s*=\s*\{(.*?)\};", src, re.DOTALL)
            if m_ns:
                self.nsreg = [int(x) for x in re.findall(r"-?\d+", m_ns.group(1))]

            # Extract sreg[]
            m_sreg = re.search(r"static char \*sreg\[\]\s*=\s*\{(.*?)\};", src, re.DOTALL)
            if m_sreg:
                strs = re.findall(r'"([^"]*)"', m_sreg.group(1))
                for idx, val in enumerate(strs):
                    self.sreg_names[idx] = val.strip()

            # Extract greg[]
            m_greg = re.search(r"static char \*greg\[\]\s*=\s*\{(.*?)\};", src, re.DOTALL)
            if m_greg:
                strs = re.findall(r'"([^"]*)"', m_greg.group(1))
                for idx, val in enumerate(strs):
                    if idx not in self.greg_names or not self.greg_names[idx]:
                        self.greg_names[idx] = val.strip()

        self.quad_ind = {1: 0, 2: 1919, 3: 3535, 4: 4747}
        self.quad_num = {1: 1914, 2: 1579, 3: 1161, 4: 1304}
        self._loaded = True

    def lookup(self, lat: float, lon: float) -> Tuple[int, str, int, str]:
        """Return (igreg, grname, isreg, srname) for (lat, lon) in degrees."""
        if not self.lln:
            return 0, "Unknown Region", 0, ""

        alat = float(lat)
        alng = float(lon)
        while alng > 180.0:
            alng -= 360.0
        while alng < -180.0:
            alng += 360.0

        if alat >= 0.0 and alng >= 0.0:
            quad = 1
        elif alat >= 0.0 and alng < 0.0:
            quad = 2
        elif alat < 0.0 and alng < 0.0:
            quad = 3
        else:
            quad = 4

        ilat = min(90, abs(int(alat)))
        ilng = min(180, abs(int(alng)))

        n = self.quad_ind[quad]
        qnum = self.quad_num[quad]
        i = 0
        while i < qnum and ilat > self.lln[i + n][0]:
            i += 1
        igreg = 0
        if i < qnum:
            j = i
            while j < qnum and ilat == self.lln[j + n][0] and ilng >= self.lln[j + n][1]:
                j += 1
            igreg = self.lln[max(0, j - 1) + n][2]

        if igreg <= 0 or igreg >= len(self.nsreg):
            igreg = 0
        isreg = self.nsreg[igreg] if 0 <= igreg < len(self.nsreg) else 0
        grname = self.greg_names.get(igreg, "")
        srname = self.sreg_names.get(isreg, "")
        return igreg, grname, isreg, srname

    def region_by_id(self, igreg: int) -> Tuple[str, int, str]:
        """Return (grname, isreg, srname) for geographical region number igreg."""
        isreg = self.nsreg[igreg] if 0 <= igreg < len(self.nsreg) else 0
        return self.greg_names.get(igreg, ""), isreg, self.sreg_names.get(isreg, "")
