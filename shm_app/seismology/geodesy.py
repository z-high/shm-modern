"""Spherical geodesic forward and inverse functions matching earthloc.c."""

from __future__ import annotations

import math
from typing import Tuple


def sphere_add(lat_deg: float, lon_deg: float, dist_deg: float, azim_deg: float) -> Tuple[float, float]:
    """Compute target (elat, elon) from starting (slat, slon), angular distance dist_deg,
    and azimuth azim_deg matching mb_sphereadd in earthloc.c.
    """
    slat = 90.0 - float(lat_deg)
    slon = float(lon_deg)
    if slon < 0.0:
        slon += 360.0

    bazim = float(azim_deg) % 360.0
    invert = bazim > 180.0
    if invert:
        bazim = 360.0 - bazim

    slat_r = math.radians(slat)
    dist_r = math.radians(float(dist_deg))
    bazim_r = math.radians(bazim)

    arg = math.cos(dist_r) * math.cos(slat_r) + math.sin(dist_r) * math.sin(slat_r) * math.cos(bazim_r)
    arg = max(-1.0, min(1.0, arg))
    elat_r = math.acos(arg)

    denom = math.sin(elat_r) * math.sin(slat_r)
    if abs(denom) < 1e-12:
        gamma_r = 0.0
    else:
        arg2 = (math.cos(dist_r) - math.cos(elat_r) * math.cos(slat_r)) / denom
        arg2 = max(-1.0, min(1.0, arg2))
        gamma_r = math.acos(arg2)

    elat = math.degrees(elat_r)
    gamma = math.degrees(gamma_r)
    if invert:
        gamma = 360.0 - gamma
    elon = slon + gamma
    if elon > 360.0:
        elon -= 360.0

    elat = 90.0 - elat
    if elon > 180.0:
        elon -= 360.0
    return elat, elon


def sphere_diff(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> Tuple[float, float, float]:
    """Compute (distance_deg, azim_1_to_2_deg, backazim_2_to_1_deg) between two geographic coordinates
    matching mb_spherediff in earthloc.c.
    """
    def _one_way(s_lat: float, s_lon: float, e_lat: float, e_lon: float) -> Tuple[float, float]:
        slat = 90.0 - float(s_lat)
        elat = 90.0 - float(e_lat)
        slon = float(s_lon) + 360.0 if s_lon < 0.0 else float(s_lon)
        elon = float(e_lon) + 360.0 if e_lon < 0.0 else float(e_lon)

        gamma = elon - slon
        if gamma < 0.0:
            gamma += 360.0
        invert = gamma > 180.0
        if invert:
            gamma = 360.0 - gamma

        slat_r = math.radians(slat)
        elat_r = math.radians(elat)
        gamma_r = math.radians(gamma)

        arg = math.cos(elat_r) * math.cos(slat_r) + math.sin(elat_r) * math.sin(slat_r) * math.cos(gamma_r)
        arg = max(-1.0, min(1.0, arg))
        dist_r = math.acos(arg)
        if dist_r < 1e-9:
            return 0.0, 0.0

        dist_deg = math.degrees(dist_r)
        if abs(math.sin(slat_r)) < 1e-12:
            # At North Pole (s_lat=+90), every direction is South (180); at South Pole, North (0)
            return dist_deg, (180.0 if float(s_lat) >= 0.0 else 0.0)

        denom = math.sin(slat_r) * math.sin(dist_r)
        if abs(denom) < 1e-12:
            baz_r = 0.0
        else:
            arg2 = (math.cos(elat_r) - math.cos(slat_r) * math.cos(dist_r)) / denom
            arg2 = max(-1.0, min(1.0, arg2))
            baz_r = math.acos(arg2)

        baz_deg = math.degrees(baz_r)
        if invert:
            baz_deg = 360.0 - baz_deg
        return dist_deg, baz_deg % 360.0

    dist_12, az_12 = _one_way(lat1_deg, lon1_deg, lat2_deg, lon2_deg)
    _, az_21 = _one_way(lat2_deg, lon2_deg, lat1_deg, lon1_deg)
    return dist_12, az_12, az_21
