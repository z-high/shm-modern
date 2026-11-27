"""Time parsing, formatting, and arithmetic matching SH/SHM tc_* routines."""

from __future__ import annotations

import calendar
import datetime
import math
import re
from typing import Tuple

MONTH_NAMES = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]
MONTH_MAP = {name: idx + 1 for idx, name in enumerate(MONTH_NAMES)}


def is_leap_year(year: int) -> bool:
    """Return True if year is a leap year."""
    return calendar.isleap(year)


def julian_to_month_day(year: int, jday: int) -> Tuple[int, int]:
    """Convert (year, julian_day 1..366) to (month 1..12, day 1..31)."""
    base = datetime.date(year, 1, 1) + datetime.timedelta(days=max(0, jday - 1))
    return base.month, base.day


def month_day_to_julian(year: int, month: int, day: int) -> int:
    """Convert (year, month, day) to julian day 1..366."""
    dt = datetime.date(year, month, day)
    return dt.timetuple().tm_yday


def parse_sh_time(timestr: str) -> float:
    """Parse an SHM timestamp or ISO/GSE timestamp into a UTC POSIX epoch float.

    Supported formats:
    - DD-MMM-YYYY_HH:MM:SS.SSS (or space instead of '_', optional time/fractional part)
    - DD-MM-YYYY_HH:MM:SS.SSS
    - YYYY/MM/DD HH:MM:SS.SSS (GSE2)
    - YYYY-MM-DDTHH:MM:SS.SSS or YYYY-MM-DD HH:MM:SS.SSS (ISO)
    - YYYYDDD HH MM SS MMM (GSE1 WID1)
    """
    if not timestr or not timestr.strip():
        return 0.0
    s = timestr.strip()

    # Try SH standard format: DD-MMM-YYYY[_ ]HH:MM:SS.SSS
    m = re.match(
        r"^(\d{1,2})-([A-Za-z]{3}|\d{1,2})-(\d{2,4})(?:[_ T]+(\d{1,2})(?::(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?)?)?$",
        s,
    )
    if m:
        day = int(m.group(1))
        mon_raw = m.group(2).upper()
        if mon_raw.isdigit():
            month = int(mon_raw)
        else:
            if mon_raw not in MONTH_MAP:
                raise ValueError(f"Unrecognized month token {mon_raw!r} in SH time string: {timestr!r}")
            month = MONTH_MAP[mon_raw]
        year = int(m.group(3))
        if year < 100:
            year += 1900 if year >= 60 else 2000
        hour = int(m.group(4)) if m.group(4) is not None else 0
        minute = int(m.group(5)) if m.group(5) is not None else 0
        sec_float = float(m.group(6)) if m.group(6) is not None else 0.0
        sec_int = int(math.floor(sec_float))
        frac = sec_float - sec_int
        dt = datetime.datetime(year, month, day, hour, minute, 0, tzinfo=datetime.timezone.utc)
        return dt.timestamp() + sec_int + frac

    # Try ISO / GSE2 format: YYYY[-/]MM[-/]DD[_ T]+HH:MM:SS.SSS
    m2 = re.match(
        r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[_ T]+(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d*)?))?$",
        s,
    )
    if m2:
        year = int(m2.group(1))
        month = int(m2.group(2))
        day = int(m2.group(3))
        hour = int(m2.group(4)) if m2.group(4) else 0
        minute = int(m2.group(5)) if m2.group(5) else 0
        sec_float = float(m2.group(6)) if m2.group(6) else 0.0
        sec_int = int(math.floor(sec_float))
        frac = sec_float - sec_int
        dt = datetime.datetime(year, month, day, hour, minute, 0, tzinfo=datetime.timezone.utc)
        return dt.timestamp() + sec_int + frac

    # Try numeric float directly
    try:
        val = float(s)
    except ValueError as exc:
        raise ValueError(f"Unrecognized SH time string: {timestr!r}") from exc
    if not math.isfinite(val):
        raise ValueError(f"Non-finite numeric time value: {timestr!r}")
    return val


def format_sh_time(epoch: float, millis_digits: int = 3) -> str:
    """Format a UTC POSIX epoch float into SHM standard DD-MMM-YYYY_HH:MM:SS.SSS."""
    ep = float(epoch)
    if not math.isfinite(ep):
        raise ValueError(f"Cannot format non-finite epoch: {epoch!r}")
    # Round to desired fractional precision carefully to avoid 60.000 seconds
    scale = 10 ** millis_digits
    rounded = round(ep * scale) / scale
    dt = datetime.datetime.fromtimestamp(rounded, tz=datetime.timezone.utc)
    mon_str = MONTH_NAMES[dt.month - 1]
    if millis_digits <= 0:
        return f"{dt.day:02d}-{mon_str}-{dt.year:04d}_{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}"
    frac = rounded - math.floor(rounded)
    ms = int(round(frac * scale))
    if ms >= scale:
        ms = 0
    return (
        f"{dt.day:02d}-{mon_str}-{dt.year:04d}_"
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:0{millis_digits}d}"
    )


def format_gse2_time(epoch: float) -> str:
    """Format a UTC POSIX epoch float into GSE2 WID2 format YYYY/MM/DD HH:MM:SS.SSS."""
    ep = float(epoch)
    if not math.isfinite(ep):
        raise ValueError(f"Cannot format non-finite epoch: {epoch!r}")
    rounded = round(ep * 1000.0) / 1000.0
    dt = datetime.datetime.fromtimestamp(rounded, tz=datetime.timezone.utc)
    ms = int(round((rounded - math.floor(rounded)) * 1000.0)) % 1000
    return f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}"


def tc_tadd(timestr: str, offset_sec: float) -> str:
    """Add offset_sec to an SH time string and return a formatted SH time string."""
    return format_sh_time(parse_sh_time(timestr) + float(offset_sec))


def tc_tdiff(t_end: str, t_start: str) -> float:
    """Return (t_end - t_start) in seconds."""
    return parse_sh_time(t_end) - parse_sh_time(t_start)


def ntime_to_epoch(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    ms: float = 0.0,
) -> float:
    """Convert numeric time components to UTC epoch float."""
    dt = datetime.datetime(year, month, day, hour, minute, 0, tzinfo=datetime.timezone.utc)
    return dt.timestamp() + second + ms / 1000.0


def epoch_to_ntime(epoch: float) -> Tuple[int, int, int, int, int, int, int]:
    """Convert UTC epoch float to (year, month, day, hour, minute, second, millisecond)."""
    ep = float(epoch)
    if not math.isfinite(ep):
        raise ValueError(f"Cannot convert non-finite epoch: {epoch!r}")
    rounded = round(ep * 1000.0) / 1000.0
    dt = datetime.datetime.fromtimestamp(rounded, tz=datetime.timezone.utc)
    ms = int(round((rounded - math.floor(rounded)) * 1000.0)) % 1000
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, ms
