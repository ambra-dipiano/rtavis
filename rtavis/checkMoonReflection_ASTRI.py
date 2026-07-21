"""
ASTRI Moon reflection veto: safe observing windows and sky views.

Given a source (pointing) and an observing night (astronomical dusk → dawn),
reports when the Moon is NOT in the M1 or M2 reflection zones on the celestial
sphere, and optionally saves Aitoff (RA/Dec) and polar (Alt/Az) sky snapshots.

Reflection zones (angular separation from pointing):
  M1 — cap within ``m1-radius-deg`` (default 30°)
  M2 — annulus around the antipode: ``m2-sep-min``–``m2-sep-max`` (default 125–170°)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.coordinates.name_resolve import NameResolveError
from astropy.time import Time
from typing import List, Optional, Sequence, Tuple


OBSERVATORY_ASTRI = EarthLocation(lat=28.2983 * u.deg, lon=-16.5097 * u.deg, height=2390 * u.m)

SOURCE_COLOR = "black"
FOV_EDGE_COLOR = "black"
MOON_COLOR = "#5dade2"  
M1_ZONE_COLOR = "#e63946"
M2_ZONE_COLOR = "#7b2cbf"
M1_ZONE_ALPHA = 0.65
M2_ZONE_ALPHA = 0.65
BELOW_HORIZON_COLOR = "#4a5568"
BELOW_HORIZON_ALPHA = 0.55
SAFE_SKY_COLOR = "#0d1b2a"
GRID_COLOR = "#aaaaaa"


def _parse_time(s: str) -> Time:
    return Time(s.replace(" ", "T"), scale="utc")


def _format_time_print(t: Time) -> str:
    return t.iso.split(".")[0]


def _parse_step(step: str) -> u.Quantity:
    s = step.strip().lower().replace(" ", "")
    if not s:
        raise ValueError("Empty --step")
    if s.endswith("m") and not s.endswith("min"):
        s = s[:-1] + "min"
    q = u.Quantity(s)
    if not q.unit.is_equivalent(u.s):
        raise ValueError(f"--step must be a time quantity (e.g. 10m, 30s, 2h). Got: {step!r}")
    return q


def _resolve_coord(
    name: Optional[str],
    ra: Optional[str],
    dec: Optional[str],
    label_default: str,
) -> Tuple[SkyCoord, str]:
    if ra and dec:
        try:
            coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="fk5", equinox="J2000.0")
        except ValueError:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="fk5", equinox="J2000.0")
        return coord, name or label_default
    if name:
        try:
            coord = SkyCoord.from_name(name).transform_to("fk5")
        except NameResolveError as exc:
            raise ValueError(
                f"Could not resolve {name!r} with SkyCoord.from_name (Sesame). "
                "Pass --ra and --dec instead."
            ) from exc
        return coord, name
    raise ValueError("Provide --source (Sesame name) OR both --ra and --dec")


def _moon_illumination_percent(times: Time) -> np.ndarray:
    sun = get_sun(times)
    moon = get_body("moon", times)
    elong = sun.separation(moon).to_value(u.deg)
    return (1 + np.cos(np.radians(180.0 - elong))) / 2.0 * 100.0


def _crossing_indices(values: np.ndarray, threshold: float) -> np.ndarray:
    above = values > threshold
    return np.where(np.diff(above.astype(int)) != 0)[0]


def astronomical_night_window(
    date: str,
    sun_alt_dark_deg: float = -18.0,
    location: EarthLocation = OBSERVATORY_ASTRI,
) -> Tuple[Time, Time]:
    """
    Return (dusk, dawn) UTC for the observing night starting on ``date``.

    Dusk = Sun crosses below ``sun_alt_dark_deg`` in the evening of ``date``.
    Dawn = next Sun crossing above the same threshold.
    """
    day = Time(date.strip().split("T")[0], scale="utc")
    t0 = day - 6 * u.hour
    t1 = day + 36 * u.hour
    step = 1 * u.min
    dt_s = (t1 - t0).to_value(u.s)
    n = int(np.floor(dt_s / step.to_value(u.s))) + 1
    times = t0 + np.arange(n) * step

    frame = AltAz(obstime=times, location=location)
    sun_alt = get_sun(times).transform_to(frame).alt.to_value(u.deg)

    dusk_times: List[Time] = []
    dawn_times: List[Time] = []
    for idx in _crossing_indices(sun_alt, sun_alt_dark_deg):
        rising = sun_alt[idx + 1] > sun_alt[idx]
        if rising:
            dawn_times.append(times[idx + 1])
        else:
            dusk_times.append(times[idx + 1])

    if not dusk_times or not dawn_times:
        raise ValueError(
            f"No astronomical night crossings found near {date!r} "
            f"(Sun alt threshold {sun_alt_dark_deg:.1f} deg)."
        )

    # Night of ``date``: dusk after noon UTC on that calendar day, then next dawn.
    noon = day + 12 * u.hour
    dusk = None
    for t in dusk_times:
        if t >= noon and t < day + 1.5 * u.day:
            dusk = t
            break
    if dusk is None:
        raise ValueError(f"No astronomical dusk found on {date!r}.")

    dawn = None
    for t in dawn_times:
        if t > dusk:
            dawn = t
            break
    if dawn is None:
        raise ValueError(f"No astronomical dawn found after dusk on {date!r}.")

    return dusk, dawn


def _mask_to_intervals(times: Time, mask: np.ndarray) -> List[Tuple[Time, Time]]:
    """Contiguous True runs in *mask* → list of (start, end) inclusive."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(times[i], times[j]) for i, j in zip(starts, ends)]


def _format_intervals(intervals: Sequence[Tuple[Time, Time]]) -> str:
    if not intervals:
        return "  (none)"
    lines = []
    for t0, t1 in intervals:
        lines.append(f"  {_format_time_print(t0)}  →  {_format_time_print(t1)}")
    return "\n".join(lines)


def _antipode(coord: SkyCoord) -> SkyCoord:
    return SkyCoord(
        ra=(coord.icrs.ra + 180 * u.deg) % (360 * u.deg),
        dec=-coord.icrs.dec,
        frame="icrs",
    )


def _moon_reflection_flags(
    sep_deg: np.ndarray,
    m1_radius_deg: float,
    m2_sep_min_deg: float,
    m2_sep_max_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (in_m1, in_m2) from Moon–pointing angular separation (deg)."""
    sep_arr = np.atleast_1d(np.asarray(sep_deg, dtype=float))
    in_m1 = sep_arr <= m1_radius_deg
    in_m2 = (sep_arr >= m2_sep_min_deg) & (sep_arr <= m2_sep_max_deg)
    return in_m1, in_m2


def _topocentric_sky_direction(altaz: SkyCoord) -> SkyCoord:
    """
    Map local AltAz directions (unit sphere) to ICRS for all-sky RA/Dec plots.

    Dropping distance is essential for the Moon: a GCRS/ICRS transform that
    keeps the lunar distance does not represent the topocentric pointing direction.
    """
    frame = altaz.frame
    direction = SkyCoord(alt=altaz.alt, az=altaz.az, frame=frame)
    return direction.transform_to("icrs")


@dataclass(frozen=True)
class ReflectionInputs:
    pointing: SkyCoord
    start: Time
    end: Time
    step: u.Quantity
    m1_radius_deg: float
    m2_sep_min_deg: float
    m2_sep_max_deg: float
    fov_deg: float
    sun_alt_dark_deg: float


def compute_reflection_veto(inp: ReflectionInputs) -> dict:
    if inp.end <= inp.start:
        raise ValueError("End must be after start.")

    dt_s = (inp.end - inp.start).to_value(u.s)
    n = int(np.floor(dt_s / inp.step.to_value(u.s))) + 1
    times = inp.start + np.arange(n) * inp.step

    frame = AltAz(obstime=times, location=OBSERVATORY_ASTRI)
    # Topocentric Moon (with ASTRI location) for correct local altitude / sky direction.
    moon_body = get_body("moon", times, location=OBSERVATORY_ASTRI)
    moon_altaz = moon_body.transform_to(frame)
    pointing_altaz = inp.pointing.transform_to(frame)
    sun_altaz = get_sun(times).transform_to(frame)

    # Separation on the local sky (what the telescope / mirrors actually see).
    sep = moon_altaz.separation(pointing_altaz).to_value(u.deg)
    in_m1, in_m2 = _moon_reflection_flags(
        sep, inp.m1_radius_deg, inp.m2_sep_min_deg, inp.m2_sep_max_deg
    )
    dark = sun_altaz.alt.to_value(u.deg) < inp.sun_alt_dark_deg
    moon_up = moon_altaz.alt.to_value(u.deg) > 0.0
    # No reflection risk if the Moon is below the local horizon.
    reflection_risk = (in_m1 | in_m2) & moon_up
    safe = ~reflection_risk
    safe_observable = safe & dark

    moon_sky = _topocentric_sky_direction(moon_altaz)

    return {
        "times": times,
        "moon": moon_sky,
        "moon_altaz": moon_altaz,
        "pointing_altaz": pointing_altaz,
        "sun_altaz": sun_altaz,
        "sep_deg": sep,
        "in_m1": in_m1,
        "in_m2": in_m2,
        "reflection_risk": reflection_risk,
        "safe": safe,
        "safe_observable": safe_observable,
        "dark": dark,
        "moon_up": moon_up,
        "moon_illum_pct": _moon_illumination_percent(times),
        "antipode": _antipode(inp.pointing),
    }


def _ra_deg_to_aitoff_rad(ra_deg: np.ndarray | float) -> np.ndarray | float:
    """Map ICRS RA (deg) to Aitoff longitude, matching matplotlib + astropy convention."""
    ra = np.asarray(ra_deg, dtype=float)
    wrapped = np.where(ra > 180.0, ra - 360.0, ra)
    if np.ndim(ra_deg) == 0:
        return np.deg2rad(float(wrapped))
    return np.deg2rad(wrapped)


def _coord_to_aitoff(coord: SkyCoord) -> Tuple[np.ndarray, np.ndarray]:
    ra_rad = np.atleast_1d(_ra_deg_to_aitoff_rad(coord.icrs.ra.deg))
    dec_rad = coord.icrs.dec.to_value(u.rad)
    return ra_rad, dec_rad


def _sky_zone_map(
    pointing: SkyCoord,
    m1_radius_deg: float,
    m2_sep_min_deg: float,
    m2_sep_max_deg: float,
    ra_step_deg: float = 1.0,
    dec_step_deg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-sky grid in RA/Dec → boolean masks for M1 and M2 (entire celestial sphere).

    RA is built in [-180, 180) so Aitoff contourf does not wrap discontinuously.
    """
    ra_vals = np.arange(-180.0, 180.0, ra_step_deg)
    dec_vals = np.arange(-90.0, 90.0 + dec_step_deg * 0.5, dec_step_deg)
    ra_grid, dec_grid = np.meshgrid(ra_vals, dec_vals)
    flat = SkyCoord(ra=ra_grid.ravel() * u.deg, dec=dec_grid.ravel() * u.deg, frame="icrs")
    sep = flat.separation(pointing.icrs).to_value(u.deg).reshape(ra_grid.shape)

    m1_mask = sep <= m1_radius_deg
    m2_mask = (sep >= m2_sep_min_deg) & (sep <= m2_sep_max_deg)
    return ra_grid, dec_grid, m1_mask, m2_mask


def _below_horizon_mask(
    ra_grid_deg: np.ndarray,
    dec_grid_deg: np.ndarray,
    t: Time,
    location: EarthLocation = OBSERVATORY_ASTRI,
) -> np.ndarray:
    """True where the sky direction is below the local horizon at *t*."""
    frame = AltAz(obstime=t, location=location)
    flat = SkyCoord(
        ra=ra_grid_deg.ravel() * u.deg,
        dec=dec_grid_deg.ravel() * u.deg,
        frame="icrs",
    )
    alt = flat.transform_to(frame).alt.to_value(u.deg).reshape(ra_grid_deg.shape)
    return alt < 0.0


def _sky_title(label: str, t: Time, status: str) -> str:
    return f"{label} — {_format_time_print(t)} UTC\nmoon reflection: {status}"


def _style_aitoff_axes(ax) -> None:
    """Black tick labels by default; recolor to white those drawn inside the map."""
    ax.tick_params(axis="both", colors="black", labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("black")


def _color_aitoff_tick_labels(ax) -> None:
    """Outside the Aitoff ellipse: black labels; inside: white labels."""
    from matplotlib.path import Path as MplPath

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boundary = MplPath(ax.patch.get_transform().transform(ax.patch.get_path().vertices))
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        bbox = lbl.get_window_extent(renderer=renderer)
        cx = 0.5 * (bbox.x0 + bbox.x1)
        cy = 0.5 * (bbox.y0 + bbox.y1)
        ax_x, ax_y = ax.transAxes.inverted().transform((cx, cy))
        # Labels in the margin (Dec ticks above/below the map) stay black.
        if ax_x < 0.0 or ax_x > 1.0 or ax_y < 0.0 or ax_y > 1.0:
            lbl.set_color("black")
        elif boundary.contains_point((cx, cy)):
            lbl.set_color("white")
        else:
            lbl.set_color("black")


def _small_circle_coords(center: SkyCoord, radius_deg: float, n: int = 180) -> SkyCoord:
    bearings = np.linspace(0, 360, n, endpoint=False) * u.deg
    pts = [center.directional_offset_by(b, radius_deg * u.deg) for b in bearings]
    return SkyCoord(ra=[p.ra for p in pts], dec=[p.dec for p in pts], frame=center.frame)


def _plot_time_series(
    path: str,
    res: dict,
    inp: ReflectionInputs,
    label: str,
) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times_dt = res["times"].datetime
    sep = np.asarray(res["sep_deg"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, height_ratios=[2, 1, 1])
    ax_alt, ax_sep, ax_illum = axes

    ax_alt.plot(times_dt, res["pointing_altaz"].alt, color=SOURCE_COLOR, lw=2, label=f"{label} (pointing)")
    ax_alt.plot(times_dt, res["moon_altaz"].alt, color=MOON_COLOR, ls="--", lw=1.5, label="Moon")
    ax_alt.fill_between(times_dt, 0, 90, where=res["dark"], color="dimgray", alpha=0.2, label="Astro. night")
    ax_alt.fill_between(times_dt, 0, 90, where=res["reflection_risk"], color="gray", alpha=0.35, label="Reflection risk")
    ax_alt.set_ylabel("Altitude (deg)")
    ax_alt.set_ylim(0, 90)
    ax_alt.legend(loc=0, fontsize=8)
    ax_alt.grid(True, alpha=0.3)
    ax_alt.set_title(f"Moon reflection veto — {label}")

    ax_sep.plot(times_dt, sep, color="tab:blue", lw=2, label="Moon–pointing sep.")
    ax_sep.axhspan(0, inp.m1_radius_deg, color=M1_ZONE_COLOR, alpha=0.2, label=f"M1 ≤{inp.m1_radius_deg:.0f}°")
    ax_sep.axhspan(inp.m2_sep_min_deg, inp.m2_sep_max_deg, color=M2_ZONE_COLOR, alpha=0.2,
                   label=f"M2 {inp.m2_sep_min_deg:.0f}–{inp.m2_sep_max_deg:.0f}°")
    ax_sep.set_ylabel("Sep (deg)")
    ax_sep.set_ylim(0, 180)
    ax_sep.legend(loc=0, fontsize=8)
    ax_sep.grid(True, alpha=0.3)

    ax_illum.plot(times_dt, res["moon_illum_pct"], color="orange", lw=2)
    ax_illum.fill_between(times_dt, 0, res["moon_illum_pct"], alpha=0.25, color="orange")
    ax_illum.set_ylabel("Illum (%)")
    ax_illum.set_ylim(0, 100)
    ax_illum.grid(True, alpha=0.3)
    ax_illum.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax_illum.set_xlabel("Time (UTC)")

    fig.autofmt_xdate(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _sky_legend_handles(inp: ReflectionInputs, *, include_horizon: bool = True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=M1_ZONE_COLOR, alpha=M1_ZONE_ALPHA, label="Moon in M1"),
        Patch(facecolor=M2_ZONE_COLOR, alpha=M2_ZONE_ALPHA, label="Moon in M2"),
    ]
    if include_horizon:
        handles.append(
            Patch(facecolor=BELOW_HORIZON_COLOR, alpha=BELOW_HORIZON_ALPHA, label="Below ASTRI horizon")
        )
    handles.extend(
        [
            Line2D([0], [0], marker="*", color="w", markerfacecolor=SOURCE_COLOR, ms=10, ls="None", label="Pointing"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=MOON_COLOR, ms=8, ls="None", label="Moon"),
            Line2D([0], [0], marker="x", color=M2_ZONE_COLOR, ms=8, mew=2, ls="None", label="Antipode (M2 centre)"),
            Line2D([0], [0], color=FOV_EDGE_COLOR, lw=1.5, label=f"FOV ~{inp.fov_deg:.0f}°"),
        ]
    )
    return handles


def _altaz_to_polar(altaz: SkyCoord) -> Tuple[np.ndarray, np.ndarray]:
    """Local sky → polar: r = zenith angle (deg), theta = azimuth (rad, N up, E right)."""
    zenith = (90.0 * u.deg - altaz.alt).to_value(u.deg)
    theta = np.deg2rad(altaz.az.to_value(u.deg))
    return zenith, theta


def _draw_sky_axes(
    ax,
    inp: ReflectionInputs,
    label: str,
    t: Time,
    status: str,
) -> None:
    """Aitoff RA/Dec sky with M1/M2 zones, horizon mask, pointing, antipode."""
    ax.set_facecolor(SAFE_SKY_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.35, lw=0.5)
    _style_aitoff_axes(ax)

    ra_grid, dec_grid, m1_mask, m2_mask = _sky_zone_map(
        inp.pointing, inp.m1_radius_deg, inp.m2_sep_min_deg, inp.m2_sep_max_deg,
    )
    ra_rad = np.deg2rad(ra_grid)
    dec_rad = np.deg2rad(dec_grid)
    below = _below_horizon_mask(ra_grid, dec_grid, t)

    ax.contourf(
        ra_rad, dec_rad, (m2_mask & ~below).astype(float),
        levels=[0.5, 1.5], colors=[M2_ZONE_COLOR], alpha=M2_ZONE_ALPHA, zorder=1,
    )
    ax.contourf(
        ra_rad, dec_rad, (m1_mask & ~below).astype(float),
        levels=[0.5, 1.5], colors=[M1_ZONE_COLOR], alpha=M1_ZONE_ALPHA, zorder=2,
    )
    ax.contourf(
        ra_rad, dec_rad, below.astype(float),
        levels=[0.5, 1.5], colors=[BELOW_HORIZON_COLOR], alpha=BELOW_HORIZON_ALPHA, zorder=3,
    )

    fov_pts = _small_circle_coords(inp.pointing, inp.fov_deg / 2.0)
    ra_f, dec_f = _coord_to_aitoff(fov_pts)
    ax.plot(ra_f, dec_f, color=FOV_EDGE_COLOR, lw=1.5, zorder=5)

    ra_s, dec_s = _coord_to_aitoff(inp.pointing)
    ax.plot(ra_s, dec_s, "*", color=SOURCE_COLOR, ms=14, zorder=6)

    anti = _antipode(inp.pointing)
    ra_a, dec_a = _coord_to_aitoff(anti)
    ax.plot(ra_a, dec_a, "x", color=M2_ZONE_COLOR, ms=10, mew=2, zorder=6)

    ax.set_title(_sky_title(label, t, status), fontsize=10, pad=12)
    ax.legend(
        handles=_sky_legend_handles(inp),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        framealpha=0.95,
    )


def _polar_zone_map(
    pointing: SkyCoord,
    t: Time,
    m1_radius_deg: float,
    m2_sep_min_deg: float,
    m2_sep_max_deg: float,
    alt_step_deg: float = 1.0,
    az_step_deg: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Local Alt/Az grid → (theta, zenith, m1_mask, m2_mask) for polar contourf.

    Only the above-horizon sky (alt >= 0) is returned; the disk rim is the horizon.
    """
    alt_vals = np.arange(0.0, 90.0, alt_step_deg)
    az_vals = np.arange(0.0, 360.0, az_step_deg)
    alt_grid, az_grid = np.meshgrid(alt_vals, az_vals, indexing="ij")
    frame = AltAz(obstime=t, location=OBSERVATORY_ASTRI)
    flat = SkyCoord(alt=alt_grid.ravel() * u.deg, az=az_grid.ravel() * u.deg, frame=frame)
    sep = flat.icrs.separation(pointing.icrs).to_value(u.deg).reshape(alt_grid.shape)

    m1_mask = sep <= m1_radius_deg
    m2_mask = (sep >= m2_sep_min_deg) & (sep <= m2_sep_max_deg)
    zenith = 90.0 - alt_grid
    theta = np.deg2rad(az_grid)
    return theta, zenith, m1_mask, m2_mask


def _draw_polar_sky_axes(
    ax,
    inp: ReflectionInputs,
    label: str,
    t: Time,
    status: str,
) -> None:
    """Polar Alt/Az sky (zenith centre, N up) with the same zones/markers as Aitoff."""
    ax.set_facecolor(SAFE_SKY_COLOR)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # r = zenith angle: 0 at centre (zenith), 90 at rim (horizon).
    ax.set_rlim(0, 90)
    # Radial labels: show altitude (90 - zenith angle); they sit inside → white.
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90", "75", "60", "45", "30", "15", "0"], fontsize=9, color="white")
    ax.set_rlabel_position(22.5)
    # Azimuth labels sit outside the disk → black (readable on white figure bg).
    az_ticks = np.arange(0, 360, 30)
    ax.set_thetagrids(az_ticks, labels=[f"{int(a)}" for a in az_ticks], fontsize=9, color="black")
    ax.tick_params(axis="x", colors="black", labelsize=9)
    ax.tick_params(axis="y", colors="white", labelsize=9)
    ax.grid(True, color=GRID_COLOR, alpha=0.45, lw=0.6)

    theta, zenith, m1_mask, m2_mask = _polar_zone_map(
        inp.pointing, t, inp.m1_radius_deg, inp.m2_sep_min_deg, inp.m2_sep_max_deg,
    )
    ax.contourf(
        theta, zenith, m2_mask.astype(float),
        levels=[0.5, 1.5], colors=[M2_ZONE_COLOR], alpha=M2_ZONE_ALPHA, zorder=1,
    )
    ax.contourf(
        theta, zenith, m1_mask.astype(float),
        levels=[0.5, 1.5], colors=[M1_ZONE_COLOR], alpha=M1_ZONE_ALPHA, zorder=2,
    )

    frame = AltAz(obstime=t, location=OBSERVATORY_ASTRI)
    # Only plot markers that are above the local horizon (no rim-clamping).
    fov_pts = _small_circle_coords(inp.pointing, inp.fov_deg / 2.0).transform_to(frame)
    fov_up = fov_pts.alt.to_value(u.deg) >= 0.0
    if np.any(fov_up):
        r_f, th_f = _altaz_to_polar(fov_pts[fov_up])
        ax.plot(th_f, r_f, color=FOV_EDGE_COLOR, lw=1.5, zorder=5)

    pnt = inp.pointing.transform_to(frame)
    if pnt.alt.to_value(u.deg) >= 0.0:
        r_p, th_p = _altaz_to_polar(pnt)
        ax.plot(th_p, r_p, "*", color=SOURCE_COLOR, ms=14, zorder=6)

    anti = _antipode(inp.pointing).transform_to(frame)
    if anti.alt.to_value(u.deg) >= 0.0:
        r_a, th_a = _altaz_to_polar(anti)
        ax.plot(th_a, r_a, "x", color=M2_ZONE_COLOR, ms=10, mew=2, zorder=6)

    ax.set_title(_sky_title(label, t, status), fontsize=10, pad=14)
    ax.legend(
        handles=_sky_legend_handles(inp, include_horizon=False),
        loc="upper left",
        bbox_to_anchor=(1.12, 1.0),
        fontsize=8,
        framealpha=0.95,
    )


def _plot_sky_snapshot(
    path: str,
    t: Time,
    inp: ReflectionInputs,
    label: str,
    res: dict,
) -> None:
    import matplotlib.pyplot as plt

    idx = int(np.argmin(np.abs((res["times"] - t).jd)))
    t_plot = res["times"][idx]
    moon = res["moon"][idx]
    status = "at risk" if bool(np.asarray(res["reflection_risk"])[idx]) else "safe"

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111, projection="aitoff")
    _draw_sky_axes(ax, inp, label, t_plot, status)

    ra_m, dec_m = _coord_to_aitoff(moon)
    ax.plot(ra_m, dec_m, "o", color=MOON_COLOR, ms=9, zorder=8)

    plt.tight_layout()
    _color_aitoff_tick_labels(ax)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sky_polar(
    path: str,
    t: Time,
    inp: ReflectionInputs,
    label: str,
    res: dict,
) -> None:
    import matplotlib.pyplot as plt

    idx = int(np.argmin(np.abs((res["times"] - t).jd)))
    t_plot = res["times"][idx]
    status = "at risk" if bool(np.asarray(res["reflection_risk"])[idx]) else "safe"

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    _draw_polar_sky_axes(ax, inp, label, t_plot, status)

    moon_altaz = res["moon_altaz"][idx]
    if moon_altaz.alt.to_value(u.deg) >= 0.0:
        r_m, th_m = _altaz_to_polar(moon_altaz)
        ax.plot(th_m, r_m, "o", color=MOON_COLOR, ms=9, zorder=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "ASTRI Moon reflection veto on the celestial sphere. "
            "Reports safe time windows for a source during an astronomical night, "
            "with optional Aitoff (RA/Dec) and polar (Alt/Az) sky snapshots."
        )
    )

    tgt = ap.add_argument_group("Source / pointing")
    tgt.add_argument("--source", "--source-name", dest="source_name", type=str,
                       help="Source name (Sesame) or label with --ra/--dec.")
    tgt.add_argument("--ra", type=str, help="RA (deg or hh:mm:ss).")
    tgt.add_argument("--dec", type=str, help="Dec (deg or dd:mm:ss).")
    tgt.add_argument("--label", type=str, default=None, help="Label for plots and printouts.")

    win = ap.add_argument_group("Observing night (UTC)")
    win.add_argument(
        "--date",
        type=str,
        required=True,
        help="Night date YYYY-MM-DD (astronomical dusk on this evening → dawn next morning).",
    )
    win.add_argument("--step", type=str, default="5m", help="Time step. Default: 5m")
    win.add_argument(
        "--sun-alt-dark-deg",
        type=float,
        default=-18.0,
        help="Astronomical night: Sun altitude threshold (deg). Default: -18",
    )

    lim = ap.add_argument_group("Reflection zones (separation from pointing, deg)")
    lim.add_argument(
        "--m1-radius-deg",
        type=float,
        default=30.0,
        help="M1 cap radius: Moon within this angle of pointing → reflection. Default: 30",
    )
    lim.add_argument(
        "--m2-sep-min-deg",
        type=float,
        default=125.0,
        help="M2 annulus inner edge (from pointing). Default: 125",
    )
    lim.add_argument(
        "--m2-sep-max-deg",
        type=float,
        default=170.0,
        help="M2 annulus outer edge (from pointing). Default: 170",
    )
    lim.add_argument(
        "--fov-deg",
        type=float,
        default=10.0,
        help="Source field-of-view diameter on sky plot (deg). Default: 10",
    )

    out = ap.add_argument_group("Outputs")
    out.add_argument("--plot", type=str, default=None, help="Save time-series plot (png/pdf).")
    out.add_argument(
        "--plot-sky",
        type=str,
        default=None,
        help="Save Aitoff RA/Dec sky snapshot at mid-night (or --sky-time).",
    )
    out.add_argument(
        "--plot-sky-polar",
        type=str,
        default=None,
        help="Save polar Alt/Az sky snapshot at mid-night (or --sky-time).",
    )
    out.add_argument(
        "--sky-time",
        type=str,
        default=None,
        help="UTC epoch for sky plots (YYYY-MM-DDTHH:MM:SS). Default: mid-night.",
    )

    args = ap.parse_args()

    if args.m2_sep_min_deg >= args.m2_sep_max_deg:
        ap.error("--m2-sep-min-deg must be less than --m2-sep-max-deg")

    try:
        pointing, label = _resolve_coord(args.source_name, args.ra, args.dec, args.label or "Target")
        if args.label:
            label = args.label
    except ValueError as exc:
        ap.error(str(exc))

    try:
        start, end = astronomical_night_window(args.date, args.sun_alt_dark_deg)
    except ValueError as exc:
        ap.error(str(exc))

    step = _parse_step(args.step)
    inp = ReflectionInputs(
        pointing=pointing,
        start=start,
        end=end,
        step=step,
        m1_radius_deg=float(args.m1_radius_deg),
        m2_sep_min_deg=float(args.m2_sep_min_deg),
        m2_sep_max_deg=float(args.m2_sep_max_deg),
        fov_deg=float(args.fov_deg),
        sun_alt_dark_deg=float(args.sun_alt_dark_deg),
    )
    res = compute_reflection_veto(inp)

    anti = res["antipode"]
    print("Observatory: ASTRI (Teide)")
    print(f"Source / pointing: {label}  ({pointing.to_string('hmsdms')})")
    print(f"Antipode (M2 centre): ({anti.to_string('hmsdms')})")
    print(f"Astronomical night (UTC): {_format_time_print(start)}  →  {_format_time_print(end)}   step={step}")
    print(
        f"Reflection zones: M1 ≤{inp.m1_radius_deg:.0f}°  |  "
        f"M2 {inp.m2_sep_min_deg:.0f}–{inp.m2_sep_max_deg:.0f}° from pointing"
    )

    safe_all = res["safe"]
    safe_obs = res["safe_observable"]
    n = len(safe_all)
    print(f"\nSafe samples (no Moon reflection risk*): {np.sum(safe_all)}/{n} ({100.0 * np.mean(safe_all):.1f}%)")
    print(f"Safe samples (during astronomical night): {np.sum(safe_obs)}/{n} ({100.0 * np.mean(safe_obs):.1f}%)")
    print("  * Moon below horizon counts as safe (no illumination of mirrors).")

    print("\nSafe windows (no Moon reflection risk):")
    print(_format_intervals(_mask_to_intervals(res["times"], safe_all)))

    print("\nSafe windows (astronomical night):")
    print(_format_intervals(_mask_to_intervals(res["times"], safe_obs)))

    risk_intervals = _mask_to_intervals(res["times"], res["reflection_risk"])
    if risk_intervals:
        print("\nReflection-risk windows:")
        print(_format_intervals(risk_intervals))

    sep = np.asarray(res["sep_deg"])
    illum = np.asarray(res["moon_illum_pct"])
    print("\nMoon summary:")
    print(f"  Illumination (%): min={np.min(illum):.1f}  med={np.median(illum):.1f}  max={np.max(illum):.1f}")
    print(f"  Sep to pointing (deg): min={np.min(sep):.2f}  med={np.median(sep):.2f}  max={np.max(sep):.2f}")
    print(f"  In M1 zone: {np.sum(res['in_m1'])} samples")
    print(f"  In M2 zone: {np.sum(res['in_m2'])} samples")

    if args.plot:
        _plot_time_series(args.plot, res, inp, label)
        print(f"\nSaved time-series plot: {args.plot}")

    t_mid = start + (end - start) / 2
    t_sky = _parse_time(args.sky_time) if args.sky_time else t_mid
    if args.plot_sky:
        _plot_sky_snapshot(args.plot_sky, t_sky, inp, label, res)
        print(f"Saved sky snapshot (Aitoff): {args.plot_sky}")

    if args.plot_sky_polar:
        _plot_sky_polar(args.plot_sky_polar, t_sky, inp, label, res)
        print(f"Saved sky snapshot (polar): {args.plot_sky_polar}")


if __name__ == "__main__":
    main()
