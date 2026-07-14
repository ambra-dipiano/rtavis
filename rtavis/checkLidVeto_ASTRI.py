import argparse
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_moon, get_sun
from astropy.coordinates.name_resolve import NameResolveError
from astropy.time import Time
from typing import Optional, Tuple


OBSERVATORY_ASTRI = EarthLocation(lat=28.2983 * u.deg, lon=-16.5097 * u.deg, height=2390 * u.m)
MOON_COLOR = "navy"
MOON_LINESTYLE = "--"
POINTING_COLOR = "red"
POINTING_LINESTYLE = "-"
SOURCE_COLOR = "darkorange"
VETO_PRIMARY_COLOR = "crimson"
VETO_SECONDARY_COLOR = "darkviolet"
LIDS_OPEN_COLOR = "#0d2d4d"  # dark blue — lids open (pointing patch on sky view)
LIDS_CLOSED_COLOR = "#44a6fc"  # light blue — lids closed (slightly darker for white grid labels)
SKY_GRID_COLOR = "#ffffff"


def _parse_time(s: str) -> Time:
    return Time(s.replace(" ", "T"), scale="utc")


def _is_date_only(s: str) -> bool:
    normalized = s.strip().replace(" ", "T")
    return "T" not in normalized and ":" not in normalized


def _resolve_window_and_sky_time(
    date: Optional[str],
    start_str: Optional[str],
    end_str: Optional[str],
) -> Tuple[Time, Time, Time]:
    """
    Resolve time-series window and sky-view epoch (UTC).

    --date YYYY-MM-DD           -> sky at midnight; night window [date-1h, date+1d+1h]
    --date YYYY-MM-DDTHH:MM:SS  -> sky at that instant; same night window for calendar date
    --start / --end             -> sky at mid-window
    (none)                      -> today UTC at midnight
    """
    if date:
        normalized = date.strip().replace(" ", "T")
        day = Time(normalized.split("T")[0], scale="utc")
        t_sky = day if _is_date_only(date) else _parse_time(date)
        start = day - 1 * u.hour
        end = day + 1 * u.day + 1 * u.hour
        return start, end, t_sky

    if start_str and end_str:
        start = _parse_time(start_str)
        end = _parse_time(end_str)
        t_sky = start + (end - start) / 2
        return start, end, t_sky

    day = Time(Time.now().strftime("%Y-%m-%d"), scale="utc")
    start = day - 1 * u.hour
    end = day + 1 * u.day + 1 * u.hour
    return start, end, day


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


def _resolve_coord(name: Optional[str], ra: Optional[str], dec: Optional[str], label_default: str) -> Tuple[SkyCoord, str]:
    """Resolve pointing/source from --ra/--dec or from astropy Sesame (--source name)."""
    if ra and dec:
        try:
            coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="fk5", equinox="J2000.0")
        except ValueError:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="fk5", equinox="J2000.0")
        label = name or label_default
        return coord, label
    if name:
        try:
            coord = SkyCoord.from_name(name).transform_to("fk5")
        except NameResolveError as exc:
            raise ValueError(
                f"Could not resolve {name!r} with astropy SkyCoord.from_name (Sesame). "
                "Pass --ra and --dec instead, or use a name Sesame knows (e.g. 'Cygnus X-3')."
            ) from exc
        return coord, name
    raise ValueError("Provide --source (Sesame name) OR both --ra and --dec")


def _moon_illumination_percent(times: Time) -> np.ndarray:
    sun = get_sun(times)
    moon = get_moon(times)
    elong = sun.separation(moon).to_value(u.deg)
    return (1 + np.cos(np.radians(180.0 - elong))) / 2.0 * 100.0


def _sep_deg(a: SkyCoord, b: SkyCoord) -> u.Quantity:
    return a.icrs.separation(b.icrs)


def _azimuth_offset_deg(moon_az_deg, pointing_az_deg):
    """Moon azimuth minus pointing azimuth, wrapped to [-180, 180]."""
    return (np.asarray(moon_az_deg) - np.asarray(pointing_az_deg) + 180.0) % 360.0 - 180.0


def _in_symmetric_az_band(offset_deg, band_az_min: float, band_az_max: float) -> np.ndarray:
    """True where |azimuth offset| from pointing is within [band_az_min, band_az_max]."""
    abs_off = np.abs(np.asarray(offset_deg))
    return (abs_off >= band_az_min) & (abs_off <= band_az_max)


@dataclass(frozen=True)
class LidInputs:
    pointing: SkyCoord
    source: Optional[SkyCoord]
    start: Time
    end: Time
    step: u.Quantity
    open_pointing_deg: float
    m2_corona_az_min: float
    m2_corona_az_max: float


def compute_lids(inp: LidInputs):
    if inp.end <= inp.start:
        raise ValueError("--end must be after --start")

    dt_s = (inp.end - inp.start).to_value(u.s)
    n = int(np.floor(dt_s / inp.step.to_value(u.s))) + 1
    times = inp.start + np.arange(n) * inp.step

    frame = AltAz(obstime=times, location=OBSERVATORY_ASTRI)
    moon = get_moon(times)

    moon_altaz = moon.transform_to(frame)
    pointing_altaz = inp.pointing.transform_to(frame)

    sep_primary = _sep_deg(moon, inp.pointing).to(u.deg)
    az_offset = _azimuth_offset_deg(
        moon_altaz.az.to_value(u.deg),
        pointing_altaz.az.to_value(u.deg),
    )
    in_m1_band = _in_symmetric_az_band(az_offset, 0.0, inp.open_pointing_deg)
    in_m2_corona = _in_symmetric_az_band(az_offset, inp.m2_corona_az_min, inp.m2_corona_az_max)
    in_closed_az = in_m1_band | in_m2_corona

    open_by_m1 = ~in_m1_band
    open_by_m2 = ~in_m2_corona
    lids_open = ~in_closed_az
    lids_closed = in_closed_az

    source_altaz = None
    sep_source = None
    if inp.source is not None:
        source_altaz = inp.source.transform_to(frame)
        sep_source = _sep_deg(moon, inp.source.transform_to(frame)).to(u.deg)

    illum = _moon_illumination_percent(times)

    return {
        "times": times,
        "moon_altaz": moon_altaz,
        "pointing_altaz": pointing_altaz,
        "source_altaz": source_altaz,
        "moon_illum_pct": illum,
        "sep_primary_deg": sep_primary,
        "az_offset_deg": az_offset,
        "in_m1_band": in_m1_band,
        "in_m2_corona": in_m2_corona,
        "sep_source_deg": sep_source,
        "open_by_m1": open_by_m1,
        "open_by_m2": open_by_m2,
        "lids_open": lids_open,
        "lids_closed": lids_closed,
    }


def _sky_disk_mask(
    alt_grid_deg: np.ndarray,
    az_grid_deg: np.ndarray,
    center: SkyCoord,
    radius_deg: float,
    frame: AltAz,
) -> np.ndarray:
    """True where sky direction is within radius_deg of center."""
    alts, azs = np.meshgrid(alt_grid_deg, az_grid_deg, indexing="ij")
    flat = SkyCoord(alt=alts.ravel() * u.deg, az=azs.ravel() * u.deg, frame=frame)
    dist = flat.icrs.separation(center.icrs).to_value(u.deg).reshape(alts.shape)
    return dist < radius_deg


def _plot_time_series(
    path: str,
    res: dict,
    inp: LidInputs,
    pointing_label: str,
    source_label: Optional[str],
    has_distinct_source: bool,
):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times_dt = res["times"].datetime
    az_off = np.asarray(res["az_offset_deg"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, height_ratios=[2, 1, 1, 1])
    ax_alt, ax_illum, ax_p, ax_s = axes

    ax_alt.plot(
        times_dt,
        res["pointing_altaz"].alt,
        label=f"Pointing ({pointing_label})",
        color=POINTING_COLOR,
        linestyle=POINTING_LINESTYLE,
        linewidth=2,
    )
    if has_distinct_source:
        ax_alt.plot(
            times_dt,
            res["source_altaz"].alt,
            label=f"Source ({source_label})",
            color=SOURCE_COLOR,
            linestyle="-.",
            linewidth=1.8,
        )
    ax_alt.plot(times_dt, res["moon_altaz"].alt, color=MOON_COLOR, linestyle=MOON_LINESTYLE, label="Moon", linewidth=1.5)
    ax_alt.set_ylabel("Altitude (deg)")
    ax_alt.set_ylim(0, 90)
    ax_alt.grid(True, alpha=0.3)
    ax_alt.legend(loc=0, fontsize=8)
    ax_alt.set_title(f"ASTRI lids: {pointing_label}")

    ax_illum.plot(times_dt, res["moon_illum_pct"], color="orange", linewidth=2, label="Moon illumination")
    ax_illum.fill_between(times_dt, 0, res["moon_illum_pct"], alpha=0.25, color="orange")
    ax_illum.set_ylabel("Illum (%)")
    ax_illum.set_ylim(0, 100)
    ax_illum.grid(True, alpha=0.3)
    ax_illum.legend(loc="lower center", fontsize=8)

    pad_p = max(1.0, (az_off.max() - az_off.min()) * 0.08)
    ax_p.plot(times_dt, az_off, color=VETO_PRIMARY_COLOR, linewidth=2, label="Moon AZ offset from pointing")
    ax_p.axhspan(-inp.open_pointing_deg, inp.open_pointing_deg, color=VETO_PRIMARY_COLOR, alpha=0.15, label=f"M1 band ±{inp.open_pointing_deg:.0f} deg")
    ax_p.fill_between(times_dt, az_off.min() - pad_p, az_off.max() + pad_p, where=res["lids_closed"], color="gray", alpha=0.15)
    ax_p.set_ylabel("AZ offset (deg)")
    ax_p.set_ylim(az_off.min() - pad_p, az_off.max() + pad_p)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc=0, fontsize=8)
    ax_p.set_title(f"M1 band: ±0–{inp.open_pointing_deg:.0f} deg from pointing AZ", fontsize=9)

    pad_s = max(1.0, (az_off.max() - az_off.min()) * 0.08)
    ax_s.plot(times_dt, az_off, color=VETO_SECONDARY_COLOR, linewidth=2, label="Moon AZ offset from pointing")
    ax_s.axhspan(inp.m2_corona_az_min, inp.m2_corona_az_max, color=VETO_SECONDARY_COLOR, alpha=0.15, label="M2 corona (+)")
    ax_s.axhspan(-inp.m2_corona_az_max, -inp.m2_corona_az_min, color=VETO_SECONDARY_COLOR, alpha=0.15, label="M2 corona (-)")
    ax_s.fill_between(times_dt, az_off.min() - pad_s, az_off.max() + pad_s, where=res["lids_closed"], color="gray", alpha=0.15, label="Lids closed")
    ax_s.set_ylabel("AZ offset (deg)")
    ax_s.set_ylim(az_off.min() - pad_s, az_off.max() + pad_s)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc=0, fontsize=8)
    ax_s.set_title(
        f"M2 reflection corona: ±{inp.m2_corona_az_min:.0f}–{inp.m2_corona_az_max:.0f} deg from pointing AZ",
        fontsize=9,
    )
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax_s.set_xlabel("Time (UTC)")

    fig.autofmt_xdate(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _altaz_to_polar(altaz: SkyCoord) -> Tuple[np.ndarray, np.ndarray]:
    """Map AltAz to polar sky-view coords: r = zenith angle, theta = azimuth (rad, N up)."""
    zenith = (90.0 * u.deg - altaz.alt).to_value(u.deg)
    theta = np.deg2rad(altaz.az.to_value(u.deg))
    return zenith, theta


def _lids_open_now(
    moon_altaz: SkyCoord,
    pointing_altaz: SkyCoord,
    open_pointing_deg: float,
    m2_corona_az_min: float,
    m2_corona_az_max: float,
) -> bool:
    offset = _azimuth_offset_deg(
        moon_altaz.az.to_value(u.deg),
        pointing_altaz.az.to_value(u.deg),
    )
    in_m1 = _in_symmetric_az_band(offset, 0.0, open_pointing_deg)
    in_m2 = _in_symmetric_az_band(offset, m2_corona_az_min, m2_corona_az_max)
    return not (bool(np.asarray(in_m1)) or bool(np.asarray(in_m2)))


def _sky_az_closed_mask(
    alt_grid_deg: np.ndarray,
    az_grid_deg: np.ndarray,
    pointing_az_deg: float,
    m1_az_max: float,
    m2_corona_az_min: float,
    m2_corona_az_max: float,
) -> np.ndarray:
    """Sky grid points where Moon azimuth offset falls in M1 or M2 closed bands."""
    _, az_mesh = np.meshgrid(alt_grid_deg, az_grid_deg, indexing="ij")
    offset = _azimuth_offset_deg(az_mesh, pointing_az_deg)
    in_m1 = _in_symmetric_az_band(offset, 0.0, m1_az_max)
    in_m2 = _in_symmetric_az_band(offset, m2_corona_az_min, m2_corona_az_max)
    return in_m1 | in_m2


def _plot_sky_view(
    path: str,
    t: Time,
    pointing: SkyCoord,
    source: Optional[SkyCoord],
    inp: LidInputs,
    pointing_label: str,
    source_label: Optional[str],
    has_distinct_source: bool,
) -> str:
    """Save sky-view snapshot (NOW). Returns one-line lid status."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    frame = AltAz(obstime=t, location=OBSERVATORY_ASTRI)
    moon = get_moon(t)
    moon_az = moon.transform_to(frame)
    pnt = pointing.transform_to(frame)

    alt_grid = np.linspace(0, 89.5, 180)
    az_grid = np.linspace(0, 359.5, 360)
    alt_mesh, az_mesh = np.meshgrid(alt_grid, az_grid, indexing="ij")
    r_mesh = 90.0 - alt_mesh
    theta_mesh = np.deg2rad(az_mesh)

    closed_az = _sky_az_closed_mask(
        alt_grid, az_grid, pnt.az.to_value(u.deg),
        inp.open_pointing_deg, inp.m2_corona_az_min, inp.m2_corona_az_max,
    )
    zone = np.where(closed_az, 1.0, 0.0)

    lids_open_now = _lids_open_now(
        moon_az, pnt,
        inp.open_pointing_deg, inp.m2_corona_az_min, inp.m2_corona_az_max,
    )
    status = "LIDS OPEN" if lids_open_now else "LIDS CLOSED"

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)
    ax.set_facecolor(LIDS_CLOSED_COLOR)

    # Altitude rings: label zenith angle from horizon (0) to zenith (90).
    alt_ticks = np.arange(0, 91, 15)
    ax.set_yticks(alt_ticks)
    ax.set_yticklabels([f"{90 - int(t)}" for t in alt_ticks], fontsize=10, color=SKY_GRID_COLOR)
    ax.set_rlabel_position(22.5)

    # Azimuth spokes every 30 deg (degrees, not cardinal names).
    az_step = 30
    az_ticks = np.arange(0, 360, az_step)
    ax.set_thetagrids(az_ticks, labels=[f"{int(a)}" for a in az_ticks], fontsize=10, color=SKY_GRID_COLOR)
    ax.tick_params(axis="both", colors=SKY_GRID_COLOR, labelsize=10)

    ax.pcolormesh(
        theta_mesh, r_mesh, zone,
        cmap=ListedColormap([LIDS_OPEN_COLOR, LIDS_CLOSED_COLOR]),
        alpha=1.0, shading="auto", vmin=0, vmax=1, zorder=1,
    )

    # Grid drawn on top of shading so it stays visible.
    ax.set_axisbelow(False)
    ax.grid(True, linestyle="-", color=SKY_GRID_COLOR, alpha=0.85, linewidth=0.8, zorder=5)
    for az_minor in np.arange(0, 360, 15):
        if az_minor % az_step == 0:
            continue
        th = np.deg2rad(az_minor)
        ax.plot([th, th], [0, 90], color=SKY_GRID_COLOR, alpha=0.45, linewidth=0.4, zorder=5)
    for alt_minor in np.arange(0, 90, 15):
        if alt_minor in alt_ticks:
            continue
        ax.plot(np.linspace(0, 2 * np.pi, 360), np.full(360, alt_minor), color=SKY_GRID_COLOR, alpha=0.45, linewidth=0.4, zorder=5)

    r_p, th_p = _altaz_to_polar(pnt)
    r_m, th_m = _altaz_to_polar(moon_az)
    ax.plot(th_p, r_p, "*", color=POINTING_COLOR, markersize=16, zorder=10)
    ax.plot(th_m, r_m, "o", color=MOON_COLOR, markersize=10, zorder=11)

    ax.set_title(f"{pointing_label}  |  {_format_time_print(t)} UTC", fontsize=11, pad=14)

    ax.legend(
        handles=[
            Patch(facecolor=LIDS_OPEN_COLOR, edgecolor=LIDS_OPEN_COLOR, label="Lids open"),
            Patch(
                facecolor=LIDS_CLOSED_COLOR,
                edgecolor="#8eb5d6",
                label=f"Lids closed (±0–{inp.open_pointing_deg:.0f}°, ±{inp.m2_corona_az_min:.0f}–{inp.m2_corona_az_max:.0f}° AZ)",
            ),
            Line2D([0], [0], marker="*", color="w", markerfacecolor=POINTING_COLOR, markersize=12, linestyle="None", label="Pointing"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=MOON_COLOR, markersize=8, linestyle="None", label="Moon"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        framealpha=0.95,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return status


def main():
    ap = argparse.ArgumentParser(
        description="ASTRI lid open/close vs Moon: symmetric azimuth bands for M1 (±0–30°) and M2 (±135–170°)."
    )

    tgt = ap.add_argument_group("Pointing / source")
    tgt.add_argument("--source", "--source-name", dest="source_name", type=str, help="Source label or Sesame name (SkyCoord.from_name).")
    tgt.add_argument("--ra", type=str, help="RA (degrees or hh:mm:ss). Use with --dec instead of --source name.")
    tgt.add_argument("--dec", type=str, help="Dec (degrees or dd:mm:ss). Use with --ra instead of --source name.")
    tgt.add_argument("--pointing-name", type=str, help="Pointing name if different from source.")
    tgt.add_argument("--pointing-ra", type=str, help="Pointing RA if different from source.")
    tgt.add_argument("--pointing-dec", type=str, help="Pointing Dec if different from source.")
    tgt.add_argument("--label", type=str, default=None, help="Label for pointing in plots/printouts.")

    win = ap.add_argument_group("Time window (UTC)")
    win.add_argument("--start", type=str, help="Start datetime UTC.")
    win.add_argument("--end", type=str, help="End datetime UTC.")
    win.add_argument(
        "--date",
        type=str,
        default=None,
        help="UTC date/time. Date only (YYYY-MM-DD) -> sky view at midnight; "
        "with time (YYYY-MM-DDTHH:MM:SS) -> sky view at that instant. "
        "Night window is [date-1h, date+1d+1h]. Default: today at midnight.",
    )
    win.add_argument("--step", type=str, default="10m", help="Time step. Default: 10m")

    lim = ap.add_argument_group("Lid open limits")
    lim.add_argument(
        "--open-pointing-deg",
        "--closed-pointing-deg",
        "--primary-veto-deg",
        dest="open_pointing_deg",
        type=float,
        default=30.0,
        help="M1 closed band: max |azimuth offset| from pointing (deg). Default: 30",
    )
    lim.add_argument(
        "--m2-corona-az-min",
        "--secondary-corona-az-min",
        dest="m2_corona_az_min",
        type=float,
        default=135.0,
        help="M2 reflection corona: minimum azimuth offset from pointing (deg). Default: 135",
    )
    lim.add_argument(
        "--m2-corona-az-max",
        "--secondary-corona-az-max",
        dest="m2_corona_az_max",
        type=float,
        default=170.0,
        help="M2 reflection corona: maximum azimuth offset from pointing (deg). Default: 170",
    )

    out = ap.add_argument_group("Outputs")
    out.add_argument("--plot", type=str, default=None, help="Save time-series plot.")
    out.add_argument("--plot-sky-view", "--plot-sky", dest="plot_sky_view", type=str, default=None, help="Save sky-view snapshot at --date epoch.")

    args = ap.parse_args()

    start, end, t_sky = _resolve_window_and_sky_time(args.date, args.start, args.end)
    if args.start or args.end:
        if not (args.start and args.end):
            ap.error("Provide both --start and --end, or use --date instead")

    # Pointing
    try:
        if args.pointing_name or (args.pointing_ra and args.pointing_dec):
            pointing, pointing_label = _resolve_coord(
                args.pointing_name,
                args.pointing_ra,
                args.pointing_dec,
                args.label or "Pointing",
            )
            if args.label:
                pointing_label = args.label
        elif args.source_name or (args.ra and args.dec):
            pointing, pointing_label = _resolve_coord(args.source_name, args.ra, args.dec, args.label or "Target")
            if args.label:
                pointing_label = args.label
        else:
            ap.error("Provide --source or --ra/--dec (or separate --pointing-ra/--pointing-dec)")
    except ValueError as exc:
        ap.error(str(exc))

    # Optional distinct source
    source = None
    source_label = None
    has_distinct_source = False
    if args.source_name or (args.ra and args.dec):
        try:
            source, source_label = _resolve_coord(args.source_name, args.ra, args.dec, "Source")
            if source.separation(pointing) > 0.01 * u.arcsec:
                has_distinct_source = True
            else:
                source = None
        except ValueError:
            pass

    if args.m2_corona_az_min >= args.m2_corona_az_max:
        ap.error("--m2-corona-az-min must be less than --m2-corona-az-max")

    step = _parse_step(args.step)
    inp = LidInputs(
        pointing=pointing,
        source=source,
        start=start,
        end=end,
        step=step,
        open_pointing_deg=float(args.open_pointing_deg),
        m2_corona_az_min=float(args.m2_corona_az_min),
        m2_corona_az_max=float(args.m2_corona_az_max),
    )
    res = compute_lids(inp)

    print("Observatory: ASTRI (Teide)")
    print(f"Pointing: {pointing_label}  ({pointing.to_string('hmsdms')})")
    if has_distinct_source:
        print(f"Source: {source_label}  ({source.to_string('hmsdms')})")
    print(f"Window (UTC): {_format_time_print(start)}  ->  {_format_time_print(end)}   step={step}")
    print(
        f"Lids open when Moon azimuth offset NOT in ±0–{inp.open_pointing_deg:.0f} deg (M1) "
        f"and NOT in ±{inp.m2_corona_az_min:.0f}–{inp.m2_corona_az_max:.0f} deg (M2)"
    )

    open_mask = res["lids_open"]
    n = len(open_mask)
    print(f"\nLids open samples: {np.sum(open_mask)}/{n} ({100.0 * np.mean(open_mask):.1f}%)")
    if not np.all(open_mask):
        closed = res["lids_closed"]
        i0 = np.where(closed)[0][0]
        i1 = np.where(closed)[0][-1]
        print(f"First closed: {_format_time_print(res['times'][i0])}")
        print(f"Last closed:  {_format_time_print(res['times'][i1])}")
    else:
        print("Lids open for the full window.")

    sep_p = res["sep_primary_deg"].to_value(u.deg)
    az_off = np.asarray(res["az_offset_deg"])
    illum = np.asarray(res["moon_illum_pct"])
    print("\nMoon summary:")
    print(f"  Illumination (%): min={np.min(illum):.1f}  med={np.median(illum):.1f}  max={np.max(illum):.1f}")
    print(f"  Sep to pointing (deg): min={np.min(sep_p):.2f}  med={np.median(sep_p):.2f}  max={np.max(sep_p):.2f}")
    print(f"  AZ offset from pointing (deg): min={np.min(az_off):.2f}  med={np.median(az_off):.2f}  max={np.max(az_off):.2f}")
    print(f"  Open by M1 band:      {np.sum(res['open_by_m1'])} samples")
    print(f"  Open by M2 corona:    {np.sum(res['open_by_m2'])} samples")
    print(f"  In M1 band:           {np.sum(res['in_m1_band'])} samples")
    print(f"  In M2 corona:         {np.sum(res['in_m2_corona'])} samples")

    if args.plot:
        _plot_time_series(args.plot, res, inp, pointing_label, source_label, has_distinct_source)
        print(f"\nSaved plot: {args.plot}")

    if args.plot_sky_view:
        status = _plot_sky_view(
            args.plot_sky_view,
            t_sky,
            pointing,
            source if has_distinct_source else None,
            inp,
            pointing_label,
            source_label,
            has_distinct_source,
        )
        print(f"Saved sky view: {args.plot_sky_view}")
        print(f"{_format_time_print(t_sky)} UTC — {status}")


if __name__ == "__main__":
    main()
