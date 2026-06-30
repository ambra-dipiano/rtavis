import argparse
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_moon, get_sun
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
LIDS_CLOSED_COLOR = "#d6e8f7"  # light blue — lids closed (moon / transit patch on sky view)
SKY_GRID_COLOR = "#ffffff"


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


def _resolve_coord(name: Optional[str], ra: Optional[str], dec: Optional[str], label_default: str) -> Tuple[SkyCoord, str]:
    if name:
        coord = SkyCoord.from_name(name).transform_to("fk5")
        return coord, name
    if ra and dec:
        try:
            coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="fk5", equinox="J2000.0")
        except ValueError:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="fk5", equinox="J2000.0")
        return coord, label_default
    raise ValueError("Provide --source/--pointing-name OR both RA and Dec")


def _moon_illumination_percent(times: Time) -> np.ndarray:
    sun = get_sun(times)
    moon = get_moon(times)
    elong = sun.separation(moon).to_value(u.deg)
    return (1 + np.cos(np.radians(180.0 - elong))) / 2.0 * 100.0


def _opposite_pointing(pointing: SkyCoord) -> SkyCoord:
    """Antipode of pointing on the celestial sphere (180 deg), fixed while tracking."""
    return pointing.icrs.directional_offset_by(0.0 * u.deg, 180.0 * u.deg)


def _sep_deg(a: SkyCoord, b: SkyCoord) -> u.Quantity:
    return a.icrs.separation(b.icrs)


@dataclass(frozen=True)
class LidInputs:
    pointing: SkyCoord
    source: Optional[SkyCoord]
    start: Time
    end: Time
    step: u.Quantity
    open_pointing_deg: float
    open_opposite_deg: float


def compute_lids(inp: LidInputs):
    if inp.end <= inp.start:
        raise ValueError("--end must be after --start")

    dt_s = (inp.end - inp.start).to_value(u.s)
    n = int(np.floor(dt_s / inp.step.to_value(u.s))) + 1
    times = inp.start + np.arange(n) * inp.step

    frame = AltAz(obstime=times, location=OBSERVATORY_ASTRI)
    moon = get_moon(times)
    opposite_sky = _opposite_pointing(inp.pointing)

    moon_altaz = moon.transform_to(frame)
    pointing_altaz = inp.pointing.transform_to(frame)
    opposite_altaz = opposite_sky.transform_to(frame)

    sep_primary = _sep_deg(moon, inp.pointing).to(u.deg)
    sep_secondary = _sep_deg(moon, opposite_sky).to(u.deg)

    open_by_pointing = sep_primary >= inp.open_pointing_deg * u.deg
    open_by_opposite = sep_secondary >= inp.open_opposite_deg * u.deg
    lids_open = open_by_pointing & open_by_opposite
    lids_closed = ~lids_open

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
        "opposite_altaz": opposite_altaz,
        "opposite_sky": opposite_sky,
        "source_altaz": source_altaz,
        "moon_illum_pct": illum,
        "sep_primary_deg": sep_primary,
        "sep_secondary_deg": sep_secondary,
        "sep_source_deg": sep_source,
        "open_by_pointing": open_by_pointing,
        "open_by_opposite": open_by_opposite,
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
    sep_p = res["sep_primary_deg"].to_value(u.deg)
    sep_s = res["sep_secondary_deg"].to_value(u.deg)

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

    pad_p = max(0.5, (sep_p.max() - sep_p.min()) * 0.15)
    ax_p.plot(times_dt, sep_p, color=VETO_PRIMARY_COLOR, linewidth=2, label="Moon–pointing")
    ax_p.axhline(inp.open_pointing_deg, color=VETO_PRIMARY_COLOR, linestyle="--", linewidth=1, alpha=0.7, label=f"Open limit {inp.open_pointing_deg:.0f} deg")
    ax_p.fill_between(times_dt, sep_p.min() - pad_p, sep_p.max() + pad_p, where=res["lids_closed"], color="gray", alpha=0.15)
    ax_p.set_ylabel("Moon Separation (deg)")
    ax_p.set_ylim(sep_p.min() - pad_p, sep_p.max() + pad_p)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc=0, fontsize=8)
    ax_p.set_title("Moon–pointing separation (Moon proper motion vs fixed target)", fontsize=9)

    pad_s = max(1.0, (sep_s.max() - sep_s.min()) * 0.08)
    ax_s.plot(times_dt, sep_s, color=VETO_SECONDARY_COLOR, linewidth=2, label="Moon–opposite")
    ax_s.axhline(inp.open_opposite_deg, color=VETO_SECONDARY_COLOR, linestyle="--", linewidth=1, alpha=0.7, label=f"Open limit {inp.open_opposite_deg:.0f} deg")
    ax_s.fill_between(times_dt, 0, sep_s.max() + pad_s, where=res["lids_closed"], color="gray", alpha=0.15, label="Lids closed")
    ax_s.set_ylabel("Moon Separation (deg)")
    ax_s.set_ylim(max(0, sep_s.min() - pad_s), sep_s.max() + pad_s)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc=0, fontsize=8)
    ax_s.set_title("Moon–opposite separation (Moon proper motion vs fixed antipode)", fontsize=9)
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
    moon: SkyCoord,
    pointing: SkyCoord,
    opposite_sky: SkyCoord,
    open_pointing_deg: float,
    open_opposite_deg: float,
) -> bool:
    sep_p = _sep_deg(moon, pointing).to_value(u.deg)
    sep_s = _sep_deg(moon, opposite_sky).to_value(u.deg)
    return (sep_p >= open_pointing_deg) and (sep_s >= open_opposite_deg)


def _sky_lids_open_zones(
    alt_grid_deg: np.ndarray,
    az_grid_deg: np.ndarray,
    pointing: SkyCoord,
    opposite_sky: SkyCoord,
    open_pointing_deg: float,
    open_opposite_deg: float,
    frame: AltAz,
) -> np.ndarray:
    """Sky patch around pointing (+ antipode when above horizon): lids open on sky view."""
    near_pointing = _sky_disk_mask(alt_grid_deg, az_grid_deg, pointing, open_pointing_deg, frame)
    near_opposite = _sky_disk_mask(alt_grid_deg, az_grid_deg, opposite_sky, open_opposite_deg, frame)
    return near_pointing | near_opposite


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
    opposite_sky = _opposite_pointing(pointing)
    moon_az = moon.transform_to(frame)
    pnt = pointing.transform_to(frame)

    alt_grid = np.linspace(0, 89.5, 180)
    az_grid = np.linspace(0, 359.5, 360)
    alt_mesh, az_mesh = np.meshgrid(alt_grid, az_grid, indexing="ij")
    r_mesh = 90.0 - alt_mesh
    theta_mesh = np.deg2rad(az_mesh)

    lids_open_zone = _sky_lids_open_zones(
        alt_grid, az_grid, pointing, opposite_sky,
        inp.open_pointing_deg, inp.open_opposite_deg, frame,
    )
    zone = np.where(lids_open_zone, 0.0, 1.0)

    lids_open_now = _lids_open_now(moon, pointing, opposite_sky, inp.open_pointing_deg, inp.open_opposite_deg)
    status = "LIDS OPEN" if lids_open_now else "LIDS CLOSED"

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)
    ax.set_facecolor(LIDS_CLOSED_COLOR)

    # Altitude rings: label zenith angle from horizon (0) to zenith (90).
    alt_ticks = np.arange(0, 91, 15)
    ax.set_yticks(alt_ticks)
    ax.set_yticklabels([f"{90 - int(t)}" for t in alt_ticks], fontsize=10)
    ax.set_rlabel_position(22.5)

    # Azimuth spokes every 30 deg (degrees, not cardinal names).
    az_step = 30
    az_ticks = np.arange(0, 360, az_step)
    ax.set_thetagrids(az_ticks, labels=[f"{int(a)}" for a in az_ticks], fontsize=10)

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
    ax.plot(th_p, r_p, "o", color=POINTING_COLOR, markersize=10, zorder=10)
    ax.plot(th_m, r_m, "*", color=MOON_COLOR, markersize=16, zorder=11)

    ax.set_title(f"{pointing_label}  |  {_format_time_print(t)} UTC", fontsize=11, pad=14)

    ax.legend(
        handles=[
            Patch(facecolor=LIDS_OPEN_COLOR, edgecolor=LIDS_OPEN_COLOR, label="Lids open"),
            Patch(facecolor=LIDS_CLOSED_COLOR, edgecolor="#9bb8d3", label="Lids closed"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=POINTING_COLOR, markersize=8, linestyle="None", label="Pointing"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor=MOON_COLOR, markersize=12, linestyle="None", label="Moon"),
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
        description="ASTRI lid open/close vs Moon during transits (primary and secondary mirror reflection)."
    )

    tgt = ap.add_argument_group("Pointing / source")
    tgt.add_argument("--source", "--source-name", dest="source_name", type=str, help="Source name (Sesame). Used as pointing if pointing not set.")
    tgt.add_argument("--ra", type=str, help="RA (pointing or source).")
    tgt.add_argument("--dec", type=str, help="Dec (pointing or source).")
    tgt.add_argument("--pointing-name", type=str, help="Pointing name if different from source.")
    tgt.add_argument("--pointing-ra", type=str, help="Pointing RA if different from source.")
    tgt.add_argument("--pointing-dec", type=str, help="Pointing Dec if different from source.")
    tgt.add_argument("--label", type=str, default=None, help="Label for pointing in plots/printouts.")

    win = ap.add_argument_group("Time window (UTC)")
    win.add_argument("--start", type=str, help="Start datetime UTC.")
    win.add_argument("--end", type=str, help="End datetime UTC.")
    win.add_argument("--date", type=str, help="Single-night mode: YYYY-MM-DD (uses [date-1h, date+1d+1h]).")
    win.add_argument("--step", type=str, default="10m", help="Time step. Default: 10m")

    lim = ap.add_argument_group("Lid open limits (angular distance on sky)")
    lim.add_argument(
        "--open-pointing-deg",
        "--closed-pointing-deg",
        "--primary-veto-deg",
        dest="open_pointing_deg",
        type=float,
        default=30.0,
        help="Lids open if Moon at least this angle from pointing. Default: 30",
    )
    lim.add_argument(
        "--open-opposite-deg",
        "--closed-opposite-deg",
        "--secondary-veto-deg",
        dest="open_opposite_deg",
        type=float,
        default=50.0,
        help="Lids open if Moon at least this angle from pointing antipode (180 deg on sky). Default: 50",
    )

    out = ap.add_argument_group("Outputs")
    out.add_argument("--plot", type=str, default=None, help="Save time-series plot.")
    out.add_argument("--plot-sky-view", "--plot-sky", dest="plot_sky_view", type=str, default=None, help="Save sky-view snapshot (NOW).")
    out.add_argument("--sky-time", type=str, default=None, help="UTC time for sky view. Default: mid-window.")

    args = ap.parse_args()

    if args.date:
        start = Time(args.date, scale="utc") - 1 * u.hour
        end = Time(args.date, scale="utc") + 1 * u.day + 1 * u.hour
    else:
        if not args.start or not args.end:
            ap.error("Provide either --date OR both --start and --end")
        start = _parse_time(args.start)
        end = _parse_time(args.end)

    # Pointing
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
        ap.error("Provide pointing coordinates (--ra/--dec or --source) or separate --pointing-ra/--pointing-dec")

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

    step = _parse_step(args.step)
    inp = LidInputs(
        pointing=pointing,
        source=source,
        start=start,
        end=end,
        step=step,
        open_pointing_deg=float(args.open_pointing_deg),
        open_opposite_deg=float(args.open_opposite_deg),
    )
    res = compute_lids(inp)

    print("Observatory: ASTRI (Teide)")
    print(f"Pointing: {pointing_label}  ({pointing.to_string('hmsdms')})")
    if has_distinct_source:
        print(f"Source: {source_label}  ({source.to_string('hmsdms')})")
    print(f"Window (UTC): {_format_time_print(start)}  ->  {_format_time_print(end)}   step={step}")
    print(
        f"Lids open when: Moon >= {inp.open_pointing_deg:.1f} deg from pointing "
        f"AND Moon >= {inp.open_opposite_deg:.1f} deg from pointing antipode"
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
    sep_s = res["sep_secondary_deg"].to_value(u.deg)
    illum = np.asarray(res["moon_illum_pct"])
    print("\nMoon summary:")
    print(f"  Illumination (%): min={np.min(illum):.1f}  med={np.median(illum):.1f}  max={np.max(illum):.1f}")
    print(f"  Sep to pointing (deg): min={np.min(sep_p):.2f}  med={np.median(sep_p):.2f}  max={np.max(sep_p):.2f}")
    print(f"  Sep to opposite (deg):   min={np.min(sep_s):.2f}  med={np.median(sep_s):.2f}  max={np.max(sep_s):.2f}")
    print(f"  Open by pointing:  {np.sum(res['open_by_pointing'])} samples")
    print(f"  Open by opposite:  {np.sum(res['open_by_opposite'])} samples")

    if args.plot:
        _plot_time_series(args.plot, res, inp, pointing_label, source_label, has_distinct_source)
        print(f"\nSaved plot: {args.plot}")

    if args.plot_sky_view:
        if args.sky_time:
            t_sky = _parse_time(args.sky_time)
        else:
            t_sky = start + (end - start) / 2
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
