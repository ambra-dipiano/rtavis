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
SOURCE_COLOR = "red"
SOURCE_LINESTYLE = "-"


def _parse_time(s: str) -> Time:
    # Accept "YYYY-MM-DDTHH:MM:SS" (recommended) and "YYYY-MM-DD HH:MM:SS".
    return Time(s.replace(" ", "T"), scale="utc")


def _format_time_print(t: Time) -> str:
    return t.iso.split(".")[0]


def _parse_step(step: str) -> u.Quantity:
    """
    Parse a human step string into a time quantity.

    Accepts examples: "30s", "10m" (minutes), "10min", "2h", "1d".
    """
    s = step.strip().lower().replace(" ", "")
    if not s:
        raise ValueError("Empty --step")

    # In astropy units "m" is meters so here it is mapped to minutes.
    if s.endswith("m") and not s.endswith("min"):
        s = s[:-1] + "min"

    q = u.Quantity(s)
    if not q.unit.is_equivalent(u.s):
        raise ValueError(f"--step must be a time quantity (e.g. 10m, 30s, 2h). Got: {step!r}")
    return q


def _moon_illumination_percent(times: Time) -> np.ndarray:
    sun = get_sun(times)
    moon = get_moon(times)
    elong = sun.separation(moon).to_value(u.deg)
    return (1 + np.cos(np.radians(180.0 - elong))) / 2.0 * 100.0


def _moon_phase(times: Time) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (illum_pct, elongation_deg_0_180, phase_deg_0_360, phase_name[]) for each time.

    - elongation_deg_0_180: Sun–Moon separation, 0..180 deg.
    - phase_deg_0_360: a continuous phase angle, where:
        New=0/360, First Quarter=90, Full=180, Last Quarter=270.
    - phase_name: one of New / First Quarter / Full / Last Quarter (nearest by phase_deg_0_360).
    """
    sun = get_sun(times)
    moon = get_moon(times)
    elong = sun.separation(moon).to(u.deg).value  # 0..180
    illum = (1 + np.cos(np.radians(180.0 - elong))) / 2.0 * 100.0

    # Waxing if elongation increasing. Use a longer baseline to avoid numerical sign flips.
    # Evaluate elongation at +/- 6 hours; clamp at window edges.
    baseline = 6 * u.hour
    t_prev = times - baseline
    t_next = times + baseline
    # Clamp to [times[0], times[-1]]
    t_prev = Time(np.maximum(t_prev.jd, times[0].jd), format="jd", scale="utc")
    t_next = Time(np.minimum(t_next.jd, times[-1].jd), format="jd", scale="utc")
    elong_prev = get_sun(t_prev).separation(get_moon(t_prev)).to(u.deg).value
    elong_next = get_sun(t_next).separation(get_moon(t_next)).to(u.deg).value
    waxing = (elong_next - elong_prev) >= 0

    phase_0_360 = np.where(waxing, elong, 360.0 - elong)

    # Nearest phase name on 8 bins (every 45 deg).
    phase_deg = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    phase_names = np.array(
        [
            "New",
            "Waxing first quarter",
            "Waxing half",
            "Waxing last quarter",
            "Full",
            "Waning first quarter",
            "Waning half",
            "Waning last quarter",
        ],
        dtype=object,
    )
    dist = np.abs(((phase_0_360[:, None] - phase_deg[None, :] + 180.0) % 360.0) - 180.0)
    names = phase_names[np.argmin(dist, axis=1)]

    return illum, elong, phase_0_360, names


def _crossings(altitudes: u.Quantity, threshold: u.Quantity) -> np.ndarray:
    above = altitudes > threshold
    return np.where(np.diff(above.astype(int)) != 0)[0]

def _circular_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Circular distance on a 0..360 circle, returned in [0, 180]."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _overlay_phase_markers(ax, times_dt, phase_deg_0_360: np.ndarray, *, label_step_deg: float):
    """
    Overlay phase labels on an existing axis.

    - Always labels the first and last time in the window (at least 2 labels).
    - Adds additional labels when the (unwrapped) phase angle changes by >= label_step_deg.
    """
    if label_step_deg <= 0:
        label_step_deg = 45.0

    # Always show start/end phase for reference.
    start_deg = float(phase_deg_0_360[0])
    end_deg = float(phase_deg_0_360[-1])

    def _major_name(d: float) -> str:
        phase_deg = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
        phase_names = np.array(
            [
                "New",
                "Waxing first quarter",
                "Waxing half",
                "Waxing last quarter",
                "Full",
                "Waning first quarter",
                "Waning half",
                "Waning last quarter",
            ],
            dtype=object,
        )
        dist = np.abs(((d - phase_deg + 180.0) % 360.0) - 180.0)
        return str(phase_names[int(np.argmin(dist))])

    # Degree shown is the continuous phase angle in [0, 360):
    # New=0/360, Waxing half=90, Full=180, Waning half=270.
    start_txt = f"{_major_name(start_deg)} ({start_deg:.0f}°)"
    end_txt = f"{_major_name(end_deg)} ({end_deg:.0f}°)"

    ax.text(
        0.01,
        0.98,
        start_txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        color="purple",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.6, edgecolor="none"),
    )
    ax.text(
        0.99,
        0.98,
        end_txt,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=12,
        color="purple",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.6, edgecolor="none"),
    )

    # Add additional labels every ~label_step_deg change in phase.
    # Unwrap to avoid a big jump at 360->0 boundary.
    phase_unwrapped = np.degrees(np.unwrap(np.radians(phase_deg_0_360)))
    total_span = float(phase_unwrapped[-1] - phase_unwrapped[0])
    if total_span <= 0:
        return

    step = max(label_step_deg, total_span / 2.0) if total_span < 2 * label_step_deg else label_step_deg

    last_idx = 0
    for i in range(1, len(times_dt) - 1):
        if (phase_unwrapped[i] - phase_unwrapped[last_idx]) >= step:
            d = float(phase_deg_0_360[i] % 360.0)
            txt = f"{_major_name(d)} ({d:.0f}°)"
            ax.annotate(
                txt,
                xy=(times_dt[i], 0.90),
                xycoords=("data", "axes fraction"),
                xytext=(0, 6),
                textcoords="offset points",
                rotation=0,
                va="bottom",
                ha="center",
                fontsize=12,
                color="purple",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
            )
            last_idx = i


@dataclass(frozen=True)
class VisibilityInputs:
    source: Optional[SkyCoord]
    start: Time
    end: Time
    step: u.Quantity
    min_alt: u.Quantity
    sun_alt_dark: u.Quantity


def compute_window(inp: VisibilityInputs):
    if inp.end <= inp.start:
        raise ValueError("--end must be after --start")

    dt_s = (inp.end - inp.start).to_value(u.s)
    n = int(np.floor(dt_s / inp.step.to_value(u.s))) + 1
    times = inp.start + np.arange(n) * inp.step

    frame = AltAz(obstime=times, location=OBSERVATORY_ASTRI)
    sun_altaz = get_sun(times).transform_to(frame)
    moon_altaz = get_moon(times).transform_to(frame)

    illum_pct, phase_angle_deg, phase_deg_0_360, phase_name = _moon_phase(times)

    dark = sun_altaz.alt < inp.sun_alt_dark
    if inp.source is not None:
        src_altaz = inp.source.transform_to(frame)
        moon_sep = get_moon(times).icrs.separation(inp.source.icrs).to(u.deg)
        visible = (src_altaz.alt >= inp.min_alt) & dark
    else:
        src_altaz = None
        moon_sep = None
        visible = None

    return {
        "times": times,
        "sun_altaz": sun_altaz,
        "moon_altaz": moon_altaz,
        "src_altaz": src_altaz,
        "moon_sep": moon_sep,
        "moon_illum_pct": illum_pct,
        "moon_phase_angle_deg": phase_angle_deg,
        "moon_phase_deg_0_360": phase_deg_0_360,
        "moon_phase_name": phase_name,
        "dark": dark,
        "visible": visible,
    }


def main():
    ap = argparse.ArgumentParser(description="ASTRI source visibility over a time window, including Moon separation, illumination and phase.")

    src = ap.add_argument_group("Source")
    src.add_argument("--source-name", "--source", dest="source_name", type=str, help="Resolve a target name (via Sesame). Example: 'Mrk 501'")
    src.add_argument("--ra", type=str, help="Right ascension, e.g. '16:56:28' or degrees.")
    src.add_argument("--dec", type=str, help="Declination, e.g. '+39:45:36' or degrees.")
    src.add_argument("--label", type=str, default=None, help="Label used in outputs and plots.")
    src.add_argument("--moon-only", action="store_true", help="Compute Moon presence, illumination and phase only (no source required).")

    win = ap.add_argument_group("Time window (UTC)")
    win.add_argument("--start", type=str, help="Start datetime UTC. Example: 2026-05-01T00:00:00")
    win.add_argument("--end", type=str, help="End datetime UTC. Example: 2026-05-15T00:00:00")
    win.add_argument("--date", type=str, help="Convenience single-night mode: YYYY-MM-DD; uses [date-1h, date+1d+1h].")
    win.add_argument("--step", type=str, default="20m", help="Time step, e.g. 1m, 10m, 30s, 2h. Default: 20m")

    cfg = ap.add_argument_group("Visibility criteria")
    cfg.add_argument("--min-alt-deg", type=float, default=0.0, help="Minimum source altitude (deg). Default: 0")
    cfg.add_argument("--sun-alt-dark-deg", type=float, default=-18.0, help="Consider 'dark' when Sun alt < this (deg). Default: -18 (astronomical)")

    out = ap.add_argument_group("Outputs")
    out.add_argument("--plot", type=str, default=None, help="Save a plot to this path (png/pdf).")
    out.add_argument("--phase-label-step-deg", type=float, default=45.0, help="On the illumination panel, add phase labels roughly every N degrees (always includes start/end). Default: 45")

    args = ap.parse_args()

    if args.date:
        start = Time(args.date, scale="utc") - 1 * u.hour
        end = Time(args.date, scale="utc") + 1 * u.day + 1 * u.hour
    else:
        if not args.start or not args.end:
            ap.error("Provide either --date OR both --start and --end")
        start = _parse_time(args.start)
        end = _parse_time(args.end)

    source: Optional[SkyCoord]
    label: str

    if args.moon_only:
        source = None
        label = args.label or "Moon"
    elif args.source_name:
        source = SkyCoord.from_name(args.source_name).transform_to("fk5")
        label = args.label or args.source_name
    elif args.ra and args.dec:
        # If user passes numeric strings, assume degrees; otherwise HMS/DMS.
        try:
            ra_deg = float(args.ra)
            dec_deg = float(args.dec)
            source = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5", equinox="J2000.0")
        except ValueError:
            source = SkyCoord(ra=args.ra, dec=args.dec, unit=(u.hourangle, u.deg), frame="fk5", equinox="J2000.0")
        label = args.label or "Target"
    else:
        # No source specified: default to moon-only behavior.
        source = None
        label = args.label or "Moon"

    step = _parse_step(args.step)
    inp = VisibilityInputs(source=source, start=start, end=end, step=step, min_alt=args.min_alt_deg * u.deg, sun_alt_dark=args.sun_alt_dark_deg * u.deg)

    res = compute_window(inp)

    print("Observatory: ASTRI (Teide)")
    if source is not None:
        print(f"Source: {label}  ({source.to_string('hmsdms')})")
    else:
        print("Mode: Moon-only (no source)")
    print(f"Window (UTC): {_format_time_print(start)}  ->  {_format_time_print(end)}   step={step}")
    if source is not None:
        print(f"Criteria: src_alt>={args.min_alt_deg:.1f} deg AND sun_alt<{args.sun_alt_dark_deg:.1f} deg")

        visible = res["visible"]
        if np.any(visible):
            t0 = res["times"][np.where(visible)[0][0]]
            t1 = res["times"][np.where(visible)[0][-1]]
            frac = 100.0 * np.mean(visible)
            print(f"\nVisible samples: {np.sum(visible)}/{len(visible)} ({frac:.1f}%)")
            print(f"First visible: {_format_time_print(t0)}")
            print(f"Last visible:  {_format_time_print(t1)}")
        else:
            print("\nVisible samples: 0")

    illum_all = np.asarray(res["moon_illum_pct"])
    print("\nMoon summary (%):")
    print(f"  Illumination: min={np.min(illum_all):.1f}%  med={np.median(illum_all):.1f}%  max={np.max(illum_all):.1f}%")
    phase_deg = np.asarray(res["moon_phase_deg_0_360"])
    phase_name = np.asarray(res["moon_phase_name"], dtype=object)
    unique_names, counts = np.unique(phase_name, return_counts=True)
    dominant_phase = str(unique_names[np.argmax(counts)])
    print(
        "  Phase: "
        f"start={phase_name[0]} ({phase_deg[0]:.1f} deg), "
        f"end={phase_name[-1]} ({phase_deg[-1]:.1f} deg), "
        f"dominant={dominant_phase}"
    )

    if source is not None:
        visible = res["visible"]
        mask = visible if np.any(visible) else np.ones(len(visible), dtype=bool)
        moon_sep = res["moon_sep"][mask].to_value(u.deg)
        print("\nMoon–source separation (deg):")
        print(f"  min={np.min(moon_sep):.2f}  med={np.median(moon_sep):.2f}  max={np.max(moon_sep):.2f}")

    # Plot (optional; imported lazily)
    if args.plot:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        times_dt = res["times"].datetime
        if source is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 1])
            ax_sep = None
        else:
            fig, (ax1, ax2, ax_sep) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, height_ratios=[3, 1, 1])

        if source is not None:
            ax1.plot(
                times_dt,
                res["src_altaz"].alt,
                label=label,
                linewidth=2,
                color=SOURCE_COLOR,
                linestyle=SOURCE_LINESTYLE,
            )
        ax1.plot(
            times_dt,
            res["moon_altaz"].alt,
            linestyle=MOON_LINESTYLE,
            color=MOON_COLOR,
            label="Moon",
            linewidth=1.5,
        )
        if source is not None:
            ax1.axhline(inp.min_alt.to_value(u.deg), color="k", linewidth=1, alpha=0.6)

        # Darkness shading (Sun alt below threshold)
        ax1.fill_between(times_dt, -90, 90, where=res["dark"], color="dimgray", alpha=0.25, label="Dark")

        ax1.set_ylabel("Altitude (deg)")
        ax1.set_ylim(0, 90)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc=0)
        ax1.set_title(f"ASTRI window: {label}")

        ax2.plot(times_dt, res["moon_illum_pct"], color="orange", linewidth=2, label="Moon illumination")
        ax2.fill_between(times_dt, 0, res["moon_illum_pct"], alpha=0.25, color="orange")
        ax2.set_ylabel("Illum (%)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower center")

        # Overlay phase markers on the illumination panel
        _overlay_phase_markers(
            ax2,
            times_dt,
            np.asarray(res["moon_phase_deg_0_360"]),
            label_step_deg=float(args.phase_label_step_deg),
        )

        if source is not None:
            ax_sep.plot(
                times_dt,
                res["moon_sep"].to_value(u.deg),
                color="tab:blue",
                linewidth=2,
                label="Moon–source sep",
            )
            ax_sep.set_ylabel("Sep (deg)")
            ax_sep.grid(True, alpha=0.3)
            ax_sep.legend(loc=0)

        (ax_sep if source is not None else ax2).xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        (ax_sep if source is not None else ax2).set_xlabel("Time (UTC)")

        fig.autofmt_xdate(rotation=0, ha="center")

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig(args.plot, dpi=150)
        print(f"\nSaved plot: {args.plot}")


if __name__ == "__main__":
    main()
