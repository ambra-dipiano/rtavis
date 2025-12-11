import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy.time import Time
import astropy.units as u

# INPUTS

observatory = EarthLocation(lat=28.2983 * u.deg, lon= -16.5097 * u.deg, height=2390 * u.m)

# MARKARIAN 501
plot_label = 'Markarian 501'
#source = "Mrk 501"
#source = SkyCoord(ra='16:53:52.2', dec='+39:45:37', unit=(u.hourangle, u.deg)) #TevCAT
source = SkyCoord(ra='16:56:28', dec='+39:45:36', unit=(u.hourangle, u.deg), frame='fk5', equinox='J2000.0') #ASTRI

# CRAB
#plot_label = 'Crab Nebula'
#source = "Crab"
#source = SkyCoord(ra='05:34:30.9', dec='+22:00:44.5', unit=(u.hourangle, u.deg)) #TevCAT
#source = SkyCoord(ra='05:34:31', dec='+22:00:52', unit=(u.hourangle, u.deg)) #ASTRI


date = "2025-12-11"   # YYYY-MM-DD, start of the observing night
query_time = "11:15:00"   # Time in hh:mm:ss format (e.g., "12:30:45" for 12:30:45 UTC), or None to skip

# RESOLVE SOURCE

if type(source) == str:
    source = SkyCoord.from_name(source, unit=(u.hourangle, u.deg), frame='fk5', equinox='J2000.0')

print(f'Source coordinates:', source.to_string("hmsdms"), '\n')

# TIME GRID FOR WHOLE NIGHT

start = Time(date) - 1*u.hour # 1 hour before start of UTC day
end = Time(date) + 1*u.day + 1*u.hour
step = 1*u.min # OR: step = 10*u.s

# Create time array
delta_t = (end - start).to(u.s).value  # Total duration in seconds
num_steps = int(delta_t / step.to(u.s).value)
time_values = start.jd + np.arange(num_steps) * step.to(u.day).value
times = Time(time_values, format='jd') 
altaz_frame = AltAz(obstime=times, location=observatory)

sun_altaz = get_sun(times).transform_to(altaz_frame)
moon_altaz = get_moon(times).transform_to(altaz_frame)
source_altaz = source.transform_to(altaz_frame)

# TWILIGHT TIMES

def find_crossings(altitudes, threshold):
    """Find times when altitude crosses given threshold."""
    above = altitudes > threshold
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]
    return crossings

def get_crossing_direction(altitudes, idx):
    """Determine if crossing is upward (True) or downward (False)."""
    if idx < len(altitudes) - 1:
        return altitudes[idx + 1] > altitudes[idx]
    return altitudes[idx] > altitudes[idx - 1] if idx > 0 else True

def format_time_print(t):
    """Format Time object to hh:mm:ss string without fractional seconds."""
    return t.iso.split('.')[0]  # Remove fractional seconds

twilight_limits = {
    "civil": -6*u.deg,
    "nautical": -12*u.deg,
    "astronomical": -18*u.deg}

twilight_times = {}

for name, limit in twilight_limits.items():
    idx = find_crossings(sun_altaz.alt, limit)
    tlist = [times[i] for i in idx]
    twilight_times[name] = tlist
    
    # Determine start (sunset) and end (sunrise) of twilight
    starts = []
    ends = []
    for i in idx:
        is_rising = get_crossing_direction(sun_altaz.alt, i)
        if is_rising:
            ends.append(times[i])
        else:
            starts.append(times[i])
    
    print(f"{name.capitalize()} twilight (UTC):")
    if starts:
        for t in starts:
            print(f"  Start: {format_time_print(t)}")
    if ends:
        for t in ends:
            print(f"  End:   {format_time_print(t)}")
    if not starts and not ends:
        print("  No crossings found")

# MOON PHASE
# Calculate moon phase at midnight (representative of the night)
t_midnight = Time(date + "T00:00:00")
sun_midnight = get_sun(t_midnight)
moon_midnight = get_moon(t_midnight)

# Calculate elongation angle (angular separation between sun and moon)
elongation = sun_midnight.separation(moon_midnight)

# Determine moon phase
phase_angle = elongation.deg
illumination = (1 + np.cos(np.radians(180 - phase_angle))) / 2 * 100  # Illumination percentage

if phase_angle < 45:
    phase_name = "New Moon"
elif phase_angle < 135:
    phase_name = "First Quarter"
elif phase_angle < 225:
    phase_name = "Full Moon"
else:
    phase_name = "Last Quarter"

print(f"\nMoon phase at midnight ({format_time_print(t_midnight)}):")
print(f"  Phase: {phase_name}")
#print(f"  Elongation: {elongation:.1f}")
print(f"  Illumination: {illumination:.1f}%")
if phase_angle < 30 or phase_angle > 330:
    print("NEW MOON")
elif phase_angle > 150 and phase_angle < 210:
    print("FULL MOON")

# RISE / SET TIMES OF SOURCE WITH Alt/Az

idx_source = find_crossings(source_altaz.alt, 0*u.deg)
source_rise_set = [times[i] for i in idx_source]

print("\nSource rise/set (UTC):")
rises = []
sets = []
for i in idx_source:
    is_rising = get_crossing_direction(source_altaz.alt, i)
    if is_rising:
        rises.append(times[i])
    else:
        sets.append(times[i])

if rises:
    for t in rises:
        altaz_frame_single = AltAz(obstime=t, location=observatory)
        source_altaz_single = source.transform_to(altaz_frame_single)
        print(f"  Rise: {format_time_print(t)} - Alt={source_altaz_single.alt:.2f}, Az={source_altaz_single.az:.2f}")
if sets:
    for t in sets:
        altaz_frame_single = AltAz(obstime=t, location=observatory)
        source_altaz_single = source.transform_to(altaz_frame_single)
        print(f"  Set:  {format_time_print(t)} - Alt={source_altaz_single.alt:.2f}, Az={source_altaz_single.az:.2f}")
if not rises and not sets:
    print("  No rise/set found")

# Alt/Az at specific times
print("\nSource Alt/Az at specific times:")

# Alt/Az at maximum altitude
max_alt_idx = np.argmax(source_altaz.alt)
t_max_alt = times[max_alt_idx]
altaz_frame_max = AltAz(obstime=t_max_alt, location=observatory)
source_altaz_max = source.transform_to(altaz_frame_max)
print(f"  Maximum altitude at {format_time_print(t_max_alt)}: Alt={source_altaz_max.alt:.2f}, Az={source_altaz_max.az:.2f}")

# Alt/Az at user-specified time
if query_time is not None:
    time_str = f"{date}T{query_time}"
    t_query = Time(time_str)
    altaz_frame_query = AltAz(obstime=t_query, location=observatory)
    source_altaz_query = source.transform_to(altaz_frame_query)
    print(f"  Custom time {query_time} ({format_time_print(t_query)}): Alt={source_altaz_query.alt:.2f}, Az={source_altaz_query.az:.2f}")

# PLOT VISIBILITY

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 1])

# Calculate moon illumination over time
sun_coords = get_sun(times)
moon_coords = get_moon(times)
moon_illumination = []
for i in range(len(times)):
    elongation = sun_coords[i].separation(moon_coords[i])
    phase_angle = elongation.deg
    illum = (1 + np.cos(np.radians(180 - phase_angle))) / 2 * 100
    moon_illumination.append(illum)
moon_illumination = np.array(moon_illumination)

# Top plot: Altitude
# Source altitude
ax1.plot(times.datetime, source_altaz.alt, label=plot_label, linewidth=2)

# Moon altitude
ax1.plot(times.datetime, moon_altaz.alt, linestyle="--", label="Moon", linewidth=1.5)

# Twilight shading
def fill_twilight(limit, color, label):
    mask = sun_altaz.alt < limit
    ax1.fill_between(times.datetime, 0, 90, where=mask,
                     color=color, alpha=0.5, label=label)

fill_twilight(-6*u.deg, "lightgray", "Civil Twilight")
fill_twilight(-12*u.deg, "gray", "Nautical Twilight")
fill_twilight(-18*u.deg, "dimgray", "Astronomical Twilight")

# Labels for top plot
ax1.axhline(0, color="black", linewidth=1)
ax1.set_ylabel("Altitude (deg)", fontsize=12)
ax1.set_title(f"Visibility for {plot_label} on {date}", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-10, 90)

# Bottom plot: Moon illumination
ax2.plot(times.datetime, moon_illumination, color='orange', linewidth=2, label='Moon Illumination')
ax2.fill_between(times.datetime, 0, moon_illumination, alpha=0.3, color='orange')
ax2.set_ylabel("Illumination (%)", fontsize=12)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)

# Add reference lines and labels for moon phases
ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax2.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax2.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.7)

# Add text labels for moon phases (right side)
ax2.text(1.02, 0.02, 'New Moon', transform=ax2.transAxes, fontsize=9, 
         verticalalignment='bottom', color='gray', style='italic')
ax2.text(1.02, 0.52, 'Quarter', transform=ax2.transAxes, fontsize=9, 
         verticalalignment='center', color='gray', style='italic')
ax2.text(1.02, 0.98, 'Full Moon', transform=ax2.transAxes, fontsize=9, 
         verticalalignment='top', color='gray', style='italic')

# Format x-axis with better date/time labels
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d %b'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
ax2.set_xlabel("Time (UTC)", fontsize=12)

plt.tight_layout(rect=[0, 0, 0.98, 1])  # Leave space for labels
plt.savefig('./visibility_test.png', dpi=150)
