# rtavis

This repository hosts a custom visibility tool `rtavis` for real-time analysis. 


# Environment and package installation

To create a virtual environment with all required dependencies:

```bash
conda env create --name <envname> --file=environment.yaml
```

Note that you should already have anaconda installed: https://www.anaconda.com/

You can then proceed to install the software:

```bash
pip install .
```

for editable installation:

```bash
pip install -e .
```

## Instrument Response Functions

To complete the environment be sure to download and install the correct IRFs (only prod2 comes with ctools installation). Public IRFs can be downloaded from here: https://www.cta-observatory.org/science/cta-performance/

## Configuration 

Under `cfg` you can find a sample configuration file. Description of each parameter is commented within. This file will serve as input when running the code.

## Compute source visibility

After adjusting the configuration file to your needs, you can run the code as follows:

```bash
python runCatVisibility.py -f cfg/config.yaml
```

### Reading the visibility table

The output is saved via numpy as a binary NPY file. You can run an example of how to access data like this:

```bash
python readVisTable -f path/to/output.npy
```

### Notebook: plot the visibility
A notebook for useful plot and checks on the visibility is provided in the *notebooks* folder.

## Check source and Moon visibility in a time window (ASTRI)

A dedicated script is available to check source visibility at ASTRI over a custom UTC time window, including Moon information:
- Moon presence (altitude)
- Moon-source angular distance (if a source is provided)
- Moon illumination
- Moon phase

Script:

```bash
python rtavis/checkVisibilityWindow_ASTRI.py --help
```

### Main options

- `--start` and `--end`: UTC time window
- `--date`: convenience single-night mode (`YYYY-MM-DD`)
- `--moon-only`: evaluate Moon only, without a source
- `--source` or `--source-name`: resolve source name
- `--ra` and `--dec`: provide source coordinates directly
- `--plot`: save output plot

### Example commands

Moon only, single night:

```bash
python rtavis/checkVisibilityWindow_ASTRI.py --moon-only --date "2025-12-11" --plot moon_night.png
```

Moon only, custom window:

```bash
python rtavis/checkVisibilityWindow_ASTRI.py --moon-only --start "2026-04-28" --end "2026-05-10" --plot moon_window.png --phase-label-step-deg 45
```

With source name:

```bash
python rtavis/checkVisibilityWindow_ASTRI.py --start "2026-04-28" --end "2026-05-10" --plot source_window.png --source "Mrk 501"
```

With source coordinates:

```bash
python rtavis/checkVisibilityWindow_ASTRI.py --start "2026-04-28" --end "2026-05-10" --plot source_window.png --ra "16:56:28" --dec "+39:45:36"
```
