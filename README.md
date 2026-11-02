# Automated Limb Motion Analysis (ALMA) v2.0

## What is new in ALMA v2.0?

- **Modern UI/UX:** Redesigned interface with card-based layouts, step-by-step wizards, and improved visual feedback
- **Interactive timeline:** Pyqtgraph-based interactive timeline for footfall detection, with zoom, pan, and click-to-jump functionality
- **Configurable drag detection:** Adjustable sensitivity for drag detection via consecutive frame threshold
- **Auto-load FPS from video:** Instantly load FPS from video files via both GUI and CLI—no manual entry required
- **Camera-independent spatial calibration:** Uses body segments for robust analysis, insensitive to camera distance
- **Robust bodypart alias detection:** Automatically resolves aliases (e.g., toe/toeR/toeL, crest/iliac crest)
- **Automatic walking direction detection:** Walk direction detected automatically; treadmill correction applied as needed
- **Improved variability metric:** Removed FastDTW dependency—now faster and more compatible
- **Demo data included:** Try ALMA quickly with a provided demo video and matching DLC CSV file

### Check out our paper:
#### Aljovic, A., Zhao, S. _et al._ A deep learning-based toolbox for Automated Limb Motion Analysis (ALMA) in murine models of neurological disorders. _Commun Biol_ 5, 131 (2022). https://doi.org/10.1038/s42003-022-03077-6

![](https://github.com/sollan/slip_detector/blob/master/Screenshots/ALMA.PNG)

A behavioral data analysis toolbox for motor research in rodents. 

### To get started

The program has been tested on Windows, Mac, and Linux systems with Python 3.10, and should function even on low-end laptops. However, at least 8 GB RAM is recommended. Performance will vary depending on your data size and hardware.

0. Installation prerequisites:
Python 3.10, and (mini)conda ([Miniconda documentation](https://docs.conda.io/projects/miniconda/en/latest/#})).
1. Download / clone this repository to your computer.
2. Open a terminal and navigate to the folder.
```bash
cd Downloads/alma
```
3. Create and activate the Conda environment (Python 3.10):
```bash
conda env create -f conda_env_python_3_10.yml
conda activate venv_python_3_10
```


4. Launch ALMA:

**On Windows/Linux/macOS:**
```bash
python ./alma.py
```

### Demo data

We provide a [demo kinematic video and the corresponding DLC CSV](https://github.com/sollan/alma/tree/master/Resources) in the repository so you can quickly try the full pipeline end‑to‑end.

### Treadmill input checks and quality reports

Treadmill analysis validates body-part mapping, numeric coordinates, likelihoods,
frame rate, treadmill speed, filter cutoff, and stride limits before processing.
The current lowpass filter requires at least 19 consecutive frames. Missing or
infinite coordinates produce an explanatory error; they are not interpolated
automatically. The caller's input DataFrame is preserved for repeat analyses.

Each treadmill export includes an `*_analysis_report.json` file containing the
effective settings and belt-motion calibration, confidence summaries, retained
and rejected intervals with rejection reasons, and recording-boundary warnings.
Frame numbers in this report refer to zero-based CSV row positions. A successful
export is not independent validation of the detected gait events. Review the
warnings and the tracking/video before interpreting measurements. If no usable
measurements remain, the interface reports that explicitly.

The treadmill detector and default belt-motion calibration retain their existing
behavior. Drag filtering now counts every frame in a continuous contact run
that meets the requested minimum duration, and excludes shorter runs. This
corrects endpoint truncation and short-run acceptance in the previous filter;
drag duration and percentage can therefore change. Existing low-confidence
coordinates are reported rather than silently removed or replaced.

Run the demo regression and targeted failure checks with:

```bash
python -m unittest discover -s tests -v
```

The baseline fixture records commit `47703fc`, the bundled demo hash and the
settings used. Non-drag kinematic parameters, coordinates and event boundaries
must match that baseline exactly. Drag filtering has separate run-length tests.

### Calibration and parameter effects

Treadmill calibration is now explicit. **Belt-motion estimate** is the default
and preserves the earlier calculations. **Measured pixels per cm** uses an
entered positive spatial scale. **Measured reference segment** uses an entered
positive length for the selected body segment; no anatomical default is assumed.
Reference calibration requires at least 10 jointly confident frames and rejects
segment-length CV above 30%, following the existing helper's variability check.
It reports the confidence threshold and frames used, without silently falling
back to another segment or calibration method. Measured modes require input
before the wizard can continue. Runway analysis currently offers measured manual
calibration only, rather than displaying an unsupported reference option.

The pixel-coordinate treadmill correction continues to follow tracked belt
motion in all modes. In belt-motion mode, entered belt speed and FPS determine
the inferred pixels-per-cm scale. In measured modes, recorded belt speed is a
cross-check: the quality report includes the speed implied by tracked pixel
motion and the measured scale. Changing recorded speed alone in a measured mode
does not change numerical measurements. Choose calibration from the recording,
not from the number of retained strides.

The minimum step-height field is now enforced. Keep it at zero to retain
zero-height dragging steps; a positive value excludes them. Step/stride limits
change the retained cohort, so summary averages and variability groups can
change without altering individual angles on the surviving intervals. Changing
the filter cutoff can change event boundaries and angles as well as smoothing.
Drag clearance/run-length controls affect drag classification within swing.

To reproduce a demo sensitivity audit through the GUI analysis worker:

```bash
python tools/audit_treadmill_parameters.py --output /path/to/audit-output
```

This exports per-case CSVs, event arrays, quality reports, a summary CSV and
detailed JSON. Experiments deliberately include incorrect FPS/direction and
unmeasured example calibration values: their outputs are sensitivity checks,
not recommended settings or validation of detected gait events. Comparisons
align common retained intervals to distinguish measurement changes from cohort
selection. Legacy CSV columns labelled percentage continue to store fractions.

### Confidence-aware runway analysis

Overground analysis scores complete contact-to-contact cycles inside moving
bouts. It no longer requires equal floating-point toe positions in adjacent
frames. Contact detection combines short-window foot velocity, foot height
relative to a local floor estimate, body movement and tracking confidence.
It supports foot movement near the floor, including dragging; persistent
sliding without identifiable support transitions is left unscored rather
than converted into normal cycles. A paused animal or an unfinished entry/
exit interval is not appended as a final stride.

Choose **Spontaneous walking** and enter a measured pixels-per-cm scale.
The API now also requires an explicit positive `pixels_per_cm`; there is no
assumed physical scale. Map one limb's `toe` and at least one body reference
(`hip` or `iliac crest`). Other joints are optional: their absence makes
dependent metrics unavailable, while usable timing is retained. Analyze
bilateral tracks separately with unique mappings. Source frame-label columns
must be consecutive; missing frames should remain represented by rows.

Runway controls are separate from treadmill settings:

- **Tracking confidence:** masks unreliable coordinates before filtering.
- **Maximum gap for event timing:** only internal short gaps within this
  duration may be interpolated for foot/body event detection and foot-trajectory
  metrics. Set zero to
  disable it. Long gaps split valid segments; original frame indices remain.
- **Minimum body speed:** moving-bout gate in reference lengths per second.
  One reference length is the median valid toe-to-body-reference separation,
  not a species-specific anatomical length. An adaptive tracking-noise floor
  also applies. Review this setting for very slow or impaired locomotion.
- **Minimum contact duration:** suppresses brief contact-state fluctuations.
  Frame counts are resolved from seconds and recording FPS in the report.

Contact timing uses a short quadratic smoother (25 ms by default, rounded
to an odd window of at least three samples). Its actual frame window is
recorded. Kinematic coordinates and angles use the selected lowpass cutoff,
applied separately to finite segments longer than the filter padding. An optional
separate joint cutoff controls joint angles and toe-to-crest distance while the
main cutoff continues to control body/foot processing. No
long gaps are crossed. Joint summaries use observed-only coordinates by default.
Optional joint reconstruction has separate settings; event interpolation is
never silently substituted into joint-angle summaries. A missing knee affects ankle/knee/hip angles that
depend on it, rather than erasing an entire trial's timing. The runway
toe-to-crest distance columns use the actual iliac crest, without substituting
the hip. Foot lengths/heights and event timing that use bounded interpolation
are explicitly identified by the coordinate-source and quality fields.

Parameter CSVs retain the existing 47-column runway schema. A new interval's
end is the next contact frame, with durations and metrics using `[start,end)`;
first/last partial contacts are discarded. Five/ten-cycle variability uses
only consecutive retained intervals in one bout, with unavailable-window
reasons recorded. The existing variability implementation remains a pairwise
trajectory-difference metric after time interpolation, in pixel coordinates.
Legacy percentage columns continue to contain fractions from 0 to 1.

Coordinate CSVs preserve every input row and include phase, candidate/retained
interval identifiers, drag flags, observed/metric validity, and event-gap
interpolation flags. Phase codes are `-1` unscored, `0` foot movement and `1`
support. Initial/last partial-bout phases may be visible without a scored
cycle. Drag is NaN outside retained intervals, not a false negative. `toe
event x/y` describe the short-smoothed event signal; `toe analysis x/y` describe
the lowpass foot trajectory, including bounded event gaps. Standard joint
`x/y` columns are the selected filtered coordinates used for angles. `raw x/y`
preserve input coordinates, and `observed filtered x/y` preserve the unfilled,
confidence-masked comparison. Likelihoods are never replaced. `metric
interpolated` flags directly reconstructed frames; `metric interpolation
context` flags a whole finite filter run influenced by reconstruction.

The quality JSON records bouts, effective thresholds, partial-contact counts,
rejection reasons, interpolation, per-metric availability and variability
windows. Low-confidence scores alone do not verify anatomical correctness of
supplied or reconstructed coordinates. Validate detected events against video
and use measured calibration before interpreting physical units. Runway
results intentionally differ from the previous equality-based detector;
treadmill numerical calculations are separate and regression-tested.

For local batch processing:

```bash
python tools/analyze_runway_csv.py input.csv --fps 120 --pixels-per-cm 100 --output results
```

Replace the example FPS and scale with recording measurements. The optional
`no_outlier_filter` API/config flag bypasses only height/length ranges, not
bout or tracking-quality checks. `dragging_filter=True` explicitly excludes
intervals with at least 95% near-ground movement; it is off by default.

### Optional short-gap joint reconstruction

In spontaneous walking, select **Short-gap linear interpolation** or
**Short-gap PCHIP interpolation** under joint coordinate reconstruction.
**Observed only** remains the default. API options are `joint_interpolation`
(`none`, `linear`, `pchip`), `joint_max_gap_s` (default 0.02),
`joint_max_interpolated_fraction` (0.10), and `joint_max_segment_change` (0.35).
These options affect joint angles and toe-to-crest distance, independently of
stride detection, foot-trajectory metrics and treadmill calculations.

Reconstruction is applied to confidence-masked x/y coordinates before filtering.
Only bounded internal gaps up to the time limit can be filled. PCHIP uses the
two endpoints and, when available, one additional consecutive observation on
each side. No long gap is crossed to obtain those additional observations.
Recording edges are never extrapolated. A proposal must have at least one
adjacent joint with observed endpoint coordinates and an observed sample in
the gap for a geometry check. Proposed segment lengths must remain within the
configured relative deviation from locally interpolated endpoint lengths;
strong disagreement between endpoint lengths also rejects a proposal. No
species-specific anatomical length is assumed. A separate motion check rejects
steps exceeding nearby observed motion limits when enough samples exist.
Every filled/rejected gap and its checks are recorded in the quality JSON.

For each angle or distance, the union of directly filled frames across its
required joints must stay within the per-interval reconstruction fraction.
Complete filtered coverage is still required; longer gaps and excess fractions
leave only dependent metrics unavailable, while timing remains scored. The
parameter CSV schema is unchanged. Per-interval `metric_quality` records
observed/reconstructed coverage, estimated status, filter context, and whether
flexion/extension occurs at a directly reconstructed frame.
`observed_only_metrics` preserves the unfilled summary or `null` when unavailable.
Filtering can change surrounding observed samples too, so an estimated peak
can occur outside a directly filled frame. Dashed stick-plot bones identify
reconstructed endpoints. The estimates are not independently verified anatomy.

### Reviewing tracking and smoothing

Runway GUI stick plots default to **Tracked keypoints (unfiltered review)**.
They draw finite input coordinates directly, so filter padding cannot hide an
otherwise visible tracked point. Gray dotted bones identify at least one
low-confidence endpoint; these samples remain excluded from calculations.
Missing raw coordinates remain missing. **Analysis coordinates (filtered)**
shows the selected filtered coordinates used for joint calculations, including
dashed reconstructed endpoints. The plot title identifies the coordinate source.
Changing this display choice does not change parameter values. API callers can
choose `return_continuous(..., coordinate_source='tracked')`; its existing
`analysis` default is retained for compatibility.

For runway analysis, **Separate joint filter cutoff** is optional and inherits
the main cutoff when unchecked. API `joint_cutoff_f=None` also inherits it;
`tools/analyze_runway_csv.py --joint-cutoff 10` explicitly selects 10 Hz for
joint coordinates. The effective cutoff is recorded in the quality JSON.
Higher cutoffs retain faster motion and more tracking noise. Compare tracking,
filtered trajectories and angle peaks against video before selecting a cutoff;
closeness to tracked dots alone does not establish anatomical accuracy. Changing
only the joint cutoff preserves stride detection and foot-trajectory metrics.
The default calculation and treadmill numerical behavior remain unchanged.

Validate settings against video and known observations before relying on
estimated peaks. The audit below masks known coordinates and compares linear
and PCHIP reconstruction, including coordinate error and filtered angle/peak
errors. Its built-in articulated motion has known coordinates. With `--csv`,
the reference consists of confident tracked positions, which may themselves
contain errors. The audit counts rejected windows and uses identical held-out
positions for both methods; real recordings may lack enough continuous,
confident coverage for angle-peak comparisons.

```bash
python tools/audit_joint_interpolation.py --output reconstruction_audit
python tools/analyze_runway_csv.py input.csv --fps 120 --pixels-per-cm 100 --joint-interpolation linear --joint-max-gap-ms 20 --output results
```

Replace FPS and calibration with recording measurements. Reconstruction limits
are configurable starting values, not universal validation thresholds. Existing
treadmill outputs and observed-only runway calculations remain numerically
unchanged when joint reconstruction is disabled.

### Important: Data quality requirements

#### DeepLabCut labeling and recording setup

For accurate kinematic analysis, ensure:

1. **Side-view recording:** Position your camera to capture a **clear lateral (side) view** of the animal, similar to the demo video. The recording should show:
   - Full visibility of the limb from toe to hip/iliac crest
   - Perpendicular camera angle (90° to the direction of movement)
   - Consistent distance from the camera throughout the recording

2. **DeepLabCut labeling quality:** ALMA's accuracy depends on DLC tracking quality:
   - Label all required bodyparts: **toe, MTP, ankle, knee, hip, iliac crest**
   - Validate tracking on a few frames before bulk analysis
   - Poor tracking (jittery coordinates, bodypart swaps) will affect kinematic parameters

3. **Supported bodypart naming:** ALMA automatically detects common aliases:
   - Toe: `toe`, `toeR`, `toeL`, `toe_r`, `toe_l`
   - Iliac crest: `iliac crest`, `crest`, `crestR`, `iliacR`
   - Other bodyparts follow similar patterns (see paper for full list)

### Understanding kinematic settings

ALMA v2.0 provides extensive control over analysis parameters. Here's what each setting does and why it matters:

#### Experimental setup
- **Treadmill vs. Spontaneous:** Choose based on your recording setup
  - *Treadmill:* Constant belt speed, requires treadmill speed input
  - *Spontaneous:* Free walking with variable speed

#### Speed & calibration (Treadmill)
- **Treadmill speed (cm/s):** Belt speed of your treadmill
  - Critical for accurate stride length and velocity calculations
  - Example: 30 cm/s is typical for mouse treadmill studies
  
- **Frame rate (fps):** Video recording frame rate
  - Affects temporal resolution of all measurements
  - Use "Load from Video" button for automatic detection

#### Spatial calibration method
- **Reference body segment (Recommended):**
  - Uses known anatomical distances (e.g., ankle-toe = 1.5cm in mice)
  - **Camera-independent:** Works regardless of camera distance
  - Most accurate for multi-session studies
  - Default: ankle-toe segment (1.5cm for adult mice, or measure ankle-toe distance manually 1.5 is just an estimation)

- **Manual pixel-to-CM ratio:**
  - Only use if you have pre-calculated the conversion
  - Requires consistent camera setup across recordings

#### Movement analysis settings
- **Walking direction:** 
  - Auto-detect (recommended): Algorithm determines direction automatically
  - Manual override available if auto-detection fails

- **Drag clearance threshold (cm):**
  - Minimum toe height above ground to NOT count as dragging
  - Default: 0.1cm for mice
  - Lower values = more sensitive to ground contact
  - Adjust based on your research question (e.g., 0.05cm for subtle drags)

- **Drag detection sensitivity (frames):**
  - Minimum consecutive frames of ground contact to count as dragging
  - Default: 4 frames (balances sensitivity and noise)
  - Lower (1-2): Detects brief touches; may include tracking jitter
  - Higher (6-10): Only sustained dragging; reduces false positives
  - **Scientific impact:** Directly affects "drag duration" and "drag percentage" in output CSV

- **Lowpass filter cutoff (Hz):**
  - Butterworth filter frequency for smoothing coordinate data
  - Default: 6Hz for mouse gait
  - Lower values = more smoothing (reduces noise but may blur rapid movements)
  - Higher values = preserve fast movements (but include more noise)

#### Stride filtering (Optional)
- **Step height range (cm):** Filter out strides outside this range
  - Excludes abnormally low/high steps
  - Default: 0.0 - 2.0 cm for mice

- **Stride length range (cm):** Filter out strides outside this range
  - Excludes abnormally short/long strides
  - Default: 0.0 - 8.0 cm for mice

**Why these parameters matter:**
- They directly affect quantitative outputs (stride length, drag percentage, joint angles)
- Consistent settings are essential for comparing across animals/conditions
- Document all parameter values in your methods section for reproducibility
- The default values work well for adult mice on treadmills, but may need adjustment for:
  - Different species (rats, other rodents)
  - Injured/impaired animals (adjust drag and filtering thresholds)
  - Different locomotor tasks (overground vs. treadmill)

### Troubleshoot

**PySide6 installation issues:**
If you encounter issues with PySide6 installation, try:
```
pip install --upgrade pip
pip install pyside6
```

For Linux systems, you may need additional dependencies:
```
sudo apt-get install libxcb-xinerama0
```

In some cases (particularly some Linux distributions), you might need to install some libraries such as libsdl or libpng12. This problem can be identified by errors when starting the app, such as```ImportError: libSDL2-2.0.so.0: cannot open shared object file: No such file or directory```). 
Try to solve the ```ImportError``` accordingly, e.g.,
```
sudo apt-get install libsdl2-2.0
```
and,
```
sudo add-apt-repository ppa:linuxuprising/libpng12
sudo apt update
sudo apt install libpng12-0
```

If you run into problem installing, please open an issue with the details (error traces) and your system specs (operating system, python version, ...) on Github, and we will get back to you.


For more information or support, please visit [our wiki page](https://github.com/sollan/slip_detector/wiki) or contact us.
