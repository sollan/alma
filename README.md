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

We provide a demo kinematic video and the corresponding DLC CSV in the repository so you can quickly try the full pipeline end‑to‑end.

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
