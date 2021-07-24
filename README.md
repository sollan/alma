# Automated Limb Motion Analysis (ALMA)

![](https://github.com/sollan/slip_detector/blob/master/Screenshots/ALMA.PNG)

A behavioral data analysis toolbox for motor research in rodents. 

### To get started

0. Installation prerequisites: 
Python 3 (The dependencies have been tested for Python 3.8 on Windows, Mac, and Linux systems), pip
1. Download / clone repository
2. Open a terminal and navigate to the folder. 
```bash
cd Downloads/alma
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Ready to go! (Check the command for the correct python version, which could be py, python3, python, ...)
```bash
python ./alma.py
```

### Troubleshoot

wxPython might not install at first if the gtk requirements are not met. 
If you see an error related to wxPython during ```pip install -r requirements.txt```, try 
```
pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk2/ubuntu-16.04/ wxPython
```
(https://wxpython.org/blog/2017-08-17-builds-for-linux-with-pip/index.html)

In case libsdl or libpng12 didn't come with your linux distro (and you get an error when starting the app like ```ImportError: libSDL2-2.0.so.0: cannot open shared object file: No such file or directory```), try the following:
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


### Adjusting the GUI display and experiment setups
Open the config.yaml file with a text editor. Here you'll find the parameters related to the interface display, as well as the default parameters for analyses. If the GUI window isn't displayed properly at first, you can adjust the window size by editting the window width / height values in config.yaml, saving, and restarting the program. 


### Usage

Note: for both analysis, you need to have the bodypart coordinate data, either from DeepLabCut (validated format) or another system. For ladder rung footfall / slip analysis, you need the video recordings in addition for validating the output. 

1. Ladder rung - footfall / slip analysis
- Import the csv output from a DeepLabCut model including the estimated bodypart coordinates (or a spreadsheets containing the same header and column structures as a DLC output file). The file should contain at least the coordinates of the limb endpoints (paw / toe) for which you want to analyze the footfalls / slips. 
- Select the slip prediction algorithm (**threshold**: focus on data that exceeds a y-axis lowerbound; **deviation**: directly estimate peaks using raw location values; **baseline**: detect peaks based on the asymmetric least squares baseline).
- Select the bodyparts to analyze--the predicted footfalls / slips are then sorted, so that you can validate the prediction by skimming through the results alongside the video frames corresponding to the footfalls / slips, alter their starting and ending times for duration and depth calculation, or add additional footfalls / slips that the program missed based on coordinate graphs. 
- If desired, you can mark each identified mistake as either slip or fall for more refined results.
- Export the automatic output or validated results to a csv file.

![](https://github.com/sollan/alma/blob/master/Screenshots/loading.PNG)

_(Select parameters for footfall / slip detection.)_

***

![](https://github.com/sollan/alma/blob/master/Screenshots/validate.PNG)


_(Validate model prediction using the GUI.
Top: a frame from original video.
Bottom: model prediction of the y-axis location of a bodypart throughout the video duration. The location of the bodypart that led to the current slip prediction is displayed in the graph.
This frame is identified as "slip" by the threshold method, based on pose estimation from DLC.)_


2. Treadmill kinematics
- Import the csv output from a DeepLabCut model including the estimated bodypart coordinates (or a spreadsheets containing the same header and column structures as a DLC output file). The file should contain the coordinates of joints on one hind limb, labelled "toe", "mtp", "ankle", "knee", "hip", and "iliac crest". (We are working on additional features to include kinematic analysis with different setups, including front limb and tail, as well as to adapt the program to data in other formats, e.g., when the bodyparts are named differently, and experimental conditions, e.g., spontaneous kinematics without treadmill.)
- Alternatively, select a folder that contains multiple csv files that contain the estimated bodypart coordinates in the appropriate format, for data obtained with the _same_ experimental setup (most importantly, distance from camera, and - if using semi-automated mode - treadmill speed). 
- In semi-automated mode, the program requires the cm/s speed of the treadmill. Additional relevant parameters such as px-cm ratio will be calculated immediately and displayed, prior to starting the analysis, for validation. 
- If the conversion ratio between the px/frame speed and the cm/s speed of the treadmill is known (to be calculated during calibration outside the toolbox or estimated using the semi-automated function using known cm/s treadmill speeds), you can select "Fully automated" mode and input the conversion ratio. 
- Extract kinematic parameters and export the results as csv files.

3. Data analysis
- Random forest: using the extracted kinematic parameters, build a random forest model to classify strides by animal groups (e.g., disease vs. healthy), based on parameters of individual step cycles, then save the results (prediction accuracy, parameter importance ranking, and confusion matrix).
- PCA: using the extracted kinematic parameters, use principal component analysis to reduce dimensionality and cluster data by animal groups (e.g., disease vs. healthy), based on parameters of individual step cycles, then save the results (PCA plot, including the explained variance of two principal components).

For more information or support, please visit [our wiki page](https://github.com/sollan/slip_detector/wiki) or contact us.
