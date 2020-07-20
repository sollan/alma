# Slip detector GUI

An add-on for DeepLabCut-assisted behavioral analysis. 

### To get started

```bash
pip install -r requirements.txt
python ./slip_detector.py
```
We will hopefully provide a package installer in the near future (for a more mature version).


### Usage

In Action - Validation:
- import the csv output from a DeepLabCut model (or a spreadsheets containing the same header and column structures as a DLC output file),
- select the slip prediction algorithm (**threshold**: focus on data that exceeds a y-axis lowerbound; **deviation**: directly estimate peaks using raw location values; **baseline**: detect peaks based on the asymmetric least squares baseline),
- select the bodyparts to analyze--the predicted slips are then sorted, so that the user can validate the prediction by skimming through the results only once, 
- automatically identify "slips" during the ladder rung experiment, including the number of slips, slip depths and the on- and offset of each slip, and
- save the predicted slip properties as a csv file.


![](https://github.com/sollan/slip_detector/blob/master/Screenshots/validation%20start%20screen.png)

_(Select parameters for slip detection.)_

***


During validation, you can
- have a frame-by-frame comparison of the model-predicted bodypart location alongside the original video,
- manually correct the slip detection results, including removing false positives and adding missed slips, and
- save the validated results as a csv file. 

![](https://github.com/sollan/slip_detector/blob/master/Screenshots/validation%20main%20page.png)


_(Validate model prediction using the GUI.
Top: a frame from original video.
Bottom: model prediction of the y-axis location of a bodypart throughout the video duration. The location of the bodypart that led to the current slip prediction is displayed in the graph.
This frame is identified as "slip" by the threshold method, based on pose estimation from DLC.)_



For more information, please visit [our wiki page](https://github.com/sollan/slip_detector/wiki).
