# Slip detector GUI

An add-on for DeepLabCut-assisted behavioral analysis. 

### To get started
```bash
pip install -r requirements.txt 
# not including deeplabcut-related packages but would be
# sufficient for validating / predicting slips from 
# DLC-generated csv files
python ./wx_app.py
```
We will hopefully provide a package installer in the near future (for a more mature version).


### Current status

In Action - Validation, the user can

- import the csv output from a DeepLabCut model (or a spreadsheets containing the same header and column structures as a DLC output file),
- have a frame-by-frame comparison of the model-predicted bodypart location alongside the original video,
- automatically identify "slips" during the ladder rung experiment, including the number of slips, slip depths and the on- and offset of each slip,
- save the predicted slip properties as a csv file,
- manually correct the slip detection results, including removing the false positives and adding undetected slips, and
- save the validated results as a csv file.

For more information, please visit [our wiki page](https://github.com/sollan/slip_detector/wiki).
