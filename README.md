# Automated Limb Motion Analysis (ALMA)

![](https://github.com/sollan/slip_detector/blob/master/Screenshots/ALMA.PNG)

A behavioral data analysis toolbox for motor research in rodents. 

### To get started

The program has been tested on Windows, Mac, and Linux systems with Python 3.8, and should function even on low-end laptops. However, at least 8 GB RAM is recommended. Kinematic parameter extraction, particularly dynamic time warping, is computation-intensive and time-consuming, which will vary depending on your data size and hardware.

0. Installation prerequisites: 
Python 3, pip, (virtual environment is recommended)
1. Download / clone this repository to your computer.
2. Open a terminal and navigate to the folder. 
```bash
cd Downloads/alma
```
3. Install dependencies. The dependencies have been tested for Python 3.8. 
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


### Adjusting the GUI display and experiment setups
Open the config.yaml file with a text editor. Here you'll find the parameters related to the interface display, as well as the default parameters for analyses. If the GUI window isn't displayed properly at first, you can adjust the window size by editting the window width / height values in config.yaml, saving, and restarting the program. 


For more information or support, please visit [our wiki page](https://github.com/sollan/slip_detector/wiki) or contact us.
