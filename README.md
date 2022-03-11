# Face Detection with MTCNN
 

* 🚀🌟 This repository can run on real time, video and image source. 
* 🚀🌟 If you want to use your custom data, edit the paths in the settings.py file and and RUN! 

### Get this repo and install requirements

`git clone https://github.com/suhedaras/face_detection_with_MTCNN.git`

`cd face_detection_with_MTCNN`

**Before, you create conda environment**

`conda create -n mtcnn python==3.8`

`conda activate mtcnn`

`pip install -r requirements.txt`


**Run this command for face detection, you can use source_path = 0 for webcam**

```
python mtcnn.py
```

**Run this command to generate dataset using detected face landmarks(4 different folders: left eye, right eye, face and mouth). Generated dataset is saved in the results folder**

```
python create_dataset.py
```

**The created data set can be used in driver fatigue detection.**

