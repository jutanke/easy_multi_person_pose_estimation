# Easy Multi-Person Pose Estimation
My ugly oop'ification from [Zhe Cao's](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) code using the model from [Michal Faber](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation). 
Please respect the initial authors licenses.
Install as follows:

## Install

* Python 3.6
* Keras 2

```bash
pip install git+https://github.com/jutanke/easy_multi_person_pose_estimation
```

## Sample usage:

```python
import cv2
from poseestimation import model

img_path = './julian.JPG'
I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
# I can either be a single image or a list of images

pe = model.PoseEstimator()

# end-to-end prediction of all detected poses
positions = pe.predict(I)

# extract the heatmaps and part affinity fields
heatmaps, pafs = pe.predict_pafs_and_heatmaps(I) 
```

![pred](https://user-images.githubusercontent.com/831215/39521597-78e40f24-4e0f-11e8-8bd1-3092ab4ad63e.png)
