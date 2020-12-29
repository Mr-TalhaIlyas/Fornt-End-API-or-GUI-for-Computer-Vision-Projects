# Python GUI/Front-end for you Machine Learning or Computer Vision Projects

This repo contains instructions on how you can make a user friendly GUI for non programmers to use your models for Inference.

I have made two seperate `gui.py` file for;
* object_detection
* segementatio
both directories and `gui.py` has similar sturcture

## Dependencies
* Tkinter
* PIL
* CV2

### Note
**This repo only contains steps for integrating your **ML model** with GUI. So, model codes are not provided.**

### Object Detection

In object detecion case I trained the Faster_RCNN model on paprika dataset* (our lab's). After training I saved model's checkpoints.
you can save your model's **check points** and **config** file in the same directory to use it.

You'll have to make you own pre and post processing function in the inference file you can see the `inference.py' in object detection dir. as an example.

### Sample usage

```python
import your_pack

def pre_processing():
    '''
    Do pre-prcessing here like resize, covnert to tensor etc.
    '''
    return

def post_processing():
    '''
    Do post-prcessing here like NMS, draw boxes
    '''
    return

class SurvedModel:

    def __init__(self):
        '''
        Model should be loaded on memory here.  
        '''
        # self.your_model = ~~

    def predict (self, img):
        '''
        Preprocessing & inference & postprocessing part.
        # img;attribute = {shape:[H, W, 3],  type : ndarray}
        # return;attribute = {shape : [H, W, 3], type : ndarray}

        # return your_postprocessing(self.your_model(your_preprocessing(img)))

        Draw box on the image here. and return that. 
        '''
        # return ~~

```

After this you can just edit titles and information dialogues inside the `gui.py` file to match you own projects descriptions.

### Segmentation

For using segmentation model you can train you model and save it in an '.h5' format as follows

#### Saving your tensorflow/Keras model
Example 
```python
# Save the whole model in .h5 file
tf.keras.models.save_model(model, filepath='/home/user01/data_ssd/Talha/brats/brats_model.h5')

# <<<<<<<<<<<<<<<< Important >>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Before loading the model first define the custom objects you embedded in model.
# e.g. in this case I'll define the following 3 functions before loading the model/
# Also remember you
cl = { 'Weighted_BCEnDice_loss' : Weighted_BCEnDice_loss,
      'mean_iou': mean_iou,
      'dice_coef': dice_coef}
# Now load model
loaded_model = tf.keras.models.load_model(filepath='/home/user01/data_ssd/Talha/brats/brats_model.h5',
                                          custom_objects=cl, compile=True) 
loaded_model.summary()

```
I am using BraTS as in example so, as the scans in BraTS have 4 channels per scan opposed to 3 channels per image so I saved them in `.npy` format. Because they can't be loaded via cv2 or PIL lib If you
are using simple images like **Cityscape** or **PASCAL_VOC** dataset you will simply use image and read them via cv2 or PIL etc.

## Note
* To change the icon of your gui 
** first resize the image you want to use to 64x64px
** then convert it form `.jpg` or `.png` t0 `.ico` file. You can use any online convertor to do so.
* You also need to convert all the images that you are displaying with `tkinter` to range between [0, 255] and dtype `uint8` or else it'll through error.

## Some Functionalities of GUI

First navigate to your `env` and run the `gui.py` file via anaconda cmd.

![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img1.png)

The Model will start loading on your memoery (this will only happen at first run) you can entract with your GUI while model is loading. Because we are using threading to run the GUI in parallel.

![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img2.png)

You can see your `About Us` or `Help` sub-menue to see about what GUI does.
![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img3.png)
![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img4.png)

After the model is done loading if you pressed the `Detect` button it'll through error.

![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img5.png)

Load your image

![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img6.png)

Now press `Detect` Button.

![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img7.png)

![alt text](https://github.com/Mr-TalhaIlyas/Fornt-End-API-or-GUI-for-Computer-Vision-Projects/blob/master/screens/img8.png)
