import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1";
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import matplotlib
import matplotlib.pyplot as plt
import tarfile
import wget
import io
import scipy.misc
import numpy as np
from six import BytesIO
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
#%matplotlib inline
mpl.rcParams['figure.dpi'] = 300
import random
import seaborn as sns
import numpy as np
from tqdm import tqdm, trange
from PIL import Image, ImageDraw, ImageFont
import re
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
classes = [ "blossom_end_rot", "graymold","powdery_mildew","spider_mite","spotting_disease"]

def draw_boxes(image_in, confidences, nms_box, class_names, all_classes):
    '''
    Parameters
    ----------
    image : RGB image original shape will be resized
    confidences : confidence scores array, shape (None,)
    boxes : all the b_box coordinates array, shape (None, 4) => order [y_min, x_min, y_max, x_max]
    classes : shape (None,), names  of classes detected
    all_classes : all classes names in dataset
    '''
    img_h = 640
    img_w = 640
    # rescale and resize image
    image = cv2.resize(image_in, (img_w, img_h))/255
    boxes = np.empty((nms_box.shape))
    # form [y_min, x_min, y_max, x_max]  to [x_min, y_min, x_max, y_max]
    boxes[:,1] = nms_box[:,0] * img_h
    boxes[:,0] = nms_box[:,1] * img_w
    boxes[:,3] = nms_box[:,2] * img_h 
    boxes[:,2] = nms_box[:,3] * img_w 
    # Unnormalize
    boxes = (boxes).astype(np.uint16)
    i = 1
    colors = sns.color_palette("bright") + sns.color_palette("Paired")
    for result in zip(confidences, boxes, class_names, colors):
        conf = float(result[0])
        facebox = result[1].astype(np.int16)
        #print(facebox)
        name = result[2]
        color = colors[classes.index(name)]#result[3]
    
        cv2.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), color, 2)#255, 0, 0
        label = '{0}: {1:0.3f}'.format(name.strip(), conf)
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX   , 0.7, 1)

        cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),    # top left cornor
                     (facebox[0] + label_size[0], facebox[1] + base_line),# bottom right cornor
                     color, cv2.FILLED)#0, 0, 255
        op = cv2.putText(image, label, (facebox[0], facebox[1]),
                   cv2.FONT_HERSHEY_DUPLEX   , 0.7, (0, 0, 0)) 
        i = i+1
    return image, boxes, class_names, np.round(confidences, 3)
# build detection funciton with pre and post processing image
def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn


#%%

class SurvedModel:

    def __init__(self):
        '''
        Model should be loaded on memory here.  
        '''
        self.pipeline_config =  'C:/Users/Talha/Desktop/chkpt/paprika_model/pipeline_file.config'
        self.chkpt_dir = 'C:/Users/Talha/Desktop/chkpt/paprika_model/ckpt-34'
        configs = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        model_config = configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.chkpt_dir)
        
        self.detect_fn = get_model_detection_function(detection_model)
    
        
    def predict (self, img):
        '''
        Preprocessing & inference & postprocessing part.
        # img;attribute = {shape:[H, W, 3],  type : ndarray}
        # return;attribute = {shape : [H, W, 3], type : ndarray}
        
        # return your_postprocessing(self.your_model(your_preprocessing(img)))
        '''
        print('start from inside')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert to tensor
        img_tensor = tf.convert_to_tensor(img[np.newaxis, :,:,:], dtype=tf.float32)
        # pass through network for detection
        detections, predictions_dict, shapes = self.detect_fn(img_tensor)
        # extract the relative ouputs form dictionaries
        # box coords order [y_min, x_min, y_max, x_max]
        box = detections['detection_boxes'][0].numpy()
        clas = (detections['detection_classes'][0].numpy()).astype(int)
        prob =detections['detection_scores'][0].numpy()
        
        # apply non_max_suppression
        nms_ind = tf.image.non_max_suppression(box, prob, 200, iou_threshold=0.3,score_threshold=0.2, name=None).numpy()
        
        nms_box = box[nms_ind]
        nms_prob = prob[nms_ind]
        nms_clas = clas[nms_ind]
        labels = []
        for i in range(len(nms_clas)):
                labels.append(classes[nms_clas[i]])
                
        op, boxes, class_names, confidences = draw_boxes(img, nms_prob, nms_box, labels, classes)
        op = (op*255).astype(np.uint8)
        #cv2.imwrite('/data_ssd3/Talha/TFOD/op/{}'.format(name), cv2.cvtColor((op*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        #op = np.asarray(op).astype(np.uint8)
        print('sent op')
        return op
        
#%%

# Usage

# model = SurvedModel()
# # Load Image
# img = cv2.imread('C:/Users/Talha/Desktop/paprika old/spotting_disease/spotting_disease_1222.jpg')
# op = model.predict(img)
# plt.imshow(op)