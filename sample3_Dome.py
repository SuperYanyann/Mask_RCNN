import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2

def apply_maskX(masked_image, mask):
    for m in range(row):
        for n in range(col):
            if(mask[n, m] == True):
                #frontImage[n, m] = 255;
                masked_image[n,m,0] =255;
                masked_image[n, m, 1] = 255;
                masked_image[n, m, 2] = 255;
            else:
                masked_image[n,m,0] = 0;
                masked_image[n, m, 1] = 0;
                masked_image[n, m, 2] = 0;
    return masked_image

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/jdai/test_code_all/mask_RCNN/Mask_RCNN-master/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import samples.coco.coco as coco

# matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images_jdNewData")
IMAGE_DIR = os.path.join("/home/jdai/datasets/jd-fashion/hyq_img/#U9aa8#U611f(#U5c11#U5973_#U4f11#U95f2_#U7b80#U7ea6_#U8fd0#U52a8)")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
len_files = len(file_names)

# use to change all photos
for loop1 in range(1,len_files):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[loop1]))

    results = model.detect([image], verbose=1)
    # Visualize results
    r = results[0]
    print ('--------------------')
    print (file_names[loop1])
    print ('--------------------')
    print (np.shape(file_names))
    #skimage.io.imsave('test.jpg',r)
    #plt.savefig("examples.jpg")
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])

    boxes  = r['rois'];
    masks  = r['masks'];
    scores = r['scores'];
    class_ids = r['class_ids'];

    # Number of instances
    N = boxes.shape[0];
    N = 1;
    row = image.shape[1];
    col = image.shape[0];
    for i in range(N):
        # Bounding box
        if not np.any(boxes[i]):
            continue;
        y1, x1, y2, x2 = boxes[i];

        # Label
        class_id = class_ids[i];
        score = scores[i] if scores is not None else None
        label = class_names[class_id];

        # Mask
        mask = masks[:, :, i];
        masked_image = np.zeros((col, row, 3), dtype=np.uint8);
        masked_image = apply_maskX(masked_image, mask);

        #frontImage = np.zeros( (col, row), dtype=np.uint8 );
        frontImage = image.copy();
        for m in range(row):
            for n in range(col):
                if(masked_image[n, m, 0]<254):
                    #frontImage[n, m] = 255;
                    frontImage[n,m,0] = 255;
                    frontImage[n, m, 1] = 255;
                    frontImage[n, m, 2] = 255;
        #roiMask = masked_image[y1:y2, x1:x2];
        roiImg = frontImage[y1:y2, x1:x2];
        roiImg = cv2.cvtColor(roiImg, cv2.COLOR_BGR2RGB);

        fileName = file_names[loop1]
        fileName = "/home/jdai//datasets/jd-fashion/style-recog-samples/sample_img/hyq2_img/laber14/"+fileName
        #fileMask = fileName[0: len(fileName)-4];
        #fileMask = fileMask +"."+ str(i)+"."+label+"."+"Mask.png";

        cv2.imwrite(fileName, roiImg);
