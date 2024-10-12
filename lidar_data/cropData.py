import re
import random
import numpy as np
import os.path
import cv2
from glob import glob

# from tqdm.auto import tqdm
import math

def crop_images_within_folder(path, count):
    image_paths = glob(os.path.join(path, 'velodyne_bv_road', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(path, 'gtbv2', '*_road_*.png'))}

    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]
        image = cv2.resize(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), (200, 400))
        gt_image = cv2.resize(cv2.cvtColor(cv2.imread(gt_image_file), cv2.COLOR_BGR2RGB), (25, 100), interpolation=cv2.INTER_AREA)
        #gt_image[gt_image < 15] = 0
        #gt_image[gt_image > 0] = 100
        cv2.imwrite('./croppedData/velodyne_bv_road/um_{0:06}.png'.format(count), image)
        cv2.imwrite('./croppedData/gtbv2/um_road_{0:06}.png'.format(count), gt_image)
        count = count + 1
    return count

