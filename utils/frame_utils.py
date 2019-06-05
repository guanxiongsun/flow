import numpy as np
from os.path import *
from scipy.misc import imread
from . import flow_utils 
import xml.etree.ElementTree as ET


def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.JPEG' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:, :, :3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    # TODO
    elif ext == '.xml':
        tree = ET.parse(file_name)

        size = tree.find('size')
        H = float(size.find('height').text)
        W = float(size.find('width').text)

        objs = tree.findall('object')
        boxes = np.zeros((len(objs), 4), dtype=np.uint16)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = np.maximum(float(bbox.find('xmin').text), 0)
            y1 = np.maximum(float(bbox.find('ymin').text), 0)
            x2 = np.minimum(float(bbox.find('xmax').text), H - 1)
            y2 = np.minimum(float(bbox.find('ymax').text), W - 1)

            x = x2
            y = y2
            h = y1 - y2
            w = x1 - x2

            boxes[ix, :] = [x, y, h, w]

        return boxes
    return []
