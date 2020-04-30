"""
@author: Vincent Bonnet
@description : useful image functions
"""

import numpy as np
import skimage

def resize_image_and_keep_ratio(image, width, height):
    '''
    Resize an image while keeping its ratio
    '''
    out = np.zeros((width, height), dtype=np.float32)
    scaleX = image.shape[0] / width
    scaleY = image.shape[1] / height
    maxScale = max(scaleX, scaleY)
    newWidth = np.int(image.shape[0] / maxScale)
    newHeight = np.int(image.shape[1] / maxScale)
    tmpImage = skimage.transform.resize(image, (newWidth, newHeight))
    offsetX = np.int((width - tmpImage.shape[0]) / 2)
    offsetY = np.int((height - tmpImage.shape[1]) / 2)
    out[offsetX:offsetX+tmpImage.shape[0], offsetY:offsetY+tmpImage.shape[1]] = tmpImage
    return out
