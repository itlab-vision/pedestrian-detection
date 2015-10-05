# Expected image size for some operations is 96x160

import sys
from os import listdir
from os import makedirs
from os.path import isfile, join
from os.path import exists
import numpy as np
import cv2
import random as rnd

usage = "<-in-dir> <dir name> <-out-dir> <dir name>\n\
in-dir - directory that contains RGB images\n\
out-dir - directory where YUV images will be stored to"

def main():
    in_dir, out_dir = parseCmdArgs()
    files_to_proc = getFilesToProcess(in_dir)
    processFiles(files_to_proc, in_dir, out_dir)

def getFilesToProcess(in_dir):
    return [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

def processFiles(files_to_proc, in_dir, out_dir):
    if not exists(out_dir):
        makedirs(out_dir)

    for fname in files_to_proc:
        src_img = cv2.imread(join(in_dir, fname))
        # yuv_img = cvtRGB2YUV(src_img)

        # Generate 126x78 images with context ratio 1.4
        resized_img = resizeImage(src_img)
        saveImage(resized_img, out_dir, fname)

        # Mirror along the horizontal axis
        flipped_img = flipImage(resized_img)
        saveImage(flipped_img, out_dir, generateFileName(fname, 'f'))

        # 5 random variations: translations (in range [-2, 2]) and scale (in range ([0.95, 1.05]))
        for i in range(0, 5):
            action = rnd.random()
            if (action > 0.5):
                translated_img = translateImage(resized_img, rnd.uniform(-2, 2), rnd.uniform(-2, 2))
                translated_img_f = translateImage(flipped_img, rnd.uniform(-2, 2), rnd.uniform(-2, 2))
                saveImage(translated_img, out_dir, generateFileName(fname, 't' + str(i)))
                saveImage(translated_img_f, out_dir, generateFileName(fname, 'tf' + str(i)))
            else:
                scaled_img = scaleImageInFizedSize(resized_img, rnd.uniform(0.95, 1.05), rnd.uniform(0.95, 1.05))
                scaled_img_f = scaleImageInFizedSize(flipped_img, rnd.uniform(0.95, 1.05), rnd.uniform(0.95, 1.05))
                saveImage(scaled_img, out_dir, generateFileName(fname, 's' + str(i)))
                saveImage(scaled_img_f, out_dir, generateFileName(fname, 'sf' + str(i)))

        # scaled_img = scaleImageInFizedSize(resized_img, 0.957980871028,  1.02162841581)
        # saveImage(scaled_img, out_dir, generateFileName(fname, 'test'))

        print 'Processed image: ' + join(in_dir, fname)

def cvtRGB2YUV(src_img):
    return cv2.cvtColor(src_img, cv2.COLOR_BGR2YUV)

def resizeImage(src_img):
    croped_img = src_img[2:156, 0:96]
    resized_img = cv2.resize(croped_img, (78, 126))
    return resized_img

def flipImage(src_img):
    flipped_img = cv2.flip(src_img, 1)
    return flipped_img

def translateImage(src_img, x_shift, y_shift):
    size = (src_img.shape[0], src_img.shape[1])
    dst_img = np.zeros((src_img.shape), src_img.dtype)
    for y in range(0, size[0]):
        for x in range(0, size[1]):
            x1 = 0
            y1 = 0
            x1 = x + x_shift
            if (x1 > size[1] - 1):
                x1 = size[1] - 1
            if (x1 < 0):
                x1 = 0
            y1 = y + y_shift
            if (y1 > size[0] - 1):
                y1 = size[0] - 1
            if (y1 < 0):
                y1 = 0
            dst_img[y, x] = src_img[y1, x1]
    return dst_img

def scaleImageInFizedSize(src_img, x_scale, y_scale):
    resized_img = cv2.resize(src_img, (0, 0), fx = x_scale, fy = y_scale)
    # print 'resized img shape:', resized_img.shape
    x_diff = (src_img.shape[1] - resized_img.shape[1]) / 2
    y_diff = (src_img.shape[0] - resized_img.shape[0]) / 2
    if (x_diff < 0):
        x_diff = 0
    if (y_diff < 0):
        y_diff = 0
    top = y_diff
    bottom = src_img.shape[0] - resized_img.shape[0] - y_diff
    left = x_diff
    right = src_img.shape[1] - resized_img.shape[1] - x_diff
    if (bottom < 0):
        bottom = 0
    if (right < 0):
        right = 0
    ret_img = addReplicatedBorder(resized_img, top, bottom, left, right)
    # print 'params: ', x_scale, ', ', y_scale, 'img_shape: ', ret_img.shape
    return ret_img

def addReplicatedBorder(src_img, top, bottom, left, right):
    dst_img_size = (src_img.shape[0] + top + bottom, src_img.shape[1] + left + right, src_img.shape[2])
    dst_img = np.zeros(dst_img_size, src_img.dtype)
    dst_img[top : dst_img.shape[0] - bottom, left : dst_img.shape[1] - right] = src_img
    for y in range(0, top):
        for x in range(0, dst_img.shape[1]):
            dst_img[y, x] = dst_img[top, x]
    for y in range(dst_img.shape[0] - bottom, dst_img.shape[0]):
        for x in range(0, dst_img.shape[1]):
            dst_img[y, x] = dst_img[dst_img.shape[0] - bottom - 1, x]
    for x in range(0, left):
        for y in range(0, dst_img.shape[0]):
            dst_img[y, x] = dst_img[y, left]
    for x in range(dst_img.shape[1] - right, dst_img.shape[1]):
        for y in range(0, dst_img.shape[0]):
            dst_img[y, x] = dst_img[y, dst_img.shape[1] - right - 1]
    return dst_img

def saveImage(image, out_dir, fname):
    print image.shape
    cv2.imwrite(join(out_dir, fname), image)

def generateFileName(src_fname, appendix):
    chunks = src_fname.split('.')
    dst_fname = chunks[0] + appendix + '.' + chunks[1]
    return dst_fname

def parseCmdArgs():
    if (len(sys.argv) >= 5):
        in_dir = ''
        out_dir = ''
        for i in range(0, len(sys.argv) - 1):
            if (sys.argv[i] == '-in-dir'):
                in_dir = sys.argv[i + 1]
            elif (sys.argv[i] == '-out-dir'):
                out_dir = sys.argv[i + 1]
    else:
        print usage
        sys.exit()

    return in_dir, out_dir

if __name__ == "__main__":
    main()
