import sys
from os.path import isdir, isfile, join
from os import listdir, walk
import cv2
import numpy as np
import re

def extract_pedestrians(in_img_fnames, in_img_dir, annotation_fnames, annotation_dir, out_img_dir, desired_size):
    for img_fname in in_img_fnames:
        img_name = img_fname.split('.png')[0]
        annotation_fname = img_name + '.txt'
        if (annotation_fnames.count(annotation_fname) != 0):
            boxes = extract_pedestrian_boxes(join(annotation_dir, annotation_fname))
            if (len(boxes) != 0):
                img = cv2.imread(join(in_img_dir, img_fname))
                for i in range(0, len(boxes)):
                    box = boxes[i]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    if (min(width, height) / max(width, height) < 0.15):
                        continue

                    desired_ratio = float(desired_size[0]) / float(desired_size[1])
                    ratio = width / height
                    if (ratio < desired_ratio):
                        # Expand by width
                        correct_width = height * desired_ratio
                        correct_height = height
                    else:
                        # Expand by height
                        correct_width = width
                        correct_height = width / desired_ratio

                    out_img = extract_expanded_pedestrian(img, box, int(correct_width), int(correct_height))
                    resized_img = cv2.resize(out_img, desired_size, interpolation = cv2.INTER_CUBIC)
                    cv2.imwrite(join(out_img_dir, img_name + '_' + str(i) + '.png'), resized_img)
        else:
            print 'Annotaion for ' + img_fname + ' not found'

def extract_expanded_pedestrian(img, box, correct_width, correct_height):
    img_width = img.shape[1]
    img_height = img.shape[0]

    out_img_size = (correct_height, correct_width, img.shape[2])
    out_img = np.zeros(out_img_size, img.dtype)

    x = int(box[0])
    y = int(box[1])
    box_w = int(box[2] - box[0])
    box_h = int(box[3] - box[1])
    a_delta_w = (correct_width - box_w) / 2
    a_delta_h = (correct_height - box_h) / 2

    r_delta_w = min(x + box_w + a_delta_w, img_width) - x - box_w
    l_delta_w = x - max(x - a_delta_w, 0)
    t_delta_h = y - max(y - a_delta_h, 0)
    b_delta_h = min(y + box_h + a_delta_h, img_height) - y - box_h

    out_img[a_delta_h - t_delta_h : t_delta_h + box_h + b_delta_h + (a_delta_h - t_delta_h),\
        a_delta_w - l_delta_w : l_delta_w + box_w + r_delta_w + (a_delta_w - l_delta_w)] = \
        img[y - t_delta_h : y + box_h + b_delta_h, x - l_delta_w : x + box_w + r_delta_w]

    return out_img

def extract_pedestrian_boxes(annotation_fname):
    f = open(annotation_fname, 'r')
    boxes = []
    for line in f:
        recomp = re.compile(r'Pedestrian( -?\d+\.?\d*){7}')
        s = recomp.match(line)
        if (not s is None):
            ss = s.group().split(' ')
            box = ()
            for i in range(len(ss) - 4, len(ss)):
                box = box + (float(ss[i]),)
            boxes.append(box)

    f.close()
    return boxes

if __name__ == '__main__':
    help_message = '\
Incorrect input parameters. Parameters must be as follow:\n\
first - input images dir\n\
second - annotation files dir\n\
third - output images dir (where you want to store pedestrian images)\
forth - desired size in format wxh. For example: 36x108'
    in_img_dir = ''
    annotation_dir = ''
    out_img_dir = ''
    size_str = ''
    
    if (len(sys.argv) >= 5):
        in_img_dir = sys.argv[1]
        annotation_dir = sys.argv[2]
        out_img_dir = sys.argv[3]
        size_str = sys.argv[4]
    else:
        print help_message
        exit()

    if (not isdir(in_img_dir) or not isdir(annotation_dir) or
        not isdir(out_img_dir)):
        print help_message
        exit()

    size_split = size_str.split('x')
    if (len(size_split) != 2):
        print help_message
        exit()

    desired_size = (int(size_split[0]), int(size_split[1]))

    in_img_fnames = [f for f in listdir(in_img_dir) if isfile(join(in_img_dir, f))]
    annotation_fnames = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

    extract_pedestrians(in_img_fnames, in_img_dir, annotation_fnames, annotation_dir, out_img_dir, desired_size)
