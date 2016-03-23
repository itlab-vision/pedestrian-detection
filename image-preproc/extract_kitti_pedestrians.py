import sys
from os.path import isdir, isfile, join
from os import listdir, walk
import cv2
import re

def extract_pedestrians(in_img_fnames, in_img_dir, annotation_fnames, annotation_dir, out_img_dir):
    for img_fname in in_img_fnames:
        img_name = img_fname.split('.png')[0]
        annotation_fname = img_name + '.txt'
        if (annotation_fnames.count(annotation_fname) != 0):
            boxes = extract_pedestrian_boxes(join(annotation_dir, annotation_fname))
            if (len(boxes) != 0):
                img = cv2.imread(join(in_img_dir, img_fname))
                for i in range(0, len(boxes)):
                    box = boxes[i]
                    out_img = img[box[1]:box[3], box[0]:box[2]]
                    cv2.imwrite(join(out_img_dir, img_name + str(i) + '.png'), out_img)
        else:
            print 'Annotaion for ' + img_fname + ' not found'

def extract_pedestrian_boxes(annotation_fname):
    f = open(annotation_fname, 'r')
    boxes = []
    for line in f:
        recomp = re.compile(r'Pedestrian( \d+\.?\d*){7}')
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
third - output images dir (where you want to store pedestrian images)'
    in_img_dir = ''
    annotation_dir = ''
    out_img_dir = ''
    
    if (len(sys.argv) >= 4):
        in_img_dir = sys.argv[1]
        annotation_dir = sys.argv[2]
        out_img_dir = sys.argv[3]
    else:
        print help_message
        exit()

    if (not isdir(in_img_dir) or not isdir(annotation_dir) or
        not isdir(out_img_dir)):
        print help_message
        exit()

    in_img_fnames = [f for f in listdir(in_img_dir) if isfile(join(in_img_dir, f))]
    annotation_fnames = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

    extract_pedestrians(in_img_fnames, in_img_dir, annotation_fnames, annotation_dir, out_img_dir)
