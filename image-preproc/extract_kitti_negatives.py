import sys
from os.path import join, isfile, isdir
from os import listdir
import cv2
import random as rnd

def extract_negatives(in_img_fnames, in_img_dir, out_img_dir, desired_size, count, annotation_dir, annotation_fnames):
    img_fnames = get_img_fnames_without_pedestrians(annotation_dir, annotation_fnames, in_img_fnames)
    batch_size = int(count / len(img_fnames)) + 1
    extracted = 0
    for img_fname in img_fnames:
        img_name = img_fname.split('.png')[0]             
        img = cv2.imread(join(in_img_dir, img_fname))
        boxes = generate_boxes(img.shape[0:2][::-1], desired_size, min(batch_size, count - extracted))
        for i in range(0, len(boxes)):
            box = boxes[i]
            out_img = img[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
            cv2.imwrite(join(out_img_dir, img_name + '_' + str(i) + '.png'), out_img)

        extracted = extracted + len(boxes)
        if (extracted >= count):
            break

def generate_boxes(img_size, desired_size, count):
    if (desired_size[0] == 0 or desired_size[1] == 0 or
        desired_size[0] > img_size[0] or desired_size[1] > img_size[1]):
        return []
    boxes = []
    for i in range(0, count):
        x = rnd.randint(0, img_size[0] - desired_size[0])
        y = rnd.randint(0, img_size[1] - desired_size[1])
        boxes.append((x, y, desired_size[0], desired_size[1]))
    return boxes

def get_img_fnames_without_pedestrians(annotaion_dir, annotation_fnames, in_img_fnames):
    good_img_fnames = []
    for in_img_fname in in_img_fnames:
        in_img_name = in_img_fname.split('.png')[0]
        annotation_fname = in_img_name + '.txt'
        if (annotation_fnames.count(annotation_fname) != 0):
            f = open(join(annotation_dir, annotation_fname), 'r')
            good = True
            for line in f:
                if (line.find('Pedestrian') != -1):
                    good = False
                    break
            f.close()
            if (good):
                good_img_fnames.append(in_img_fname)
    return good_img_fnames

if __name__ == '__main__':
    help_message = '\
Incorrect input parameters. Parameters must be as follow:\n\
first - input images dir\n\
second - annotation files dir\n\
third - output images dir (where you want to store negative images)\
forth - desired size in format wxh. For example: 36x108\n\
fifth - count of negatives to extract'

    in_img_dir = ''
    annotation_dir = ''
    out_img_dir = ''
    size_str = ''
    count_str = ''
    
    if (len(sys.argv) >= 6):
        in_img_dir = sys.argv[1]
        annotation_dir = sys.argv[2]
        out_img_dir = sys.argv[3]
        size_str = sys.argv[4]
        count_str = sys.argv[5]
    else:
        print help_message
        exit()

    if (not isdir(in_img_dir) or not isdir(annotation_dir) or not isdir(out_img_dir)):
        print help_message
        exit()

    size_split = size_str.split('x')
    if (len(size_split) != 2):
        print help_message
        exit()

    desired_size = (int(size_split[0]), int(size_split[1]))
    count = int(count_str)

    in_img_fnames = [f for f in listdir(in_img_dir) if isfile(join(in_img_dir, f))]
    annotation_fnames = [f for f in listdir(annotation_dir) if isfile(join(annotation_dir, f))]

    extract_negatives(in_img_fnames, in_img_dir, out_img_dir, desired_size, count, annotation_dir, annotation_fnames)
