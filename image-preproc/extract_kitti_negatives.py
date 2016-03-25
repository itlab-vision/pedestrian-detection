import sys
from os.path import join, isfile, isdir
from os import listdir
import cv2
import random as rnd

def extract_negatives(in_img_fnames, in_img_dir, out_img_dir, desired_size, count):
    batch_size = int(count / len(in_img_fnames)) + 1
    extracted = 0
    for img_fname in in_img_fnames:
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


if __name__ == '__main__':
    help_message = '\
Incorrect input parameters. Parameters must be as follow:\n\
first - input images dir\n\
second - output images dir (where you want to store negative images)\
third - desired size in format wxh. For example: 36x108\n\
forth - count of negatives to extract'

    in_img_dir = ''
    out_img_dir = ''
    size_str = ''
    count_str = ''
    
    if (len(sys.argv) >= 5):
        in_img_dir = sys.argv[1]
        out_img_dir = sys.argv[2]
        size_str = sys.argv[3]
        count_str = sys.argv[4]
    else:
        print help_message
        exit()

    if (not isdir(in_img_dir) or not isdir(out_img_dir)):
        print help_message
        exit()

    size_split = size_str.split('x')
    if (len(size_split) != 2):
        print help_message
        exit()

    desired_size = (int(size_split[0]), int(size_split[1]))
    count = int(count_str)

    in_img_fnames = [f for f in listdir(in_img_dir) if isfile(join(in_img_dir, f))]

    extract_negatives(in_img_fnames, in_img_dir, out_img_dir, desired_size, count)
