import sys
from os import listdir
from os import makedirs
from os.path import isfile, join
from os.path import exists
import cv2

usage = "<-in-dir> <dir name> <-out-dir> <dir name>\n\
in-dir - directory that contains RGB images\n\
out-dir - directory where YUV images will be stored to"

files_to_cvt = []
if (len(sys.argv) >= 5):
    in_dir = ''
    out_dir = ''
    for i in range(0, len(sys.argv) - 1):
        if (sys.argv[i] == '-in-dir'):
            in_dir = sys.argv[i + 1]
        elif (sys.argv[i] == '-out-dir'):
            out_dir = sys.argv[i + 1]
    files_to_cvt = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
else:
    print usage
    sys.exit()

if not exists(out_dir):
    makedirs(out_dir)

for fname in files_to_cvt:
    bgr_img = cv2.imread(join(in_dir, fname))
    yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
    cv2.imwrite(join(out_dir, fname), yuv_img)
