# Don't forget install Pillow package!

import sys
from os import listdir
from os import makedirs
from os.path import isfile, join
from os.path import exists
from PIL import Image

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

rgb2yuv = (
    0.299, 0.587, 0.114, 0,
    -0.14713, -0.28886, 0.436, 0,
    0.615, -0.51499, -0.10001, 0)

if not exists(out_dir):
    makedirs(out_dir)

for fname in files_to_cvt:
    rgb_img = Image.open(join(in_dir, fname))
    yuv_img = rgb_img.convert('RGB', rgb2yuv)
    yuv_img.save(join(out_dir, fname))
