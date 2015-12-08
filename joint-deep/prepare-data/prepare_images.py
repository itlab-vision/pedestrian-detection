import cv2
import numpy as np
import math
from os import listdir
from os.path import exists
from os.path import isfile, join


def biggest(a,b,c):
    Max = a
    if b > Max:
        Max = b
    if c > Max:
        Max = c
    return Max


in_dir = 'input'
out_dir = 'output'
files_to_cvt = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

if not exists(out_dir):
    makedirs(out_dir)
    

for fname in files_to_cvt:
    img_source = cv2.imread(join(in_dir, fname))
    img_source = img_source[12:(12+84),4:(4+28)]
    img_hsv = cv2.cvtColor(img_source, cv2.COLOR_BGR2HSV)
    
    output_1 = img_hsv[:,:,2];
    height, width = img_hsv.shape[:2]
    img_hsv = cv2.resize(img_hsv,(width/2, height/2),interpolation=cv2.INTER_CUBIC)
    
    #sobel_mat = np.array([[-1 , 0 , 1] , [-2 , 0 , 2] , [-1 , 0 , 1] ])
    #sobel = np.zeros(img_hsv.shape, dtype=np.float)
    #for i in range(0,3):
       #sobelx = cv2.filter2D(img_hsv[:,:,i] , cv2.CV_32F , sobel_mat)
        #sobely = cv2.filter2D(img_hsv[:,:,i] , cv2.CV_32F , sobel_mat.transpose())
        #sobel[:,:,i] = cv2.pow(cv2.pow(sobelx,2)+cv2.pow(sobely,2),0.5)
    sobelx = cv2.Sobel(img_hsv,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_ISOLATED)
    sobely = cv2.Sobel(img_hsv,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_ISOLATED)
    sobel = cv2.pow(cv2.pow(sobelx,2)+cv2.pow(sobely,2),0.5)
    
    rows,cols = sobel.shape[:2]
    sobel_max = np.maximum(sobel[:,:,0],sobel[:,:,1])
    sobel_max = np.maximum(sobel_max,sobel[:,:,2])
    output_2 = np.zeros(output_1.shape, dtype=np.double)
    output_2[0:rows,0:cols] = sobel[:,:,0]
    output_2[0:rows,cols:] = sobel[:,:,1]
    output_2[rows:,0:cols] = sobel[:,:,2]
    output_2[rows:,cols:] = sobel_max
    output_3 = np.zeros(output_1.shape, dtype=np.double)
    output_3[0:rows,0:cols] = img_hsv[:,:,0]
    output_3[0:rows,cols:] = img_hsv[:,:,1]
    output_3[rows:,0:cols] = img_hsv[:,:,2]
    output_1 = (output_1 - np.mean(output_1)) / (math.sqrt(np.var(output_1)) + 0.00001)
    output_2 = (output_2 - np.mean(output_2)) / (math.sqrt(np.var(output_2)) + 0.00001)
    output_3 = (output_3 - np.mean(output_3)) / (math.sqrt(np.var(output_3)) + 0.00001)
    output = cv2.merge((output_1,output_3, output_2))
    #output = cv2.normalize(output,cv2.NORM_L2)
    #print output[(output.shape[0] - 5):,(output.shape[1] - 5):,2]
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',output[:,:,1]);
    #print cv2.minMaxLoc(output[:,:,2]);
    f = open ('image.txt','w')
    np.savetxt('image.txt',output[:,:,1],delimiter=' ')
    #print output.shape
    #cv2.imwrite('12345.png',output[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()










