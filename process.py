import os
import cv2 
for filename in os.listdir(input_path):
    image=cv2.imread(os.path.join(inputpath,filename))
    res=cv2.resize(image,(256,256))
    cv2.imwrite(os.path.join(output_path,filename), res)
