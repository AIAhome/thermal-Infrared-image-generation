import os
import cv2 
for filename in os.listdir(r"/data/shenliao/FLIR_ADAS_v2/images_rgb_val/data/"):
    image=cv2.imread(os.path.join("/data/shenliao/FLIR_ADAS_v2/images_rgb_val/data/",filename))
    res=cv2.resize(image,(256,256))
    cv2.imwrite(os.path.join("/data/shenliao/FLIR_ADAS_v2/test/A/",filename), res)
