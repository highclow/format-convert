import os
import cv2
import xml.dom.minidom
import numpy as np
import json
 
image_path="weapons/"
annotation_path="weapons/"
 
files_name = os.listdir(image_path)
for filename_ in files_name:
  if '.jpg' in filename_:
    filename, extension= os.path.splitext(filename_)
    img_path =image_path+filename+'.jpg'
    xml_path =annotation_path+filename+'.xml'
    print(img_path)
    img = cv2.imread(img_path)
    if img is None:
        pass
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    objects=dom.getElementsByTagName("object")
    for k, object in enumerate(objects):
        polygon = root.getElementsByTagName('polygon')[k]
        polygon = polygon.childNodes[0].data
        polygon = np.array(json.loads(polygon)).astype(int)
        polygon = polygon.reshape((-1,1,2))
        cv2.polylines(img,[polygon],True,(0,0,255), 2)
    flag=0
    flag=cv2.imwrite("visualization/{}.jpg".format(filename),img)
    if(flag):
        print(filename,"done")
print("all done ====================================")
