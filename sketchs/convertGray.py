# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:28:01 2022

@author: Mateus
"""

import cv2
import glob


red_lights = [img for img in glob.glob("./assets/add_decision_tree/red/*.jpg")]
yellow_lights = [img for img in glob.glob("./assets/add_decision_tree/yellow/*.jpg")]
green_lights = [img for img in glob.glob("./assets/add_decision_tree/green/*.jpg")]

images = red_lights + yellow_lights + green_lights

if __name__ == '__main__': 
    i=0
    
    
    for image in images:
        img = cv2.imread(image, 0)
        
        # cv2.imshow("capture", img)
        
        cv2.imwrite('./imagesGray/'+str(i)+'.jpg',img)
        i+=1

        # k=cv2.waitKey(1)
        # if k==ord('s'):
        #     cv2.imwrite('./images/'+str(i)+'.jpg',frame)
        #     i+=1
        #     print('SALVOU')
        # elif k==ord("q"):
        #     break
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()