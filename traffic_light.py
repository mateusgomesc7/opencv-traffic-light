import glob
import cv2 as cv
import numpy as np
import time

def find_traffic_light(image):
    method = cv.TM_CCORR_NORMED
    image_color = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.imread('./template-matching/template/model.jpg', 0)
    template = cv.resize(template, (50, 81))
    w, h = template.shape[::-1]
    
    res = cv.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    image_crop = image_color[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
#     cv.rectangle(image, top_left, bottom_right, 255, 2)
    
    return image_crop, top_left, bottom_right
    
if (__name__ == "__main__"):    
    image = cv.imread('./template-matching/20211126_054358.jpg')
    image = cv.resize(image, (400, 300))
    
    inicio = time.time()
    
    image, image_crop = find_traffic_light(image)
    
    final = time.time()
    print(f'Encontrar semáforo: {(final - inicio):.2f} segundos')
    
    cv.imshow('image', image)
    cv.imshow('image_crop', image_crop)
    
    cv.waitKey(0)    
    cv.destroyAllWindows()

# images = [img for img in glob.glob("./template-matching/*.jpg")]

# method = cv.TM_CCOEFF



# for i in range(len(images)):
#     images[i] = cv.resize(cv.imread(images[i], 0), (400, 300))
# 
#     
# 
#     cv.imshow(f'image {i}', images[i])
#     cv.imshow('template', template)

# for i in range(len(images)):
#     # resize e gray
#     img = cv.resize(cv.imread(images[i], 0), (300, 300))
#     
#     
#     cv.imshow(f'image {i}', img)

    