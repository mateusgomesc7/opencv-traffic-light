import time
import cv2 as cv
import numpy as np


def find_traffic_light(image):
    method = cv.TM_CCORR_NORMED
    image_color = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.imread('./assets/template-matching/template/model.jpg', 0)
    template = cv.resize(template, (50, 81))
    w, h = template.shape[::-1]

    res = cv.matchTemplate(image, template, method)
    
    threshold = 0.75
    loc = np.where(res >= threshold)
    
    # for pt in zip(*loc[::-1]):
        # cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)
    
    if len(loc[0]) > 0:
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # print(max_val)
    
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
    
        image_crop = image_color[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        # cv.rectangle(image, top_left, bottom_right, 255, 2)
        
        return image_crop, top_left, bottom_right

    return False, False, False
    

if __name__ == "__main__":
    image = cv.imread('./assets/template-matching/20211126_054358.jpg')
    image = cv.resize(image, (400, 300))

    inicio = time.time()

    image_crop, top_left, bottom_right = find_traffic_light(image)

    final = time.time()
    print(f'Encontrar semáforo: {(final - inicio):.2f} segundos')

    # cv.imshow('image', image)
    cv.imshow('image_crop', image_crop)

    cv.waitKey(10000)
    cv.destroyAllWindows()
