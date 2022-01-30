import time
import cv2 as cv
import numpy as np
import glob


def find_traffic_light(image):
    threshold = 0.75
    method = cv.TM_CCORR_NORMED
    image_color = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.imread('./assets/template-matching/template/model.jpg', 0)
    template = cv.resize(template, (50, 81))
    w, h = template.shape[::-1]

    max_values = []
    min_val, max_val, min_loc, max_loc = 0, 0, 0, 0

    for scale in np.linspace(0.5, 1.5, 20):
        dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        resized = cv.resize(image, dim)

        res = cv.matchTemplate(resized, template, method)
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            max_values.append([max_val, max_loc, scale])

    if len(max_values) > 0:
        max_val, max_loc, scale = max(max_values, key=lambda item: item[0])
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        image_crop = image_color[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        cv.rectangle(image, top_left, bottom_right, 255, 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, f'{scale}', (5, 180), font, 1, (255, 255, 255), 1, cv.LINE_AA)

        return image_crop, top_left, bottom_right, image

    return False, False, False, False


if __name__ == "__main__":
    # image = cv.imread('./assets/template-matching/20211126_054423.jpg')
    # image = cv.resize(image, (400, 300))

    # inicio = time.time()

    # image_crop, top_left, bottom_right, image = find_traffic_light(image)

    # final = time.time()
    # print(f'Encontrar semáforo: {(final - inicio):.2f} segundos')

    # cv.imshow('image', image)
    # # cv.imshow('image_crop', image_crop)

    # cv.waitKey(10000)
    # cv.destroyAllWindows()
    images = [img for img in glob.glob("./assets/template-matching/*.jpg")]
    for img in images:
        image = cv.imread(img)
        image = cv.resize(image, (400, 300))
        image_crop, top_left, bottom_right, image = find_traffic_light(image)
        cv.imshow(f'{img}', image)

    cv.waitKey(0)
    cv.destroyAllWindows()
