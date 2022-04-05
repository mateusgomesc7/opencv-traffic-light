import time
import cv2 as cv
import numpy as np
import glob

def find_traffic_light(image):
    threshold = 0.75
    method = cv.TM_CCORR_NORMED
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.imread('./assets/template-matching/template/model.jpg', 0)
    template = cv.resize(template, (50, 81))
    w, h = template.shape[::-1]

    max_values = []
    min_val, max_val, min_loc, max_loc = 0, 0, 0, 0

    for scale in np.linspace(0.5, 1.5, 20):
        dim = (int(image_gray.shape[1] * scale), int(image_gray.shape[0] * scale))
        resized = cv.resize(image_gray, dim)

        res = cv.matchTemplate(resized, template, method)
        loc = np.where(res >= threshold)

        if len(loc[0]) > 0:
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            max_values.append([max_val, max_loc, scale])

    if len(max_values) > 0:
        max_val, max_loc, scale = max(max_values, key=lambda item: item[0])
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        image_crop = image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
        if image_crop.shape[0] == 0 or image_crop.shape[1] == 0:
            return False, False, False

        return image_crop, top_left, bottom_right

    return False, False, False


def find_traffic_light_for_test(image):
    threshold = 0.75
    method = cv.TM_CCORR_NORMED
    image_color = image.copy()
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.imread('./assets/template-matching/template/model.jpg', 0)
    template = cv.resize(template, (50, 81))
    w, h = template.shape[::-1]

    max_values = []
    min_val, max_val, min_loc, max_loc = 0, 0, 0, 0

    for scale in np.linspace(0.5, 1.5, 20):
        dim = (int(image_gray.shape[1] * scale), int(image_gray.shape[0] * scale))
        resized = cv.resize(image_gray, dim)

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
        if image_crop.shape[0] == 0 or image_crop.shape[1] == 0:
            return False, False, False, False
        cv.rectangle(image_color, top_left, bottom_right, (255, 255, 255), 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image_color, f'Escala template: {(scale*100):.2f}%', (5, 30), font, 0.75, (255, 255, 255), 2, cv.LINE_AA)

        return image_crop, top_left, bottom_right, image_color

    return False, False, False, False


if __name__ == "__main__":
    # image = cv.imread('./assets/template-matching/20211126_054423.jpg')
    # image = cv.resize(image, (400, 300))

    # inicio = time.time()

    # image_crop, top_left, bottom_right, image = find_traffic_light_for_test(image)

    # final = time.time()
    # print(f'Encontrar semáforo: {(final - inicio):.2f} segundos')

    # cv.imshow('image', image)
    # # cv.imshow('image_crop', image_crop)

    # cv.waitKey(10000)
    # cv.destroyAllWindows()

    images_red = [img for img in glob.glob("./tests/red/*.jpg")]
    images_yellow = [img for img in glob.glob("./tests/yellow/*.jpg")]
    images_green = [img for img in glob.glob("./tests/green/*.jpg")]
    images = []
    images = images_red + images_yellow + images_green
    count_match = 0
    for img in images:
        image = cv.imread(img)
        image = cv.resize(image, (400, 300))
        image_crop, top_left, bottom_right, image_gray = find_traffic_light_for_test(image)
        if image_crop is False:
           count_match += 1
        else:
            cv.imshow(f'{img}', image_gray)

    print(f'Quantidade de semáforos encontrados: {len(images)-count_match}/{len(images)}')
    cv.waitKey(0)
    cv.destroyAllWindows()
