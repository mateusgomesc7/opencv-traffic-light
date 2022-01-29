import glob
import time
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def get_mask_and_pos(img_rgb, img_hsv):
    lower_s = 120
    upper_s = 255

    lower_v = 200
    upper_v = 255

    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    pos = 0
    mask_s = cv.inRange(s, lower_s, upper_s)
    if np.sum(mask_s) > 0:
        pos = np.argmax(np.sum(s, axis=1))
        img_rgb[mask_s == 0] = [0, 0, 0]
        img_hsv[mask_s == 0] = [0, 0, 0]
    else:
        mask_v = cv.inRange(v, lower_v, upper_v)
        pos = np.argmax(np.sum(v, axis=1))
        img_rgb[mask_v == 0] = [0, 0, 0]
        img_hsv[mask_v == 0] = [0, 0, 0]

    return img_rgb, img_hsv, pos


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_features(images):
    dataset_pos = []
    dataset_colors_rgb = []
    dataset_colors_hsv = []
    dataset = []

    # Se for um frame, criar uma lista
    if not type(images) is list:
        images = [images]

    for img in images:
        # Resize
        if type(img) == np.ndarray:
            img = cv.resize(img, (32, 32))
        else:
            img = cv.resize(cv.imread(img), (32, 32))
        # Cortando imagem
        img = img[10:100, 10:30, :]
        # Convertendo em RGB
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Convertendo em HSV
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # Pegando imagem com máscara e posição mais brilhosa
        img_rgb_mask, img_hsv_mask, pos = get_mask_and_pos(img_rgb, img_hsv)
        # Média das cores RGB
        colors_rgb = [np.sum(img_rgb_mask[:, :, 0] / 255.0),
                      np.sum(img_rgb_mask[:, :, 1] / 255.0),
                      np.sum(img_rgb_mask[:, :, 2] / 255.0)]
        # Média das cores HSV
        colors_hsv = [np.sum(img_hsv_mask[:, :, 0] / 255.0),
                      np.sum(img_hsv_mask[:, :, 1] / 255.0),
                      np.sum(img_hsv_mask[:, :, 2] / 255.0)]

        dataset_pos.append(pos)
        dataset_colors_rgb.append(colors_rgb)
        dataset_colors_hsv.append(colors_hsv)

    # Normalizar dados
#     dataset_pos = normalize_data(dataset_pos)
#     dataset_colors_rgb = normalize_data(dataset_colors_rgb)
#     dataset_colors_hsv = normalize_data(dataset_colors_hsv)

    # Juntar os datasets
    for i in range(len(images)):
        dataset.append(
            ([dataset_colors_rgb[i][0], dataset_colors_rgb[i][1], dataset_colors_rgb[i][2],
              dataset_colors_hsv[i][0], dataset_colors_hsv[i][1], dataset_colors_hsv[i][2],
              dataset_pos[i]])
        )

    return dataset


def get_labels(images, position_class):
    # Labels com One-Hot Encoding
    n_categories = 3
    labels = np.zeros((len(images), n_categories))
    for ii in range(len(images)):
        jj = position_class
        labels[ii, jj] = 1
    return labels


def train():
    inicio = time.time()
    # Pegando as imagens
    red_lights = [img for img in glob.glob("./assets/5_tensorflow_traffic_light_images/red/*.jpg")]
    yellow_lights = [img for img in glob.glob("./assets/5_tensorflow_traffic_light_images/yellow/*.jpg")]
    green_lights = [img for img in glob.glob("./assets/5_tensorflow_traffic_light_images/green/*.jpg")]

    features_red = get_features(red_lights)
    features_yellow = get_features(yellow_lights)
    features_green = get_features(green_lights)

    labels_red = get_labels(red_lights, position_class=0)
    labels_yellow = get_labels(yellow_lights, position_class=1)
    labels_green = get_labels(green_lights, position_class=2)

    final = time.time()
    print(f'Preparo do dataset: {(final - inicio):.2f} segundos')

    inicio = time.time()

    X = features_red + features_yellow + features_green
    y = np.concatenate((labels_red, labels_yellow, labels_green))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)

    final = time.time()
    print(f'Treino da Árvore de Decisão: {(final - inicio):.2f} segundos')

    return X_test, y_test, clf


def predict(X_test, y_test, clf):
    resultado = clf.predict(X_test)

    print(metrics.classification_report(y_test, resultado))

    print(confusion_matrix(y_test.argmax(axis=1), resultado.argmax(axis=1)))


def predict_frame(frame, clf):
    result = clf.predict(frame)
    return result


if __name__ == "__main__":
    X_test, y_test, clf = train()

    # img = cv.imread('./5_tensorflow_traffic_light_images/green/0e470b16-71f2-471c-8b31-a21f5ab4d814.jpg')

    # red_lights = [img for img in glob.glob("./5_tensorflow_traffic_light_images/yellow/*.jpg")]
    # feature = get_features(red_lights[:10])
    # predict_frame(feature, clf)
    predict(X_test, y_test, clf)

    # cv.waitKey(0)
    # cv.destroyAllWindows()
