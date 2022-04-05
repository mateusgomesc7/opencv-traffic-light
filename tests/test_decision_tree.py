import cv2 as cv
import numpy as np
import decision_tree as dt
import template_match as tm
import glob


def check_result(result):
    text = ''
    color = (0, 0, 0)
    if (result == np.array([[1., 0., 0.]])).all():
        text = 'Vermelho'
        color = (0, 0, 255)
    elif (result == np.array([[0., 1., 0.]])).all():
        text = 'Amarelo'
        color = (0, 255, 255)
    elif (result == np.array([[0., 0., 1.]])).all():
        text = 'Verde'
        color = (0, 255, 0)
    return text, color


_, _, clf = dt.train()


# Carregar imagens testes
images_red = [img for img in glob.glob("./tests/red/*.jpg")]
images_yellow = [img for img in glob.glob("./tests/yellow/*.jpg")]
images_green = [img for img in glob.glob("./tests/green/*.jpg")]
images = []
images = images_red + images_yellow + images_green

count_match = 0
count_correct = 0
for i in range(len(images)):
    image = cv.imread(images[i])
    image = cv.resize(image, (400, 300))
    image_crop, top_left, bottom_right, _ = tm.find_traffic_light(image)
    if type(image_crop) is np.ndarray:
        feature = dt.get_features(image_crop)
        result = dt.predict_frame(feature, clf)
        # print(result)
        # Escrever resultado na imagem
        text, color = check_result(result)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, text, (5, 30), font, 1, color, 1, cv.LINE_AA)
        # Colocar retângulo
        cv.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)
        
        cv.imshow(f'Image: {images[i]}', image)
        
        if (i < 10) and (result == np.array([[1., 0., 0.]])).all():
            count_correct += 1
        elif (i>= 10 and i < 20) and (result == np.array([[0., 1., 0.]])).all():
            count_correct += 1
        elif (i>= 20 and i < 30) and (result == np.array([[0., 0., 1.]])).all():
            count_correct += 1
    else:
       count_match += 1
    

print(f'Quantidade de semáforos encontrados: {len(images)-count_match}/{len(images)}')
print(f'Quantidade de acertos: {count_correct}/{len(images)}')
cv.waitKey(0)
cv.destroyAllWindows()
