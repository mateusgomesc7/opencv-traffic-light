import cv2 as cv
import numpy as np
import arvore_de_decisao as ad
import traffic_light as tl


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


_, _, clf = ad.train()

# define a video capture object
vid = cv.VideoCapture(0)

while True:
    ret, frame = vid.read()

    frame = cv.resize(frame, (266, 200))
    frame_crop, top_left, bottom_right = tl.find_traffic_light(frame)
    feature = ad.get_features(frame_crop)
    result = ad.predict_frame(feature, clf)

    # Escrever resultado na imagem
    text, color = check_result(result)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, text, (5, 180), font, 1, color, 1, cv.LINE_AA)

    cv.rectangle(frame, top_left, bottom_right, 255, 2)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()
