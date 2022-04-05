import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import template_match as tm
import decision_tree as dt
import haarcascade as hc

# ============GLOBALS=============
cam = cv.VideoCapture(0, cv.CAP_DSHOW)
sleep = .3
largura_img = 160
altura_img = 120
autonomous_mode = True

min_right = 6.5
max_right = 5
frente = 7.5
min_left = 8.5
max_left = 10

labels = ['LEFT MIN', 'LEFT MAX', 'FORWARD', 'RIGHT MIN', 'RIGHT MAX']
# ================================


# ============SETUP=============
_, _, clf = dt.train()

interpreter = tflite.Interpreter('models/mlp_model.tflite')
interpreter.allocate_tensors()

time.sleep(1)
print("iniciando")
# ===============================


# ============FUNCTIONS=============
def stop():
    print("stop")

def forward(t):
    print("forward", t)

def moveCar(direction):
    print("moveCar")
    time.sleep(sleep / 3)
    forward(sleep)

def get_traffic_light(frame):
    frame_crop, top_left, bottom_right = hc.find_traffic_light(frame)
    # frame_crop, top_left, bottom_right = tm.find_traffic_light(frame)
    
    # Verifica se o semáforo foi identificado
    if type(frame_crop) is np.ndarray:
        feature = dt.get_features(frame_crop)
        result = dt.predict_frame(feature, clf)

        text = ''
        color = (0, 0, 0)
        if (result == np.array([[1., 0., 0.]])).all():
            text = 'red'
            color = (0, 0, 255)
        elif (result == np.array([[0., 1., 0.]])).all():
            text = 'yellow'
            color = (0, 255, 255)
        elif (result == np.array([[0., 0., 1.]])).all():
            text = 'green'
            color = (0, 255, 0)
        return text, color, top_left, bottom_right
    else:
        return '', '', '', ''


def show_traffic_light(frame, signal, color, top_left, bottom_right):
    frame_traffic = frame.copy()
    # Verifica se algum semáforo foi encontrado
    if top_left:
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame_traffic, signal, (10, 10), font, 0.25, color, 1, cv.LINE_AA)
        cv.rectangle(frame_traffic, top_left, bottom_right, (255,255,255),1)
    cv.imshow('Traffic Light', frame_traffic)


def get_road(frame):
    # Transforma em escala de cinza
    test_img = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Mudando escala dos valores dos pixels de 0 à 1
    test_img = np.array(test_img, dtype=np.float32) / 255.0
    # Adicionar dimensões no array da imagem. Ex: (1, imagem, 1)
    test_img = test_img[np.newaxis, ..., np.newaxis]

    interpreter.set_tensor(0, test_img)
    interpreter.invoke()
    output = interpreter.get_tensor(10)
    print(labels[output.argmax()])
    return output

# ==================================


if __name__ == "__main__":
    # Inicializa as telas para aplicação da estrada e do semáforo  
    scale = 4
    cv.namedWindow("Road", cv.WINDOW_NORMAL)
    cv.resizeWindow("Road", largura_img*scale, altura_img*scale)
    cv.namedWindow("Traffic Light", cv.WINDOW_NORMAL)
    cv.resizeWindow("Traffic Light", largura_img*scale, altura_img*scale)
    
    while True:
        print('====================')
        if autonomous_mode == True:
            (ret, frame) = cam.read()
            frame = cv.resize(frame, (largura_img, altura_img))
            
            # Aplicação do semáforo
            signal, color, top_left, bottom_right = get_traffic_light(frame)
            print('Signal ', signal)
            show_traffic_light(frame, signal, color, top_left, bottom_right)
            
            # Aplicação da estrada
            cv.imshow('Road', frame)
            output = get_road(frame)
            
            
            if signal == 'red' or signal == 'yellow':
                stop()
                # time.sleep(1)
            else:
                if labels[output.argmax()] == labels[0]:
                    moveCar(min_left)
                    print("0")
                elif labels[output.argmax()] == labels[1]:
                    moveCar(max_left)
                    print("1")
                elif labels[output.argmax()] == labels[2]:
                    moveCar(frente)
                    print("2")
                elif labels[output.argmax()] == labels[3]:
                    moveCar(min_right)
                    print("3")
                elif labels[output.argmax()] == labels[4]:
                    moveCar(max_right)
                    print("4")
                else:
                    print("...")
                # time.sleep(1)
            
        key = cv.waitKey(1)
        if key == 27:
            break
    
    cam.release()
    cv.destroyAllWindows()
    
