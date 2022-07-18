import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import template_match as tm
import decision_tree as dt
import haarcascade as hc
import pickle

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

number_of_check_signal = 3
number_of_moves = 1

labels = ['LEFT MIN', 'LEFT MAX', 'FORWARD', 'RIGHT MIN', 'RIGHT MAX']
# ================================


# ============SETUP=============
clf = pickle.load(open('./models/decision_tree_model.sav', 'rb'))

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
    
    # Verifica se o semáforo foi identificado
    if type(frame_crop) is np.ndarray:
        feature = dt.get_features(frame_crop)
        result = dt.predict_frame(feature, clf)

        text = ''
        if (result == np.array([[1., 0., 0.]])).all():
            text = 'red'
        elif (result == np.array([[0., 1., 0.]])).all():
            text = 'yellow'
        elif (result == np.array([[0., 0., 1.]])).all():
            text = 'green'
        return text, top_left, bottom_right
    else:
        return '', '', ''


def show_traffic_light(frame, signal, top_left, bottom_right):
    frame_traffic = frame.copy()
    # Verifica se algum semáforo foi encontrado
    if top_left:
        color = (0, 0, 0)
        if signal == 'red':
            color = (0, 0, 255)
        elif signal == 'yellow':
            color = (0, 255, 255)
        elif signal == 'green':
            color = (0, 255, 0)
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

def choose_movement(output):
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
    

# ==================================


if __name__ == "__main__":
    # Inicializa as telas para aplicação da estrada e do semáforo  
    scale = 4
    cv.namedWindow("Traffic Light", cv.WINDOW_NORMAL)
    cv.resizeWindow("Traffic Light", largura_img*scale, altura_img*scale)
    
    while True:
        print('====================')
        if autonomous_mode == True:
            (ret, frame) = cam.read()
            frame = cv.resize(frame, (largura_img, altura_img))
            
            signal = ''
            for check_signal in range(number_of_check_signal):
                # Aplicação do semáforo
                signal, top_left, bottom_right = get_traffic_light(frame)
                print('Signal ', signal if signal else 'NÃO')
                show_traffic_light(frame, signal, top_left, bottom_right)
                if signal == 'red' or signal == 'yellow':
                    break
            
            # Ganha uma quantidade de movimento se não tiver sinal ou se for verde
            if signal == 'green' or signal == '':
                # Aplicação da estrada
                for x in range(number_of_moves):
                    output = get_road(frame)
                    choose_movement(output)
                    time.sleep(1)
            
        key = cv.waitKey(1)
        if key == 27:
            break
    
    cam.release()
    cv.destroyAllWindows()
    
