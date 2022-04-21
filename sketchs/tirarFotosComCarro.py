import cv2 as cv
import BlynkLib
import RPi.GPIO as GPIO
import time

# ============GLOBALS=============
CONTROL_BLYNK = False
cam = cv.VideoCapture(0)
sleep = .3
largura_img = 160
altura_img = 120
autonomous_mode = False

min_right = 6.5
max_right = 5
frente = 7.5
min_left = 8.5
max_left = 10

labels = ['LEFT MIN', 'LEFT MAX', 'FORWARD', 'RIGHT MIN', 'RIGHT MAX']

# Pinos do GPIO Raspberry 3 model B
m11 = 3
m12 = 4
m21 = 27
m22 = 22
servo = 12
# =================================

# ============SETUP=============
GPIO.setmode(GPIO.BCM)  # RASPBERRY 3 MODEL B
GPIO.setwarnings(False)

# comunicação com o controle feito no app Blynk
token = 'ZZu6ouPCpQiOPlP44WzKQ7cZSd8oqyHh'

GPIO.setup(m11, GPIO.OUT)
GPIO.setup(m12, GPIO.OUT)
GPIO.setup(m21, GPIO.OUT)
GPIO.setup(m22, GPIO.OUT)
GPIO.setup(servo, GPIO.OUT)

# PWM com 50Hz => duty cycle 2.5-12.5% (0-180 graus)
servo1 = GPIO.PWM(servo, 50)
servo1.start(7.5)
time.sleep(1)
servo1.ChangeDutyCycle(0)
print("iniciando")

blynk = BlynkLib.Blynk(token)
# ===============================

# ============FUNCTIONS=============
def forward():
    GPIO.output(m11, 1)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 0)

def stop():
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)

def save_image(valor, frame):
    global i
    if valor[0] == "1":
        cv.imwrite('./images/'+str(i)+'.jpg',frame)
        i+=1
        print('SALVOU')

def up_side(valor):
    print('up_side1')
    if valor[0] == "1":
        forward()
    elif valor[0] == "0":
        stop()

def right_side_min(valor):
    print('right_side_min')
    if valor[0] == "1":
        servo1.ChangeDutyCycle(min_right)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)
    elif valor[0] == "0":
        servo1.ChangeDutyCycle(frente)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)

def right_side_max(valor):
    print('right_side_max')
    if valor[0] == "1":
        servo1.ChangeDutyCycle(max_right)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)
    elif valor[0] == "0":
        servo1.ChangeDutyCycle(frente)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)

def left_side_min(valor):
    print('left_side_min')
    if valor[0] == "1":
        servo1.ChangeDutyCycle(min_left)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)
    elif valor[0] == "0":
        servo1.ChangeDutyCycle(frente)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)

def left_side_max(valor):
    print('left_side_max')
    if valor[0] == "1":
        servo1.ChangeDutyCycle(max_left)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)
    elif valor[0] == "0":
        servo1.ChangeDutyCycle(frente)
        time.sleep(sleep/3)
        servo1.ChangeDutyCycle(0)

# ==================================


if __name__ == '__main__': 
    i=0
    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.resizeWindow("image", 400, 300)

    while(True):
        _, frame = cam.read()
        frame = cv.resize(frame, (largura_img, altura_img))
        cv.imshow("image", frame)

        if CONTROL_BLYNK:
            blynk.run()
    
            k=cv.waitKey(1)
            if k==ord('q'):
                break

            @blynk.VIRTUAL_WRITE(8)
            def blynk_save_image(valor):
                save_image(valor, frame)
            
            @blynk.VIRTUAL_WRITE(1)
            def blynk_up_side(valor):
                up_side(valor)
    
            @blynk.VIRTUAL_WRITE(5)
            def blynk_right_side_min(valor):
                right_side_min(valor)
    
            @blynk.VIRTUAL_WRITE(2)
            def blynk_right_side_max(valor):
                right_side_max(valor)
    
            @blynk.VIRTUAL_WRITE(6)
            def blynk_left_side_min(valor):
                left_side_min(valor)
    
            @blynk.VIRTUAL_WRITE(7)
            def blynk_left_side_max(valor):
                left_side_max(valor)

        else:
            k=cv.waitKey(1)
            if k==ord('q'):
                break
            
            # if k==ord('s'):
            #     save_image(valor)
            
            
            
        
    cam.release()
    cv.destroyAllWindows()