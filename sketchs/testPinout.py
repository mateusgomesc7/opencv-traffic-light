import RPi.GPIO as GPIO
import time
import BlynkLib

# Pinos do GPIO Raspberry 3 model B
m11 = 3
m12 = 4
m21 = 27
m22 = 22
servo = 12

# ============GLOBALS=============
sleep = .3

min_right = 6.5
max_right = 5
frente = 7.5
min_left = 8.5
max_left = 10


GPIO.setmode(GPIO.BCM)  # RASPBERRY 3 MODEL B
GPIO.setwarnings(False)

# comunicação com o controle feito no app Blynk
token = 'ZZu6ouPCpQiOPlP44WzKQ7cZSd8oqyHh'

GPIO.setup(m11, GPIO.OUT)
GPIO.setup(m12, GPIO.OUT)
GPIO.setup(m21, GPIO.OUT)
GPIO.setup(m22, GPIO.OUT)
GPIO.setup(servo, GPIO.OUT)

blynk = BlynkLib.Blynk(token)

def forward():
    GPIO.output(m11, 1)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 0)
    
def backward():
    GPIO.output(m11, 0)
    GPIO.output(m12, 1)
    GPIO.output(m21, 0)
    GPIO.output(m22, 1)

def stop():
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)




# PWM com 50Hz => duty cycle 2.5-12.5% (0-180 graus)
servo1 = GPIO.PWM(servo, 50)
servo1.start(7.5)
time.sleep(1)
servo1.ChangeDutyCycle(0)
print("iniciando")

while(True):
    blynk.run()
    
    @blynk.VIRTUAL_WRITE(0)
    def teste1(valor):
        print('forward', valor)
        if valor[0] == "1":
            forward()
        elif valor[0] == "0":
            stop()
    
    @blynk.VIRTUAL_WRITE(1)
    def teste1(valor):
        print('forward', valor)
        if valor[0] == "1":
            forward()
        elif valor[0] == "0":
            stop()
    
    @blynk.VIRTUAL_WRITE(4)
    def teste1(valor):
        print('backward', valor)
        if valor[0] == "1":
            backward()
        elif valor[0] == "0":
            stop()
    
    @blynk.VIRTUAL_WRITE(6)
    def teste1(valor):
        if valor[0] == "1":
            servo1.ChangeDutyCycle(min_left)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
        elif valor[0] == "0":
            servo1.ChangeDutyCycle(frente)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
            
    @blynk.VIRTUAL_WRITE(7)
    def teste1(valor):
        if valor[0] == "1":
            servo1.ChangeDutyCycle(max_left)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
        elif valor[0] == "0":
            servo1.ChangeDutyCycle(frente)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
    
    @blynk.VIRTUAL_WRITE(5)
    def teste1(valor):
        if valor[0] == "1":
            servo1.ChangeDutyCycle(min_right)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
        elif valor[0] == "0":
            servo1.ChangeDutyCycle(frente)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
    
    @blynk.VIRTUAL_WRITE(2)
    def teste1(valor):
        if valor[0] == "1":
            servo1.ChangeDutyCycle(max_right)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
        elif valor[0] == "0":
            servo1.ChangeDutyCycle(frente)
            time.sleep(sleep/3)
            servo1.ChangeDutyCycle(0)
