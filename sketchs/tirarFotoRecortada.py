import cv2
# import os

Datos = 'images'
# if not os.path.exists(Datos):
#     print('Carpeta creada: ',Datos)
#     os.makedirs(Datos)
    
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
largura_img = 160
altura_img = 120
x1, y1 = 65, 40
x2, y2 = 95, 80
count = 0

_, frame = cap.read()
print(frame.shape)

scale = 4
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", largura_img*scale, altura_img*scale)

while True:
    _, frame = cap.read()
    
    frame = cv2.resize(frame, (largura_img, altura_img))
    
    imAux = frame.copy()
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
    objeto = imAux[y1:y2,x1:x2]

    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite(Datos+'/objeto_{}.jpg'.format(count),objeto)
        print('Imagen guardada:'+'/objeto_{}.jpg'.format(count))
        count = count +1
    if k == 27:
        break
    cv2.imshow('frame',frame)
  
cap.release()
cv2.destroyAllWindows()
