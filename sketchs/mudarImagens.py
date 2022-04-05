import glob
import cv2 as cv
from random import randrange

largura_img = 30
altura_img = 40

images_1 = [img for img in glob.glob("./novo_green/*.jpg")]
images_2 = [img for img in glob.glob("./n_green/*.jpg")]

images = images_1 + images_2

i=1

for img in images:
    img = cv.imread(img)
    cv.imwrite('./green/'+str(i)+'.jpg', img)
    i+=1

# # Rotação de 0 a 45
# for img in images:
#     img = cv.imread(img)
#     ponto = (largura_img/2, altura_img/2)
#     rotacao = cv.getRotationMatrix2D(ponto, randrange(0, 45), 1.0)   
#     rotacionado = cv.warpAffine(img, rotacao, (largura_img, altura_img))
#     cv.imwrite('./novos_carros/'+str(i)+'.jpg', rotacionado)
#     i+=1
    
# # Rotação de -45 a 0
# for img in images:
#     img = cv.imread(img)
#     ponto = (largura_img/2, altura_img/2)
#     rotacao = cv.getRotationMatrix2D(ponto, randrange(-45, 0), 1.0)   
#     rotacionado = cv.warpAffine(img, rotacao, (largura_img, altura_img))
#     cv.imwrite('./novos_carros/'+str(i)+'.jpg', rotacionado)
#     i+=1


# # Espelhado horizontalmente
# for img in images:
#     inverter = cv.flip(cv.imread(img), 1)
#     cv.imwrite('./novos_carros/'+str(i)+'.jpg',inverter)
#     i+=1
    

# for img in images:
#     img = cv.resize(cv.imread(img), (largura_img, altura_img))
#     cv.imwrite('./novo_yellow/'+str(i)+'.jpg',img)
#     i+=1
    