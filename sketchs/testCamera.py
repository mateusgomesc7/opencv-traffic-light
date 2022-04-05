import cv2 as cv
import utils
import arvore_de_decisao as ad
import traffic_light as tl
import time
import numpy as np


def check_result(result):
    text = ''
    color = (0,0,0)
    if (result == np.array([[1., 0., 0.]])).all():
        text = 'Vermelho'
        color = (0,0,255)
    elif (result == np.array([[0., 1., 0.]])).all():
        text = 'Amarelo'
        color = (0,255,255)
    elif (result == np.array([[0., 0., 1.]])).all():
        text = 'Verde'
        color = (0,255,0)
    return text, color


_, _, clf = ad.train()

# define a video capture object
vid = cv.VideoCapture(0)

while(True):
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


# VERIFICANDO A QUANTIDADE DE FRAMES:

# video = cv.VideoCapture(0);
# 
# # Number of frames to capture
# num_frames = 120;
# 
# print("Capturing {0} frames".format(num_frames))
# 
# # Start time
# start = time.time()
# 
# # Grab a few frames
# for i in range(0, num_frames) :
#     ret, frame = video.read()
#     feature = ad.get_features(frame)
#     ad.predict_frame(feature, clf)
#     cv.imshow('frame', frame)
# #     frame = cv.resize(frame, (266, 200))
# #     frame = tl.find_traffic_light(frame)
# 
# # End time
# end = time.time()
# 
# # Time elapsed
# seconds = end - start
# print ("Time taken : {0} seconds".format(seconds))
# 
# # Calculate frames per second
# fps  = num_frames / seconds
# print("Estimated frames per second : {0}".format(fps))
# 
# # Release video
# video.release()
# cv.destroyAllWindows()
