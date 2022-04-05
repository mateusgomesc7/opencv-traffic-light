import cv2 as cv

def find_traffic_light(image):
    top_left = None
    bottom_right = None
    image_crop = None
    cascade = cv.CascadeClassifier('./cascade.xml')
    # image = cv.resize(image, (400, 300))
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(image_gray, 1.3, 5)
    # width, height = template.shape[::-1] 
    for (x,y,w,h) in faces:
        top_left = (x, y)
        bottom_right = (x+w,y+h)
        image_crop = image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    return image_crop, top_left, bottom_right
        
largura_img = 160
altura_img = 120

if __name__ == "__main__":
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    
    while(True):
        _, frame = cap.read()
        frame = cv.resize(frame, (largura_img, altura_img))
        
        image_crop, top_left, bottom_right = find_traffic_light(frame)
        print('shape', frame.shape, top_left, bottom_right)
        if top_left:
            cv.rectangle(frame,top_left,bottom_right,(255,255,255),1)
        cv.imshow('frame',frame)
        
        key = cv.waitKey(1)
        
        if key == 27:
        	break
    
    cap.release()
    cv.destroyAllWindows()
