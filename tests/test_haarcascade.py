import cv2 as cv
import glob

cascade = cv.CascadeClassifier('./cascades/cascade7.xml')

largura_img = 160
altura_img = 120

if __name__ == "__main__":
    images_red = [img for img in glob.glob("./red/*.jpg")]
    images_yellow = [img for img in glob.glob("./yellow/*.jpg")]
    images_green = [img for img in glob.glob("./green/*.jpg")]
    images = images_red + images_yellow + images_green \
            + images_red + images_yellow + images_green \
            + images_red + images_yellow + images_green + images_green
    # images = [img for img in glob.glob("./images_test_haarcascade/*.jpg")]
    # images = [img for img in glob.glob("../sketchs/images/*.jpg")]
    # images = [img for img in glob.glob("../sketchs/images_novos/*.jpg")]
    
    i = 0
    count_match = 0
    for img in images:
        image = cv.imread(img)
        image = cv.resize(image, (largura_img, altura_img))
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.blur(image, (1, 1))
        
        faces = cascade.detectMultiScale(image, 1.3, 5)
        print(len(faces))
        for (x,y,w,h) in faces:
            cv.rectangle(image,(x,y),(x+w,y+h),(255, 255, 255), 2)
                
        if len(faces) > 0:
           count_match += 1
        cv.imshow(f'{i}', image)
        i+=1

    print(f'Quantidade de semáforos encontrados: {count_match}/{len(images)}')
    cv.waitKey(0)
    cv.destroyAllWindows()