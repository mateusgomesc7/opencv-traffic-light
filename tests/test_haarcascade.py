import cv2 as cv
import glob

cascade = cv.CascadeClassifier('./cascade.xml')

if __name__ == "__main__":
    images_red = [img for img in glob.glob("./tests/red/*.jpg")]
    images_yellow = [img for img in glob.glob("./tests/yellow/*.jpg")]
    images_green = [img for img in glob.glob("./tests/green/*.jpg")]
    images = images_red + images_yellow + images_green
    
    count_match = 0
    for img in images:
        image = cv.imread(img)
        image = cv.resize(image, (400, 300))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        print(len(faces))
        for (x,y,w,h) in faces:
            cv.rectangle(image,(x,y),(x+w,y+h),(255, 255, 255), 2)

        if len(faces) > 0:
           count_match += 1
        cv.imshow(f'{img}', image)

    print(f'Quantidade de semáforos encontrados: {count_match}/{len(images)}')
    cv.waitKey(0)
    cv.destroyAllWindows()