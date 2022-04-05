import cv2


if __name__ == '__main__': 
    i=0
    # cap=cv2.VideoCapture(0+cv2.CAP_DSHOW)
    cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while(1):
        ret ,frame = cap.read()

        cv2.imshow("capture", frame)

        k=cv2.waitKey(1)
        if k==ord('s'):
            cv2.imwrite('./images/'+str(i)+'.jpg',frame)
            i+=1
            print('SALVOU')
        elif k==ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()