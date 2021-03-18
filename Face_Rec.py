import os
import cv2

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
trained_face_data=cv2.CascadeClassifier(haar_model)

webcam=cv2.VideoCapture(0)
key=0
while (key!=81 and key!=113):
    successful_frame_read, frame = webcam.read()
    greyscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects = trained_face_data.detectMultiScale(greyscaled_img, 
                                 scaleFactor=1.3, 
                                 minNeighbors=4, 
                                 minSize=(30, 30)) 
    for (x,y,w,h) in rects:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Face Detector", frame)
        key=cv2.waitKey(1)
    cv2.imshow("Face Detector", frame)
    key=cv2.waitKey(1)
    if (key==81 or key==113): 
        break
webcam.release()
print("Program Completed")