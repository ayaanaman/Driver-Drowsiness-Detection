import os
import cv2
import winsound
import numpy as np
from keras.models import load_model


#haar cascade files loading

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']         #labeling classes

model = load_model('cnnCat2.h5')        #loading model

path = os.getcwd()                      #current working directory

cap = cv2.VideoCapture(0)               #CAM allocation

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

count = 0                               #flag for match responces
score = 0                               #flag for beep sound
rpred = [99]                            #right eye prediction
lpred = [99]                            #left eye prediction

while (True):
    ret, frame = cap.read()             #frame capture
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #BGR to GRAY

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))          #face detection
    left_eye = leye.detectMultiScale(gray)                  #left eye detection
    right_eye = reye.detectMultiScale(gray)                 #right eye detection

    cv2.rectangle(frame, (0, height - 50), (200, height), (255, 255, 255), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in right_eye:                            #right eye processing
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye)            #prediction
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:                               #left eye processing
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)            #prediction
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):               #matching status of both eyes
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        if (score > 5):

            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)             #saving most sleepy img
            try:
                winsound.Beep(2500, 2000)                   #beep sound fre= 2500, dur= 2sec

            except:                                         # isplaying = False
                pass


    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
