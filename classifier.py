import tensorflow as tf
import cv2 as cv
import numpy as np
import base64

face_detection = cv.CascadeClassifier('./model/haar_cascade_face_detection.xml')

settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}

capture = cv.VideoCapture(0)
input_size = (224, 224)

model = tf.keras.models.load_model('C:\\Users\\Hong Seung Pyo\\model\\keras_model.h5', compile=False)
predict = [];

if True:
    
    while True:
        
        ret, img = capture.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        model_frame = cv.resize(gray, input_size)
        model_frame = np.expand_dims(model_frame, axis=0) / 255.0
        detected = face_detection.detectMultiScale(gray, **settings)

        # index : Surprise, Neutral, Anger, Happy, Sad

        if len(detected) == 0:
            predict = [0,0,0,0,0]

        else:
            for x, y, w, h in detected:
                cv.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
                cv.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
                face = gray[y+5:y+h-5, x+20:x+w-20]
                face = cv.resize(face, (48,48)) 
                face = face/255.0

            predict = model.predict(np.array([face.reshape((48,48,1))]))[0]
        
        msg = ""

        msg += "Surprise ({:.1f})% ".format(predict[0] * 100)
        msg += "Neutral ({:.1f})% ".format(predict[1] * 100)
        msg += "Anger ({:.1f})% ".format(predict[2] * 100)
        msg += "Happy ({:.1f})% ".format(predict[3] * 100)
        msg += "Sad ({:.1f})% ".format(predict[4] * 100)

        cv.putText(img, msg, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)

        cv.imshow('Wear a helmet', img)

        print(msg)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break


