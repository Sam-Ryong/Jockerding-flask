import tensorflow as tf
import cv2 as cv
import numpy as np

face_detection = cv.CascadeClassifier('./model/haar_cascade_face_detection.xml')

settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}
size = (224, 224)
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
input_size = (224, 224)

model = tf.keras.models.load_model('C:\\Users\\Hong Seung Pyo\\model\\keras_model.h5', compile=False)
predict = [];

if True:
    
    while True:
        
        frame = capture.read()
        model_frame = cv.resize(frame, size, frame)
        model_frame = np.expand_dims(model_frame, axis=0) / 255.0
        detected = face_detection.detectMultiScale(gray, **settings)
   
        # index : Surprise, Neutral, Anger, Happy, Sad

        if len(detected) == 0:
            predict = [0,0,0,0,0]

        for x, y, w, h in detected:
                cv.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
                cv.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
                face = gray[y+5:y+h-5, x+20:x+w-20]
                face = cv.resize(face, (48,48)) 
                face = face/255.0

        predict = model.predict(np.array([face.reshape((48,48,1))]))[0]
        #dominant = np.argmax(is_helmet_prob)
        
        msg = ""

        msg += " ({:.1f})%".format(predict[0] * 100)
        msg += " ({:.1f})%".format(predict[1] * 100)
        msg += " ({:.1f})%".format(predict[2] * 100)
        msg += " ({:.1f})%".format(predict[3] * 100)
        msg += " ({:.1f})%".format(predict[4] * 100)

        cv.putText(frame, msg_helmet, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

        cv.imshow('Wear a helmet', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break


