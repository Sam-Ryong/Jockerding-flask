from flask import Flask, request, jsonify
import tensorflow as tf
import cv2 as cv
import numpy as np
import base64

app = Flask(__name__)

face_detection = cv.CascadeClassifier('model/haar_cascade_face_detection.xml')

settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}

input_size = (224, 224)

# Load the model
model = tf.keras.models.load_model('C:\\Users\\Hong Seung Pyo\\model\\keras_model.h5', compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    base64Data = data.get('base64Data', None)
    

    if base64Data:
        binary_data = base64.b64decode(base64Data)
        img = np.frombuffer(binary_data, dtype=np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        
        if img is not None:
            # Resize and preprocess the image
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            model_frame = cv.resize(gray, input_size)
            model_frame = np.expand_dims(model_frame, axis=0) / 255.0
            detected = face_detection.detectMultiScale(gray, **settings)

            if len(detected) == 0:
                return jsonify({'msg_helmet': 0,
                            'graph' : {'Surprise': 0,
                                     'Neutral' : 0,
                                     'Anger' : 0,
                                     'Happy' : 0,
                                     'Sad' : 0
                                     }})
            
            for x, y, w, h in detected:
                cv.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
                cv.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
                face = gray[y+5:y+h-5, x+20:x+w-20]
                face = cv.resize(face, (48,48)) 
                face = face/255.0

            # Predict using the model
            is_helmet_prob = model.predict(np.array([face.reshape((48,48,1))]))[0]
            is_helmet = np.argmax(is_helmet_prob)

            msg_helmet = int(is_helmet_prob[is_helmet] * 100)
    

            return jsonify({'msg_helmet': msg_helmet,
                            'graph' : {'Surprise': is_helmet_prob[0] * 100,
                                     'Neutral' : is_helmet_prob[1] * 100,
                                     'Anger' : is_helmet_prob[2] * 100,
                                     'Happy' : is_helmet_prob[3] * 100,
                                     'Sad' : is_helmet_prob[4] * 100
                                     }
                                     })
        else:
            return jsonify({'msg_helmet': 'Failed to load the image.',
                            'graph' : {'Surprise': 0,
                                     'Neutral' : 0,
                                     'Anger' : 0,
                                     'Happy' : 0,
                                     'Sad' : 0
                                     }
                                     })
    else:
        return jsonify({'msg_helmet': 'Invalid image_path.',
                            'graph' : {'Surprise': 0,
                                     'Neutral' : 0,
                                     'Anger' : 0,
                                     'Happy' : 0,
                                     'Sad' : 0
                                     }
                                     })

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
