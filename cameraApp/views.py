import base64
from PIL import Image

import io
import cv2

from rest_framework import status

import numpy as np

from keras.models import model_from_json
from keras.preprocessing import image

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse


@api_view(["POST"])
def build_frames(request):
    # print(request.body)
    decode = base64.b64decode(request.body)
    data = Image.open(io.BytesIO(decode))
    json_data = open("cameraApp/resources/ferKeras.json").read()
    model = model_from_json(json_data)
    # load weights
    model.load_weights('cameraApp/resources/ferKeras.h5')
    # face_haar_cascade = cv2.CascadeClassifier('cameraApp/resources/haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(np.array(data), cv2.COLOR_BGR2GRAY)
    # print(gray_img)
    # faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    # print(faces_detected)
    roi_gray = cv2.resize(gray_img, (48, 48))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    predictions = model.predict(img_pixels)
    print(predictions)

    # se toma la prediccion creada
    max_index = np.argmax(predictions[0])
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    predicted_emotion = emotions[max_index]
    print(predicted_emotion)
    # for (x, y, w, h) in faces_detected:
    #     cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 255), thickness=2)
    #
    #     roi_gray = gray_img[y:y + w, x:x + h]
    #     roi_gray = cv2.resize(roi_gray, (48, 48))

        # preduiccion de las imagenes

        # se añade el texto de la emocion predicha
        # cv2.putText(data, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return JsonResponse({'emotions': predicted_emotion})

