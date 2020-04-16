from django.shortcuts import render
import threading

from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
import os
import tensorflow as tf
import cv2
import json
import numpy as np
from django.views.decorators import gzip
from keras.models import model_from_json
from keras.preprocessing import image


# Create your views here.
class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        jsondata = open("cameraApp/resources/ferKeras.json").read()
        model = model_from_json(jsondata)
        # load weights
        model.load_weights('cameraApp/resources/ferKeras.h5')
        face_haar_cascade = cv2.CascadeClassifier('cameraApp/resources/haarcascade_frontalface_default.xml')
        while True:
            (self.grabbed, self.frame) = self.video.read()
            gray_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            for (x, y, w, h) in faces_detected:
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 255), thickness=2)

                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))

                # preduiccion de las imagenes
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = model.predict(img_pixels)

                # se toma la prediccion creada
                max_index = np.argmax(predictions[0])
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                # se añade el texto de la emocion predicha
                cv2.putText(self.frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)


cam = VideoCamera()


def gen(camera):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def open_web_cam(request):
    try:
        return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass
