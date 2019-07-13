import cv2 as cv
import dlib
import numpy as np
from time import time
from os import path

from faceAligner import FaceAligner
from agClassifier import AgeGenderClassifier
from store import Store

# Files
weight_file = path.join(
            "./models", "weights_v12-16-1.17-0.9852-0.6100.hdf5")
landmark_file = path.join("./models",'shape_predictor_68_face_landmarks.dat')

# Settings
IMG_SIZE = 128 # Model's input image size
DIVIDER = 1 # Reduce video input size by N
PROCESS_EVERY_NTH = 10 # Process only nth file
AGE_RANGES = ['0-10', '10-20', '20-30', '30-40', '40-50', '60+']


def draw_label(image, point, label, color, font=cv.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv.rectangle(image, (x, y - size[1]), (x + size[0], y), color, cv.FILLED)
    cv.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv.LINE_AA)

def main():
    # Face Detector
    detector = dlib.get_frontal_face_detector()
    # Landmark Detector
    face_landmark_predictor = dlib.shape_predictor(landmark_file)
    # Age/Gender classifier
    model = AgeGenderClassifier(IMG_SIZE, weight_file)()
    # Face Aligner
    face_aligner = FaceAligner(face_landmark_predictor, desiredFaceWidth=IMG_SIZE)
    # Store for keeping labels
    store = Store()


    istream = cv.VideoCapture(0)
    iframe_size = (int(istream.get(cv.CAP_PROP_FRAME_WIDTH)),
                   int(istream.get(cv.CAP_PROP_FRAME_HEIGHT)))

    istream.set(cv.CAP_PROP_FRAME_WIDTH, (int(iframe_size[0] / DIVIDER)))
    istream.set(cv.CAP_PROP_FRAME_HEIGHT, (int(iframe_size[1] / DIVIDER)))
    istream.set(cv.CAP_PROP_FPS, 30)    

    frame_counter = 0
    fps = 1
    start_time = time()

    while(True):
        ret, img = istream.read()
        if not ret:
            break

        fps = frame_counter / (time() - start_time)
        print('Current FPS: {0}'.format(int(fps)))
        
        if frame_counter % PROCESS_EVERY_NTH == 0:
            # Processing image

            input_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            grey_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            
            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), IMG_SIZE, IMG_SIZE, 3))
            
            if len(detected) > 0:
                for i, d in enumerate(detected):
                    face_aligned = face_aligner.align(input_img, grey_img, detected[i])
                    faces[i, :, :, :] = cv.resize(face_aligned, (IMG_SIZE, IMG_SIZE))

                # predict ages and genders of the detected faces
                results = model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 7).reshape(7, 1)
                predicted_ages = results[1].dot(ages).flatten()

                store.update(detected,predicted_genders,predicted_ages)

        # draw results
        detected, predicted_genders, predicted_ages = store.get_detected()

        for i, d in enumerate(detected):
            gender = "M" if predicted_genders[i][0] < 0.5 else "F"
            color = (255, 0, 0) if gender == 'M' else (0, 0, 255)
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = "{}, {} ({:.2f})".format(AGE_RANGES[int(predicted_ages[i])],
                                    gender, np.amax(predicted_genders[i]))
            draw_label(img, (d.left(), d.top()), label, color)
        
        cv.imshow("result", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    istream.release()


main()