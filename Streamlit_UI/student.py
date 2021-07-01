from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import streamlit as st
import SessionState
from keras.models import model_from_json
from keras.preprocessing import image
import face_recognition
import datetime
import pandas as pd

session_state = SessionState.get(attendance_data={'date':'','name':'unknown','t_focused':0,'t_distracted':0,'t_total':0, 'Attendance':'P','Quality':'NA'})

def app():
    st.title("Welcome Student")
    # calculate the eye aspect ratio
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_EAR(shape):
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        return ear, leftEyeHull, rightEyeHull

    # calculate lip distance
    def lip_distance(shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        return distance

    # Load model from JSON file
    json_file = open('../Model/model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights and them to model
    model.load_weights('../Model/weights2.h5')

    # using session state to preserve values on re-render
    df = pd.read_csv('../eval.csv')
    
    # variables
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 40
    YAWN_THRESH = 20
    COUNTER = 0
    YAWN_CONSEC_FRAMES = 25
    YCOUNTER = 0
   
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../Model/shape_predictor_68_face_landmarks.dat')
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    st.header("Attendace Quality Analyser")
    run = st.checkbox('Toggle Web Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    atn = st.button("Take Attendance and End Session")
    FRAME_WINDOW2 = st.image([])

    if atn and run:
        # Load present date and time
        now = datetime.datetime.now()
        today = now.day
        month = now.month
            
        # Load images.
        image_1 = face_recognition.load_image_file("../Media/1_Aditi.jpeg")
        image_1_face_encoding = face_recognition.face_encodings(image_1)[0]
            
        image_2 = face_recognition.load_image_file("../Media/2_Dania.jpeg")
        image_2_face_encoding = face_recognition.face_encodings(image_2)[0]

        image_3 = face_recognition.load_image_file("../Media/3_Nandinee.jpeg")
        image_3_face_encoding = face_recognition.face_encodings(image_3)[0]
            
        image_4 = face_recognition.load_image_file("../Media/4_Reeha.jpeg")
        image_4_face_encoding = face_recognition.face_encodings(image_4)[0]
            
        image_5 = face_recognition.load_image_file("../Media/5_Shruti.jpeg")
        image_5_face_encoding = face_recognition.face_encodings(image_5)[0]
                
        # Create arrays of known face encodings and their names
        known_face_encodings = [
            image_1_face_encoding,
            image_2_face_encoding,
            image_3_face_encoding,
            image_4_face_encoding,
            image_5_face_encoding
            ]
        known_face_names = [
            "Aditi",
            "Dania",
            "Nandinee",
            "Reeha",
            "Shruti",
            ]
            
        # Initialize some variables
    
        face_locations = []
        face_encodings = []
        face_names = []
        
        process_this_frame = True
        name = "Unknown"
        (top, right, bottom, left) = (0, 0, 0, 0)
        
        ret, frame = camera.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                #updating student features for evaluation_csv
                session_state.attendance_data['name'] = name
                session_state.attendance_data['Attendance'] = "Present"
                session_state.attendance_data['date'] = str(now.strftime("%d-%m-%Y %H:%M")).split(' ')[0]
                if (session_state.attendance_data['t_focused']/session_state.attendance_data['t_total']>0.6):
                    session_state.attendance_data['Quality'] = "focused"
                else:
                     session_state.attendance_data['Quality'] = "distracted"
        
        face_names.append(name)
        process_this_frame = not process_this_frame
            
        # display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

        # draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # display the resulting image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW2.image(frame)    

        # saving features to csv
        df.loc[len(df.index)] = session_state.attendance_data
        df.to_csv('../eval.csv', index= False)
        st.success('Attendance Recorded for {}'.format(name))
        
    while run:
        ret, frame = camera.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)       

                # draw eye contours
                ear, leftEyeHull, rightEyeHull = calculate_EAR(shape)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # calculate lip distance 
                distance = lip_distance(shape)

                # draw lip contours
                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    
                if distance > YAWN_THRESH:
                    YCOUNTER += 1
                    if YCOUNTER >= YAWN_CONSEC_FRAMES:
                        cv2.putText(frame, "Yawn Alert!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Not Sleepy", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)		
                else:
                    YCOUNTER = 0
                    cv2.putText(frame, "Not Sleepy", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   

                # thresholds and frame counters
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # detect emotion
            faces_detected = face_haar_cascade.detectMultiScale(gray, 1.1, 6, minSize=(150, 150))

            if len(faces_detected) == 0:
                cv2.putText(frame, "NOT IN FRAME!", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for (x, y, w, h) in faces_detected:
                cv2.putText(frame, "You're in the frame", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
                roi_gray = gray[y:y+w, x:x+h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0
                predictions = model.predict(img_pixels)
                max_index = int(np.argmax(predictions))

                emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
                predicted_emotion = emotions[max_index]
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if predicted_emotion in ['Angry','Disgusted','Fear','Sad','Surprise']:
                    session_state.attendance_data['t_distracted'] = session_state.attendance_data['t_distracted']+1
                else:
                    session_state.attendance_data['t_focused'] = session_state.attendance_data['t_focused']+1
                
                session_state.attendance_data['t_total'] = session_state.attendance_data['t_distracted'] + session_state.attendance_data['t_focused']
            
            # display the video
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
    camera.release()